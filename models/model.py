import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.optim import Adagrad

from decode import Hypothesis
from decode import postprocess
from models.loss import Loss

"""
: Models
:   1) Encoder
:   2) Attention module
:   3) Decoder w/ attention
:   4) Pointer-generator network
"""


class Encoder(nn.Module):
    """
    Single-layer bidirectional LSTM
    B : batch size
    E : embedding size
    H : encoder hidden state dimension
    L : sequence length
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.reduce_h = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
        self.reduce_c = nn.Linear(hidden_dim * 2, hidden_dim, bias=True)

    def forward(self, src, src_lens):
        """
        Args:
            src: source token embeddings    [B x L x E]
            src_lens: source text length    [B]

        Returns:
            enc_hidden: sequence of encoder hidden states                  [B x L x 2H]
            (final_h, final_c): Tuple for decoder state initialization     [B x L x H]
        """

        # Pack the sequence into a PackedSequence object to feed to the LSTM.
        x = pack_padded_sequence(src, src_lens, batch_first=True, enforce_sorted=False)

        # Get outputs from the LSTM
        output, (h, c) = self.lstm(x)  # [B x L x 2H], [2 x B x H], [2 x B x H]
        enc_hidden, _ = pad_packed_sequence(output, batch_first=True)

        # Concatenate bidirectional lstm states
        h = torch.cat((h[0], h[1]), dim=-1)  # [B x 2H]
        c = torch.cat((c[0], c[1]), dim=-1)  # [B x 2H]

        # Project to decoder hidden state size
        final_hidden = torch.relu(self.reduce_h(h))  # [B x H]
        final_cell = torch.relu(self.reduce_c(c))  # [B x H]

        return enc_hidden, (final_hidden, final_cell)


class Attention(nn.Module):
    """
    Attention mechanism based on Bahdanau et al. (2015) - Eq. (1)(2)
    augmented with Coverage mechanism - Eq. (11)
    B : batch size
    L : source text length
    H : encoder hidden state dimension
    """

    def __init__(self, hidden_dim, use_coverage):
        super().__init__()
        # Eq. (1)
        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)                       # v
        self.enc_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)   # W_h
        self.dec_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=True)        # W_s, b_attn

        self.use_coverage = use_coverage
        if self.use_coverage:
            # Additional parameter for coverage vector; w_c in Eq. (11)
            self.w_c = nn.Linear(1, hidden_dim * 2, bias=False)

    def forward(self, dec_input, coverage, enc_hidden, enc_pad_mask):
        """
        Args:
            dec_input: decoder hidden state             [B x H]
            coverage: coverage vector                   [B x L]
            enc_hidden: encoder hidden states           [B x L x 2H]
            enc_pad_mask: encoder padding masks         [B x L]

        Returns:
            attn_dist: attention dist'n over src tokens [B x L]
        """

        # Eq. (1)
        enc_feature = self.enc_proj(enc_hidden)         # [B x L x 2H]
        dec_feature = self.dec_proj(dec_input)          # [B x 2H]
        dec_feature = dec_feature.unsqueeze(1)          # [B x 1 x 2H]
        scores = enc_feature + dec_feature              # [B x L x 2H]

        if self.use_coverage:
            # Eq. (11)
            coverage = coverage.unsqueeze(-1)           # [B x L x 1]
            cov_feature = self.w_c(coverage)            # [B x L x 2H]
            scores = scores + cov_feature

        scores = torch.tanh(scores)                     # [B x L x 2H]
        scores = self.v(scores)                         # [B x L x 1]
        scores = scores.squeeze(-1)                     # [B x L]

        # Don't attend over padding; fill '-inf' where enc_pad_mask == True
        if enc_pad_mask is not None:
            scores = scores.float().masked_fill_(
                enc_pad_mask,
                float('-inf')
            ).type_as(scores)  # FP16 support: cast to float and back

        # Eq. (2)
        attn_dist = F.softmax(scores, dim=-1)               # [B x L]

        return attn_dist


class AttnDecoder(nn.Module):
    """
    Single-layer unidirectional LSTM with attention for a single timestep - Eq. (3)(4)
    B : batch size
    E : embedding size
    H : decoder hidden state dimension
    V : vocab size
    """

    def __init__(self, input_dim, hidden_dim, vocab_size, use_coverage):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim)
        self.attention = Attention(hidden_dim, use_coverage)
        # Eq. (4)
        self.v = nn.Linear(hidden_dim * 3, hidden_dim, bias=True)   # V, b
        self.v_out = nn.Linear(hidden_dim, vocab_size, bias=True)   # V', b'

    def forward(self, dec_input, prev_h, prev_c, enc_hidden, enc_pad_mask, coverage):
        """
        Args:
            dec_input: decoder input embedding at timestep t    [B x E]
            prev_h: decoder hidden state from prev timestep     [B x H]
            prev_c: decoder cell state from prev timestep       [B x H]
            enc_hidden: encoder hidden states                   [B x L x 2H]
            enc_pad_mask: encoder masks for attn computation    [B x L]
            coverage: coverage vector at timestep t - Eq. (10)  [B x L]

        Returns:
            vocab_dist: predicted vocab dist'n at timestep t    [B x V]
            attn_dist: attention dist'n at timestep t           [B x L]
            context_vec: context vector at timestep t           [B x 2H]
            hidden: hidden state at timestep t                  [B x H]
            cell: cell state at timestep t                      [B x H]
        """

        # Get this step's decoder hidden state
        hidden, cell = self.lstm(dec_input, (prev_h, prev_c))   # [B x H], [B x H]

        # Compute attention distribution over enc states
        attn_dist = self.attention(dec_input=hidden,
                                   coverage=coverage,
                                   enc_hidden=enc_hidden,
                                   enc_pad_mask=enc_pad_mask)   # [B x L]

        # Eq. (3) - Sum weighted enc hidden states to make context vector
        # The context vector is used later to compute generation probability
        context_vec = torch.bmm(attn_dist.unsqueeze(1), enc_hidden)     # [B x 1 x 2H]
        context_vec = torch.sum(context_vec, dim=1)                     # [B x 2H]

        # Eq. (4)
        output = self.v(torch.cat([hidden, context_vec], dim=-1))       # [B x 3H] -> [B x H]
        output = self.v_out(output)                                     # [B x V]
        vocab_dist = F.softmax(output, dim=-1)                          # [B x V]
        return vocab_dist, attn_dist, context_vec, hidden, cell


class PointerGenerator(nn.Module):
    """
        2.2. Pointer-generator network
        - Computes generation probability p_gen             - Eq. (8)
        - Computes prob dist'n over extended vocabulary     - Eq. (9)
        2.3. Coverage mechanism
        - Computes coverage vector for coverage loss        - Eq. (10)

        B : batch size
        E : decoder embedding size
        H : encoder, decoder hidden state dimension
        L : source text length
        V : vocab size
        V_x : extended vocab size
    """

    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab

        embed_dim = config.model.args.embed_dim
        hidden_dim = config.model.args.hidden_dim
        self.embedding = nn.Embedding(len(vocab), embed_dim,
                                      padding_idx=self.vocab.pad())
        self.encoder = Encoder(input_dim=embed_dim,
                               hidden_dim=hidden_dim)
        self.decoder = AttnDecoder(input_dim=embed_dim,
                                   hidden_dim=hidden_dim,
                                   vocab_size=len(vocab),
                                   use_coverage=config.loss.args.use_coverage)

        # Parameters specific to PGN - Eq. (8)
        self.w_h = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.w_s = nn.Linear(hidden_dim, 1, bias=False)
        self.w_x = nn.Linear(embed_dim, 1, bias=True)

        # Hyper-parameters used during decoding at inference time
        self.beam_size = config.decode.args.beam_size
        self.min_dec_steps = config.decode.args.min_dec_steps
        self.num_return_seq = config.decode.args.num_return_seq

    def forward(self, enc_input, enc_input_ext, enc_pad_mask, enc_len,
                dec_input, max_oov_len):
        """
        Predict summary using reference summary as decoder inputs (teacher forcing)
        Args:
            enc_input: source text id sequence                      [B x L]
            enc_input_ext: source text id seq w/ extended vocab     [B x L]
            enc_pad_mask: source text padding mask. [PAD] -> True   [B x L]
            enc_len: source text length                             [B]
            dec_input: target text id sequence                      [B x T]
            max_oov_len: max number of oovs in src                  [1]

        Returns:
            final_dists: predicted dist'n using extended vocab      [B x V_x x T]
            attn_dists: attn dist'n from each t                     [B x L x T]
            coverages: coverage vectors from each t                 [B x L x T]
        """

        # Build source text representations from encoder
        enc_emb = self.embedding(enc_input)                         # [B x L x E]
        enc_hidden, (h, c) = self.encoder(enc_emb, enc_len)         # [B x L x 2H]

        # Outputs required for loss computation
        # 1. cross-entropy (negative log-likelihood) loss - Eq. (6)
        final_dists = []

        # 2. coverage loss - Eq. (12)
        attn_dists = []
        coverages = []

        # Initialize decoder inputs
        dec_emb = self.embedding(dec_input)                         # [B x T x E]
        cov = torch.zeros_like(enc_input).float()                   # [B x L]

        for t in range(self.config.data.tgt_max_train):
            input_t = dec_emb[:, t, :]  # Decoder input at this timestep
            vocab_dist, attn_dist, context_vec, h, c = self.decoder(dec_input=input_t,
                                                                    prev_h=h,
                                                                    prev_c=c,
                                                                    enc_hidden=enc_hidden,
                                                                    enc_pad_mask=enc_pad_mask,
                                                                    coverage=cov)
            # Eq. (10) - Compute coverage vector;
            # sum of attn dist over all prev decoder timesteps
            cov = cov + attn_dist

            # Eq. (8) - Compute generation probability p_gen
            context_feat = self.w_h(context_vec)                    # [B x 1]
            decoder_feat = self.w_s(h)                              # [B x 1]
            input_feat = self.w_x(input_t)                          # [B x 1]
            gen_feat = context_feat + decoder_feat + input_feat
            p_gen = torch.sigmoid(gen_feat)                         # [B x 1]

            # Eq. (9) - Compute prob dist'n over extended vocabulary
            vocab_dist = p_gen * vocab_dist                         # [B x V]
            attn_dist = (1.0 - p_gen) * attn_dist                   # [B x L]

            # Concat some zeros to each vocab dist,
            # to hold probs for oov words that appeared in source text
            batch_size = vocab_dist.size(0)
            extra_zeros = torch.zeros((batch_size, max_oov_len),
                                      device=vocab_dist.device)
            extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)  # [B x V_x]

            final_dist = extended_vocab_dist.scatter_add(dim=-1,
                                                         index=enc_input_ext,
                                                         src=attn_dist)
            # Save outputs for loss computation
            final_dists.append(final_dist)
            attn_dists.append(attn_dist)
            coverages.append(cov)

        final_dists = torch.stack(final_dists, dim=-1)  # [B x V_x x T]
        attn_dists = torch.stack(attn_dists, dim=-1)  # [B x L x T]
        coverages = torch.stack(coverages, dim=-1)  # [B x L x T]

        return {
            'final_dist': final_dists,
            'attn_dist': attn_dists,
            'coverage': coverages
        }

    def inference(self, enc_input, enc_input_ext, enc_pad_mask, enc_len,
                  src_oovs, max_oov_len):
        """
        Predict summary using previous timestep's decoder output as this step's decoder input + beam search
        Args:
            enc_input: source text id sequence                      [B x L]
            enc_input_ext: source text id seq w/ extended vocab     [B x L]
            enc_pad_mask: source text padding mask. [PAD] -> True   [B x L]
            enc_len: source text length                             [B]
            src_oovs: list of source text oovs of each sample       [B]
            max_oov_len: max number of oovs in src                  [1]

        Returns:
            results: dictionary with 'generated_summary'
        """
        # Build source text representation from encoder
        enc_emb = self.embedding(enc_input)  # [B x L x E]
        enc_hidden, (h, c) = self.encoder(enc_emb, enc_len)  # [B x L x 2H]

        # Initialize decoder input
        cov = torch.zeros_like(enc_input).float()  # [B x L]

        # Initialize hypotheses
        batch_size = enc_input.size(0)
        # All samples start with a single hypothesis ([START])
        hyps = [
            Hypothesis(tokens=[self.vocab.start()],
                       log_probs=[0.0],
                       hidden_state=h,
                       cell_state=c,
                       coverage=cov)
            for _ in range(batch_size)
        ]
        results = []  # finished hypotheses (those that have emitted the [STOP] token)

        for steps in range(self.config.data.tgt_max_test):
            # Prepare decoder inputs (= previously generated tokens) for this step
            # K : number of hypotheses (we want top-K outputs)
            dec_input = [h.latest_token for h in hyps]
            dec_input = torch.tensor(dec_input,
                                     dtype=torch.long,
                                     device=self.device)                    # [K]
            dec_emb = self.embedding(dec_input)                             # [K x E]
            h = torch.cat([h.hidden_state for h in hyps], dim=0)            # [1 x H] -> [K x H]
            c = torch.cat([h.cell_state for h in hyps], dim=0)              # [1 x H] -> [K x H]
            coverages = torch.cat([h.coverage for h in hyps], dim=0)        # [1 x L] -> [K x L]
            enc_hiddens = torch.cat([enc_hidden for _ in hyps], dim=0)      # [1 x L x 2H] -> [K x L x 2H]
            enc_pad_masks = torch.cat([enc_pad_mask for _ in hyps], dim=0)

            vocab_dist, attn_dist, context_vec, h, c = self.decoder(dec_input=dec_emb,
                                                                    prev_h=h, prev_c=c,
                                                                    enc_hidden=enc_hiddens,
                                                                    enc_pad_mask=enc_pad_masks,
                                                                    coverage=coverages)

            # Eq. (10) - Compute coverage vector;
            # sum of attn dist over all prev decoder timesteps
            cov = cov + attn_dist

            # Eq. (8) - Compute generation probability p_gen
            context_feat = self.w_h(context_vec)                    # [K x 1]
            decoder_feat = self.w_s(h)                              # [K x 1]
            input_feat = self.w_x(dec_emb)                          # [K x 1]
            gen_feat = context_feat + decoder_feat + input_feat
            p_gen = torch.sigmoid(gen_feat)                         # [K x 1]

            # Eq. (9) - Compute prob dist'n over extended vocabulary
            vocab_dist = p_gen * vocab_dist                         # [K x V]
            attn_dist = (1.0 - p_gen) * attn_dist                   # [K x L]

            # Concat some zeros to each vocab dist,
            # to hold probs for oov words that appeared in source text
            batch_size = vocab_dist.size(0)
            extra_zeros = torch.zeros((batch_size, max_oov_len),
                                      device=vocab_dist.device)
            extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)  # [K x V_x]
            final_dist = extended_vocab_dist.scatter_add(dim=-1,
                                                         index=enc_input_ext,
                                                         src=attn_dist)

            # Find top-2k most probable token ids and update hypotheses
            log_probs = torch.log(final_dist)
            topk_probs, topk_ids = torch.topk(log_probs,
                                              k=self.beam_size * 2,
                                              dim=-1)

            all_hyps = []
            num_orig_hyps = 1 if steps == 0 else len(hyps)
            for i in range(num_orig_hyps):
                h_i = hyps[i]
                hidden_state_i = h[i].unsqueeze(0)
                cell_state_i = c[i].unsqueeze(0)
                coverage_i = cov[i].unsqueeze(0)

                for j in range(self.beam_size * 2):
                    # Update existing hypothesis with predicted token
                    new_hyp = h_i.extend(token=topk_ids[i, j].item(),
                                         log_prob=log_probs[i, j].item(),
                                         hidden_state=hidden_state_i,
                                         cell_state=cell_state_i,
                                         coverage=coverage_i)
                    all_hyps.append(new_hyp)

            # Find k most probable hypotheses among 2k candidates
            hyps = []
            for h in self.sort_hyps(all_hyps):
                if h.latest_token == self.vocab.stop():
                    if steps >= self.min_dec_steps:
                        results.append(h)
                else:
                    # save for next step
                    hyps.append(h)
                if len(hyps) == self.beam_size or len(results) == self.beam_size:
                    break

        # Reached max decode steps but not enough results
        if len(results) < self.num_return_seq:
            results = results + hyps[:self.num_return_seq - len(results)]

        sorted_results = self.sort_hyps(results)
        best_hyps = sorted_results[:self.num_return_seq]

        # Map token ids to words
        hyp_words = [self.vocab.outputids2words(h.tokens, src_oovs) for h in best_hyps]

        # Concatenate words to strings
        hyp_results = [postprocess(words,
                                   skip_special_tokens=True,
                                   clean_up_tokenization_spaces=True)
                       for words in hyp_words]
        results = {'generated_summary': hyp_results}
        return results

    def sort_hyps(self, hyps):
        """Sort hypotheses according to their log probability."""
        return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)


class SummarizationModel(pl.LightningModule):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        if config.model.type == 'PointerGenerator':
            self.model = PointerGenerator(config, vocab)
        self.criterion = Loss(args=config.loss.args)

    def training_step(self, batch, batch_idx):
        output = self.model.forward(enc_input=batch.enc_input,
                                    enc_input_ext=batch.enc_input_ext,
                                    enc_pad_mask=batch.enc_pad_mask,
                                    enc_len=batch.enc_len,
                                    dec_input=batch.dec_input,
                                    max_oov_len=batch.max_oov_len)
        loss = self.criterion(output=output,
                              batch=batch)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        return result

    def validation_step(self, batch, batch_idx):
        output = self.model.forward(enc_input=batch.enc_input,
                                    enc_input_ext=batch.enc_input_ext,
                                    enc_pad_mask=batch.enc_pad_mask,
                                    enc_len=batch.enc_len,
                                    dec_input=batch.dec_input,
                                    max_oov_len=batch.max_oov_len)
        loss = self.criterion(output=output,
                              batch=batch)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        return result

    def test_step(self, batch, batch_idx):
        predicted_summary = self.model.inference(enc_input=batch.enc_input,
                                                 enc_input_ext=batch.enc_input_ext,
                                                 enc_pad_mask=batch.enc_pad_mask,
                                                 enc_len=batch.enc_len,
                                                 src_oovs=batch.src_oovs,
                                                 max_oov_len=batch.max_oov_len)
        return predicted_summary

    def configure_optimizers(self):
        args = self.config.optimizer.args
        lr = args.lr
        lr_init_accum = args.lr_init_accum
        params = self.parameters()
        optimizer = Adagrad(
            params=params,
            lr=lr,
            initial_accumulator_value=lr_init_accum,
        )
        return optimizer
