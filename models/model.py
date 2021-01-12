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

from rouge import Rouge

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
        self.use_pretrained = config.model.use_pretrained
        self.vocab = vocab

        if self.use_pretrained:
            emb_vecs = self.load_embeddings(config.model.pretrained)
            self.embedding = nn.Embedding.from_pretrained(emb_vecs,
                                                          freeze=False,
                                                          padding_idx=self.vocab.pad())
            embed_dim = self.embedding.embedding_dim
        else:
            embed_dim = config.model.args.embed_dim
            self.embedding = nn.Embedding(len(vocab), embed_dim,
                                          padding_idx=self.vocab.pad())
                                          
        hidden_dim = config.model.args.hidden_dim
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

    def load_embeddings(self, which='fasttext'):
        num_oov = 0
        num_in_vocab = 0
        emb_vecs = []
        emb_size = 300
        import fasttext.util
        fasttext.util.download_model('ko', if_exists='ignore')
        ft = fasttext.load_model('cc.ko.300.bin')
        for w in self.vocab._id_to_word:
            if ft.get_word_id(w) == -1:  # out of ft vocab
                w_emb = torch.rand([emb_size, 1])
                nn.init.kaiming_normal_(w_emb, mode='fan_out')
                num_oov += 1
            else:
                w_emb = torch.tensor(ft.get_word_vector(w))
                num_in_vocab += 1
            emb_vecs.append(w_emb)
        emb_vecs = list(map(lambda x: x.squeeze(), emb_vecs))
        emb_vecs = torch.stack(emb_vecs)

        num_total = num_oov + num_in_vocab
        print(f"Loaded embeddings from {which}: {num_in_vocab} out of {num_total} are initialized from {which}")
        return emb_vecs

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
            weighted_attn_dist = (1.0 - p_gen) * attn_dist          # [B x L]

            # Concat some zeros to each vocab dist,
            # to hold probs for oov words that appeared in source text
            batch_size = vocab_dist.size(0)
            extra_zeros = torch.zeros((batch_size, max_oov_len),
                                      device=vocab_dist.device)
            extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)  # [B x V_x]

            final_dist = extended_vocab_dist.scatter_add(dim=-1,
                                                         index=enc_input_ext,
                                                         src=weighted_attn_dist)
            # Save outputs for loss computation
            final_dists.append(final_dist)
            attn_dists.append(attn_dist)
            coverages.append(cov)

        final_dists = torch.stack(final_dists, dim=-1)  # [B x V_x x T]
        attn_dists = torch.stack(attn_dists, dim=-1)    # [B x L x T]
        coverages = torch.stack(coverages, dim=-1)      # [B x L x T]

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
            dec_input = [self.filter_unk(hyp.latest_token) for hyp in hyps]
            dec_input = torch.tensor(dec_input,
                                     dtype=torch.long,
                                     device=enc_input.device)                   # [K]
            dec_emb = self.embedding(dec_input)                                 # [K x E]
            h = torch.cat([hyp.hidden_state for hyp in hyps], dim=0)            # [1 x H] -> [K x H]
            c = torch.cat([hyp.cell_state for hyp in hyps], dim=0)              # [1 x H] -> [K x H]
            coverages = torch.cat([hyp.coverage for hyp in hyps], dim=0)        # [1 x L] -> [K x L]
            enc_hiddens = torch.cat([enc_hidden for _ in hyps], dim=0)          # [1 x L x 2H] -> [K x L x 2H]
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
            weighted_attn_dist = (1.0 - p_gen) * attn_dist          # [K x L]

            # Concat some zeros to each vocab dist,
            # to hold probs for oov words that appeared in source text
            batch_size = vocab_dist.size(0)
            extra_zeros = torch.zeros((batch_size, max_oov_len),
                                      device=vocab_dist.device)
            extended_vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=-1)  # [K x V_x]
            final_dist = extended_vocab_dist.scatter_add(dim=-1,
                                                         index=enc_input_ext,
                                                         src=weighted_attn_dist)

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
                    if topk_ids[i, j].item() == self.vocab.unk():
                        pass
                    else:
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

            if len(results) == self.beam_size:
                break

        # Reached max decode steps but not enough results
        if len(results) < self.num_return_seq:
            results = results + hyps[:self.num_return_seq - len(results)]

        sorted_results = self.sort_hyps(results)
        best_hyps = sorted_results[:self.num_return_seq]

        # Map token ids to words
        hyp_words = [self.vocab.outputids2words(hyp.tokens, src_oovs[0]) for hyp in best_hyps]

        # Concatenate words to strings
        if self.config.model.use_pretrained and self.config.model.pretrained == 'kobert':
            bpe = True
        else:
            bpe = False
        hyp_results = [postprocess(words, bpe=bpe,
                                   skip_special_tokens=True,
                                   clean_up_tokenization_spaces=True)
                       for words in hyp_words]
        results = {'generated_summary': hyp_results}
        return results

    def sort_hyps(self, hyps):
        """Sort hypotheses according to their log probability."""
        return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)

    def filter_unk(self, idx):
        return idx if idx < self.vocab.size() else self.vocab.unk()


class SummarizationModel(pl.LightningModule):
    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.model = PointerGenerator(config, vocab)
        self.criterion = Loss(args=config.loss.args)
        self.num_step = 0
        self.cov_weight = config.loss.args.cov_weight
        self.rouge = Rouge()

    def training_step(self, batch, batch_idx):
        output = self.model.forward(enc_input=batch.enc_input,
                                    enc_input_ext=batch.enc_input_ext,
                                    enc_pad_mask=batch.enc_pad_mask,
                                    enc_len=batch.enc_len,
                                    dec_input=batch.dec_input,
                                    max_oov_len=batch.max_oov_len)

        nll_loss, cov_loss = self.criterion(output=output,
                                            batch=batch)
        loss = nll_loss + self.cov_weight * cov_loss
        # self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.logger.log_metrics({'train_loss': loss,
                                 'train/nll_loss': nll_loss}, self.num_step)
        if self.cov_weight > 0:
            self.logger.log_metrics({'train/cov_loss': cov_loss}, self.num_step)
        self.num_step += 1
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.model.forward(enc_input=batch.enc_input,
                                    enc_input_ext=batch.enc_input_ext,
                                    enc_pad_mask=batch.enc_pad_mask,
                                    enc_len=batch.enc_len,
                                    dec_input=batch.dec_input,
                                    max_oov_len=batch.max_oov_len)
        nll_loss, cov_loss = self.criterion(output=output,
                                            batch=batch)
        loss = nll_loss + self.cov_weight * cov_loss
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.logger.log_metrics({'val_loss': loss}, self.num_step)

        result = self.test_step(batch, batch_idx)
        scores = self.rouge.get_scores(result['generated_summary'],
                                       result['gold_summary'], avg=True)
        rouge_1 = scores['rouge-1']['f'] * 100.0
        rouge_2 = scores['rouge-2']['f'] * 100.0
        rouge_l = scores['rouge-l']['f'] * 100.0
        pred = {
            'rouge_1': rouge_1,
            'rouge_2': rouge_2,
            'rouge_l': rouge_l,
            'val_loss': loss
        }
        return pred

    def validation_epoch_end(self, validation_step_outputs):
        rouge_1, rouge_2, rouge_l, val_loss = [], [], [], []
        for pred in validation_step_outputs:
            rouge_1.append(pred['rouge_1'])
            rouge_2.append(pred['rouge_2'])
            rouge_l.append(pred['rouge_l'])
            val_loss.append(pred['val_loss'])
        rouge_1_avg = sum(rouge_1) / len(rouge_1)
        rouge_2_avg = sum(rouge_2) / len(rouge_2)
        rouge_l_avg = sum(rouge_l) / len(rouge_l)
        val_loss_avg = sum(val_loss) / len(val_loss)
        results = {
            'rouge_1_avg': rouge_1_avg,
            'rouge_2_avg': rouge_2_avg,
            'rouge_l_avg': rouge_l_avg,
            'val_loss_avg': val_loss_avg,
        }
        return results

    def test_step(self, batch, batch_idx):
        result = self.model.inference(enc_input=batch.enc_input,
                                      enc_input_ext=batch.enc_input_ext,
                                      enc_pad_mask=batch.enc_pad_mask,
                                      enc_len=batch.enc_len,
                                      src_oovs=batch.src_oovs,
                                      max_oov_len=batch.max_oov_len)
        result['source'] = [''.join(w) for w in batch.src_text]
        result['gold_summary'] = [''.join(w) for w in batch.tgt_text]
        return result

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
