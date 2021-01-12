from collections import Counter

import gluonnlp as nlp
import torch
from konlpy.tag import Mecab
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data.utils import collate_tokens
from data.utils import load_dataset
from data.vocab import Vocab


class TextDataset(Dataset):
    """
    Args:
        txt: list of text samples
        max_len: max sequence length
    """

    def __init__(self, txt, max_len):
        self.tokenizer = self.load_tokenizer('mecab')
        self.dataset = self.build_dataset(txt)
        self.max_len = max_len
        self.vocab = None

    def __getitem__(self, index):
        txt = self.dataset[index]
        tokens = self.tokenize(txt)
        tokens = self.truncate(tokens)
        length = len(tokens)
        return tokens, length

    def __len__(self):
        return len(self.dataset)

    def build_vocab(self, vocab_size, min_freq, specials):
        counter = Counter()
        for t in self.dataset:
            tokens = self.tokenize(t)
            counter.update(tokens)
        vocab = Vocab.from_counter(counter=counter,
                                   vocab_size=vocab_size,
                                   min_freq=min_freq,
                                   specials=specials)
        return vocab

    def build_dataset(self, txt):
        txt = list(map(self.preprocess, txt))
        return txt

    def load_tokenizer(self, which):
        """Loads mecab tokenizer"""
        m = Mecab()
        tokenizer = m.morphs
        return tokenizer

    def preprocess(self, text):
        # TODO: add preprocessing functions relevant to korean & our dataset
        text = text.lower()
        return text

    def truncate(self, tokens):
        if len(tokens) > self.max_len:
            return tokens[:self.max_len]
        else:
            return tokens

    def tokenize(self, text):
        """Converts text string to list of tokens"""
        tokens = self.tokenizer(text)
        return tokens


class SummDataset(Dataset):
    """
    Args:
        src: (TextDataset) source dataset
        tgt: (TextDataset) target dataset
    """

    def __init__(self, vocab, src, tgt=None):
        if tgt is not None:
            assert len(src) == len(tgt), "Source and target must contain the same number of examples"
        self.src, self.tgt = src, tgt
        self.vocab = vocab

    def __getitem__(self, index):
        src, src_len = self.src[index]
        if self.tgt is not None:
            tgt, tgt_len = self.tgt[index]
        else:
            tgt, tgt_len = None, None
        return src, src_len, tgt, tgt_len

    def __len__(self):
        """Returns size of the dataset"""
        return len(self.src)


class Batch:
    def __init__(self, data, vocab, max_decode):
        src, src_len, tgt, tgt_len = list(zip(*data))
        self.vocab = vocab
        self.pad_id = self.vocab.pad()
        self.max_decode = max_decode

        # Encoder info
        self.enc_input, self.enc_len, self.enc_pad_mask = None, None, None
        # Additional info for pointer-generator network
        self.enc_input_ext, self.max_oov_len, self.src_oovs = None, None, None
        # Decoder info
        self.dec_input, self.dec_target, self.dec_len, self.dec_pad_mask = None, None, None, None

        # Build batch inputs
        self.init_encoder_seq(src, src_len)
        self.init_decoder_seq(tgt, tgt_len)

        # Save original strings
        self.src_text = src
        self.tgt_text = tgt

    def init_encoder_seq(self, src, src_len):
        src_ids = [self.vocab.tokens2ids(s) for s in src]

        self.enc_input = collate_tokens(values=src_ids,
                                        pad_idx=self.pad_id)
        self.enc_len = torch.LongTensor(src_len)
        self.enc_pad_mask = (self.enc_input == self.pad_id)

        # Save additional info for pointer-generator
        # Determine max number of source text OOVs in this batch
        src_ids_ext, oovs = zip(*[self.vocab.source2ids_ext(s) for s in src])
        # Store the version of the encoder batch that uses article OOV ids
        self.enc_input_ext = collate_tokens(values=src_ids_ext,
                                            pad_idx=self.pad_id)
        self.max_oov_len = max([len(oov) for oov in oovs])
        # Store source text OOVs themselves
        self.src_oovs = oovs

    def init_decoder_seq(self, tgt, tgt_len):
        tgt_ids = [self.vocab.tokens2ids(t) for t in tgt]
        tgt_ids_ext = [self.vocab.target2ids_ext(t, oov) for t, oov in zip(tgt, self.src_oovs)]

        # create decoder inputs
        dec_input, _ = zip(*[self.get_decoder_input_target(t, self.max_decode) for t in tgt_ids])

        self.dec_input = collate_tokens(values=dec_input,
                                        pad_idx=self.pad_id,
                                        pad_to_length=self.max_decode)

        # create decoder targets using extended vocab
        _, dec_target = zip(*[self.get_decoder_input_target(t, self.max_decode) for t in tgt_ids_ext])

        self.dec_target = collate_tokens(values=dec_target,
                                         pad_idx=self.pad_id,
                                         pad_to_length=self.max_decode)

        self.dec_len = torch.LongTensor(tgt_len)
        self.dec_pad_mask = (self.dec_input == self.pad_id)

    def get_decoder_input_target(self, tgt, max_len):
        dec_input = [self.vocab.start()] + tgt
        dec_target = tgt + [self.vocab.stop()]
        # truncate inputs longer than max length
        if len(dec_input) > max_len:
            dec_input = dec_input[:max_len]
            dec_target = dec_target[:max_len]
        assert len(dec_input) == len(dec_target)
        return dec_input, dec_target

    def __len__(self):
        return self.enc_input.size(0)

    def __str__(self):
        batch_info = {
            'src_text': self.src_text,
            'tgt_text': self.tgt_text,
            'enc_input': self.enc_input,  # [B x L]
            'enc_input_ext': self.enc_input_ext,  # [B x L]
            'enc_len': self.enc_len,  # [B]
            'enc_pad_mask': self.enc_pad_mask,  # [B x L]
            'src_oovs': self.src_oovs,  # list of length B
            'max_oov_len': self.max_oov_len,  # single int value
            'dec_input': self.dec_input,  # [B x T]
            'dec_target': self.dec_target,  # [B x T]
            'dec_len': self.dec_len,  # [B]
            'dec_pad_mask': self.dec_pad_mask,  # [B x T]
        }
        return str(batch_info)

    def to(self, device):
        self.enc_input = self.enc_input.to(device)
        self.enc_input_ext = self.enc_input_ext.to(device)
        self.enc_len = self.enc_len.to(device)
        self.enc_pad_mask = self.enc_pad_mask.to(device)
        self.dec_input = self.dec_input.to(device)
        self.dec_target = self.dec_target.to(device)
        self.dec_len = self.dec_len.to(device)
        self.dec_pad_mask = self.dec_pad_mask.to(device)
        return self


def build_dataset(data_path, config, is_train, vocab=None, load_vocab=None):
    args = config.data
    if is_train:
        src_txt, tgt_txt = load_dataset(data_path)
        src_train = TextDataset(src_txt, args.src_max_train)
        tgt_train = TextDataset(tgt_txt, args.tgt_max_train)
        if load_vocab is not None:
            vocab = Vocab.from_json(load_vocab)
        else:
            vocab = src_train.build_vocab(vocab_size=args.vocab_size,
                                          min_freq=args.vocab_min_freq,
                                          specials=[PAD_TOKEN,
                                                    UNK_TOKEN,
                                                    START_DECODING,
                                                    STOP_DECODING])
        dataset = SummDataset(src=src_train,
                              tgt=tgt_train,
                              vocab=vocab)
        return dataset, vocab

    else:
        assert vocab is not None
        src_txt, tgt_txt = load_dataset(data_path)
        src_test = TextDataset(src_txt, args.src_max_test)
        tgt_test = TextDataset(tgt_txt, args.tgt_max_test)
        dataset = SummDataset(src=src_test,
                              tgt=tgt_test,
                              vocab=vocab)
        return dataset


def build_dataloader(dataset, vocab, batch_size, max_decode, is_train, num_workers):
    shuffle = True if is_train else False
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             collate_fn=lambda data, v=vocab, t=max_decode: Batch(data=data,
                                                                                  vocab=v,
                                                                                  max_decode=t),
                             num_workers=num_workers)
    return data_loader
