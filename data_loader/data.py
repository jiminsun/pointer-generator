# Most of this file is copied from
# https://github.com/abisee/pointer-generator/blob/master/data.py
# https://github.com/atulkum/pointer_summarizer/blob/master/data_util/data.py

import json
from collections import Counter

import gluonnlp as nlp
import torch
from konlpy.tag import Mecab
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from data_loader.utils import collate_tokens
from data_loader.utils import load_dataset
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

# TODO: add special tokens to BERT & KoBERT vocab
# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences


# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):
    """
    Vocabulary class for mapping between words and ids (integers)
    """

    def __init__(self):
        self._word_to_id = {}
        self._id_to_word = []
        self._count = 0

    @classmethod
    def from_json(cls, vocab_file):
        vocab = cls()
        with open(vocab_file, 'r') as f:
            vocab._word_to_id = json.load(f)
        vocab._id_to_word = [w for w, id_ in sorted(vocab._word_to_id,
                                                    key=vocab._word_to_id.get,
                                                    reverse=True)]
        vocab._count = len(vocab._id_to_word)
        vocab.specials = filter(vocab._id_to_word)
        return vocab

    @classmethod
    def from_counter(cls, counter, vocab_size, specials, min_freq):
        vocab = cls()
        word_and_freq = sorted(counter.items(), key=lambda tup: tup[0])
        word_and_freq.sort(key=lambda tup: tup[1], reverse=True)

        for w in specials:
            vocab._word_to_id[w] = vocab._count
            vocab._id_to_word += [w]
            vocab._count += 1

        for word, freq in word_and_freq:
            if freq < min_freq or vocab._count == vocab_size:
                break
            vocab._word_to_id[word] = vocab._count
            vocab._id_to_word += [word]
            vocab._count += 1
        return vocab

    def save(self, fpath):
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(self._word_to_id, f, ensure_ascii=False, indent=4)
        print(f'vocab file saved as {fpath}')

    def __len__(self):
        """Returns size of the vocabulary."""
        return self._count

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        unk_id = self.unk()
        return self._word_to_id.get(word, unk_id)

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError(f'Id not found in vocab: {word_id}')
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary."""
        return self._count

    def pad(self):
        """Helper to get index of pad symbol"""
        return self._word_to_id[PAD_TOKEN]

    def unk(self):
        """Helper to get index of unk symbol"""
        return self._word_to_id[UNK_TOKEN]

    def start(self):
        return self._word_to_id[START_DECODING]

    def stop(self):
        return self._word_to_id[STOP_DECODING]

    def extend(self, oovs):
        extended_vocab = self._id_to_word + list(oovs)
        return extended_vocab

    def tokens2ids(self, tokens):
        ids = [self.word2id(t) for t in tokens]
        return ids

    def source2ids_ext(self, src_tokens):
        """Maps source tokens to ids if in vocab, extended vocab ids if oov.

        Args:
            src_tokens: list of source text tokens

        Returns:
            ids: list of source text token ids
            oovs: list of oovs in source text
        """
        ids = []
        oovs = []
        for t in src_tokens:
            t_id = self.word2id(t)
            unk_id = self.word2id(UNK_TOKEN)
            if t_id == unk_id:
                if t not in oovs:
                    oovs.append(t)
                ids.append(self.size() + oovs.index(t))
            else:
                ids.append(t_id)
        return ids, oovs

    def target2ids_ext(self, tgt_tokens, oovs):
        """Maps target text to ids, using extended vocab (vocab + oovs).

        Args:
            tgt_tokens: list of target text tokens
            oovs: list of oovs from source text (copy mechanism)

        Returns:
            ids: list of target text token ids
        """
        ids = []
        for t in tgt_tokens:
            t_id = self.word2id(t)
            unk_id = self.word2id(UNK_TOKEN)
            if t_id == unk_id:
                if t in oovs:
                    ids.append(self.size() + oovs.index(t))
                else:
                    ids.append(unk_id)
            else:
                ids.append(t_id)
        return ids

    def outputids2words(self, ids, src_oovs):
        """Maps output ids to words

        Args:
            ids: list of ids
            src_oovs: list of oov words

        Returns:
            words: list of words mapped from ids

        """
        words = []
        extended_vocab = self.extend(src_oovs)
        for i in ids:
            try:
                w = self.id2word(i)  # might be oov
            except ValueError as e:
                assert src_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary."
                try:
                    w = extended_vocab[i]
                except IndexError as e:
                    raise ValueError(f'Error: model produced word ID {i} \
                                       but this example only has {len(src_oovs)} article OOVs')
            words.append(w)
        return words


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
        """Loads tokenizer from KoBERT (wordpiece) or mecab"""
        if which == 'kobert':
            _, vocab = get_pytorch_kobert_model()
            tok = get_tokenizer()
            tokenizer = nlp.data.BERTSPTokenizer(tok, vocab, lower=False)
        elif which == 'mecab':
            m = Mecab()
            tokenizer = m.morphs
        else:
            raise ValueError('tokenizer should either be kobert or mecab')
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
