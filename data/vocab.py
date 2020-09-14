# Most of this file is copied from
# https://github.com/abisee/pointer-generator/blob/master/data.py
# https://github.com/atulkum/pointer_summarizer/blob/master/data_util/data.py

import json

PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNK_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences


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

