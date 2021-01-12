import os
import pickle
import torch
import random
import json
import pandas as pd
from tqdm import tqdm


def load_dataset(data_dir):
    print(f"Loading dataset from {data_dir}")
    if data_dir.endswith('.json'):
        src, tgt = load_json(data_dir)
    elif data_dir.endswith('.txt'):
        with open(data_dir, 'r') as f:
            data = [l.split('\t') for l in f.readlines()]
        src, tgt = list(zip(*data))
    elif data_dir.endswith('.csv'):
        src, tgt = load_csv(data_dir)
    else:
        src, tgt = None, None
    return src, tgt


def split_pkl(data_path):
    data_dir = os.path.dirname(data_path)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    news_keys = list(data.keys())
    random.shuffle(news_keys)

    train_keys = news_keys[150:]
    dev_keys = news_keys[:100]
    test_keys = news_keys[100:150]

    train_dict = {k: data[k] for k in train_keys}
    dev_dict = {k: data[k] for k in dev_keys}
    test_dict = {k: data[k] for k in test_keys}

    data_paths = [os.path.join(data_dir, f'nikl_{d}.pkl') for d in ['train', 'dev', 'test']]
    data_dicts = [train_dict, dev_dict, test_dict]

    for p, d in zip(*(data_paths, data_dicts)):
        with open(p, 'wb') as f:
            pickle.dump(d, f)
            print(f'File saved as {p}')


def load_json(data_dir):
    import json
    with open(data_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
    src = [ex['content'] for ex in data]
    tgt = [ex['bot_summary'] for ex in data]
    return src, tgt


def load_csv(data_dir):
    sep = '\t' if data_dir.endswith('.tsv') else ','
    import pandas as pd
    try:
        df = pd.read_csv(data_dir, sep=sep,
                         header=0, encoding='utf-8')
    except:
        try:
            sep = '\t'
            df = pd.read_csv(data_dir, sep=sep,
                             header=0, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(data_dir, sep=',',
                             header=0, encoding='ISO-8859-1')
    print(df.head())
    if 'abstractive' in df.columns:
        src = list(df['contents'].values)
        tgt = list(df['abstractive'].values)

    elif 'bot_summary' in df.columns:
        df['content'] = df['content'].astype(str)
        df['bot_summary'] = df['bot_summary'].astype(str)

        src = list(df['content'].values)
        tgt = list(df['bot_summary'].values)

    else:
        raise IndexError
    return src, tgt


def collate_tokens(values, pad_idx, left_pad=False,
                   pad_to_length=None):
    # Simplified version of `collate_tokens` from fairseq.data.data_utils
    """Convert a list of 1d tensors into a padded 2d tensor."""
    values = list(map(torch.LongTensor, values))
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res