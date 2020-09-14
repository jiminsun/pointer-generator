import torch


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
    df = pd.read_csv(data_dir, sep=sep,
                     header=0, encoding='utf-8')
    src = list(df['content'].values)
    tgt = list(df['bot_summary'].values)
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