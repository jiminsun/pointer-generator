import os
import json
from datetime import datetime

import easydict
import torch
from setproctitle import setproctitle


def config_parser(args=None, interpreter=None):
    if interpreter:
        with open(interpreter, 'rb') as f:
            config = dict(json.load(f))
        config = easydict.EasyDict({**config})
    else:
        with open(args.config_path, 'rb') as f:
            config = dict(json.load(f))
        config = easydict.EasyDict({**config, **vars(args)})

    for k, v in vars(args).items():
        config[k] = v

    return config


def write_output(test_loader, test_outputs, fname):
    save_dir, _ = os.path.split(fname)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    header = ['id', 'src', 'tgt', 'pred']
    with open(fname, 'w') as f:
        print('\t'.join(header), file=f)
        for batch_idx, batch in enumerate(test_loader):
            top_k_outputs = test_outputs[batch_idx]['generated_summary']
            for k, output in enumerate(top_k_outputs, 1):
                data = [str(batch_idx) + '_' + str(k), batch.src_text[0], batch.tgt_text[0], output]
                print('\t'.join(data), file=f)
    print(f"Predicted outputs saved as {fname}")


def generate_exp_name(config):
    exp_name = datetime.now().strftime("%m-%d-%H:%M:%S")
    if len(config.exp_name):
        exp_name = exp_name + f'-{config.exp_name}'
    # set process title to exp name
    setproctitle(exp_name)
    print(f'Experiment results saved in logs/{exp_name}')
    return exp_name


def generate_output_name(config):
    if config.model_path is not None:
        basename = os.path.basename(config.model_path)
        model_name, _ = os.path.splitext(basename)
    else:
        model_name = 'best'
    if len(config.note):
        model_name += f'-{config.note}'
    basename = os.path.basename(config.path.test)
    filename, _ = os.path.splitext(basename)
    output_name = f'pred-{filename}-{model_name}.tsv'
    return output_name