import argparse
import sys

import pytorch_lightning as pl

from data_loader.data import Vocab
from data_loader.data import build_dataloader
from data_loader.data import build_dataset
from eval_rouge import report_rouge
from models.model import SummarizationModel
from utils import *


def main(config):
    # load vocab
    log_path, _ = os.path.split(config.model_path)
    vocab = Vocab.from_json(os.path.join(log_path, config.path.vocab))

    print(f"Vocab size : {len(vocab)}")

    test_data = build_dataset(
        data_path=config.path.test,
        config=config,
        is_train=False,
        vocab=vocab
    )

    test_loader = build_dataloader(
        dataset=test_data,
        vocab=vocab,
        batch_size=1,
        max_decode=config.data.tgt_max_test,
        is_train=False,
        num_workers=config.data_loader.num_workers,
    )
    # config.model.use_pretrained = False
    model = SummarizationModel.load_from_checkpoint(
        checkpoint_path=config.model_path,
        config=config,
        vocab=vocab,
    )
    model.freeze()
    model.eval()
    
    if config.device == -1:
        gpus = None
    else:
        gpus = [config.device]

    trainer = pl.Trainer(
        gpus=gpus,
    )

    test_outputs = trainer.test(
        model=model,
        test_dataloaders=test_loader,
        ckpt_path=config.model_path,
        verbose=False
    )

    output_name = generate_output_name(config)
    output_fname = os.path.join(log_path, output_name)
    write_output(test_loader=test_loader,
                 test_outputs=test_outputs,
                 fname=output_fname)
    report_rouge(output_fname)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Pointer-generator network')
    args.add_argument(
        '-cp', '--config-path',
        default='config.json',
        type=str,
        help='path to config file'
    )

    args.add_argument(
        '-m', '--model-path',
        default=None,
        type=str,
        help='path to load model'
    )

    args.add_argument(
        '-d', '--device',
        default=0,
        type=int,
    )

    args.add_argument(
        '-n', '--note',
        default='',
        type=str,
        help='note to append to result output file name'
    )

    sys.path.append(
        os.path.dirname(os.path.abspath(os.path.dirname("__file__")))
    )

    config = config_parser(args.parse_args())
    main(config)