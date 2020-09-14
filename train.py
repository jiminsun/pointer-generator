import argparse
import os

import pytorch_lightning as pl
import sys
from pytorch_lightning import loggers as pl_loggers

from data_loader.data import Vocab
from data_loader.data import build_dataloader
from data_loader.data import build_dataset
from models.model import SummarizationModel
from utils import *


def main(config):
    # setup GPU device if available
    device, device_ids = prepare_device(config['n_gpu'])

    # fix random seeds for reproducibility
    SEED = 123
    pl.seed_everything(SEED)

    # Train
    # 1. Load training dataset and build / load vocab
    if os.path.exists(config.path.vocab):
        train_data, vocab = build_dataset(
            data_path=config.path.train,
            load_vocab=config.path.vocab,
            config=config,
            is_train=True
        )
    else:
        train_data, vocab = build_dataset(
            data_path=config.path.train,
            load_vocab=None,
            config=config,
            is_train=True
        )
        vocab.save(config.path.vocab)

    train_loader = build_dataloader(
        dataset=train_data,
        vocab=vocab,
        batch_size=config.data_loader.batch_size.train,
        max_decode=config.data.tgt_max_train,
        is_train=True,
        num_workers=config.data_loader.num_workers,
    )

    # 2. Load validation dataset
    val_data = build_dataset(
        data_path=config.path.val,
        config=config,
        is_train=False,
        vocab=vocab
    )

    val_loader = build_dataloader(
        dataset=val_data,
        vocab=vocab,
        batch_size=config.data_loader.batch_size.val,
        max_decode=config.data.tgt_max_test,
        is_train=False,
        num_workers=config.data_loader.num_workers,
    )

    # 3. Build model instance
    model = SummarizationModel(
        config=config,
        vocab=vocab,
    )

    # 4. Set logger, trainer
    tb_logger = pl_loggers.TensorBoardLogger('logs/', version=1)

    trainer = pl.Trainer(
        logger=tb_logger,
        default_root_dir='./checkpoints/',
        gpus=1,
        max_epochs=config.trainer.epochs,
    )

    # 5. Train!
    trainer.fit(model, train_loader, val_loader)

    # 6. Evaluation
    vocab = Vocab.from_json(config.path.vocab)
    test_data = build_dataset(
        data_path=config.path.test,
        config=config,
        is_train=False,
        vocab=vocab
    )

    test_loader = build_dataloader(
        dataset=test_data,
        vocab=vocab,
        batch_size=config.data_loader.batch_size.test,
        max_decode=config.data.tgt_max_test,
        is_train=False,
        num_workers=config.data_loader.num_workers,
    )

    test_outputs = trainer.test(model, test_loader)
    print(test_outputs[0:3])


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Pointer-generator network')
    args.add_argument(
        '-cp', '--config-path',
        default='config.json',
        type=str,
        help='path to config file'
    )

    sys.path.append(
        os.path.dirname(os.path.abspath(os.path.dirname("__file__")))
    )

    config = config_parser(args.parse_args())
    main(config)