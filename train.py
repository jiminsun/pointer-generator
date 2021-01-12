import argparse
import os
import sys

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data.data import Vocab
from data.data import build_dataloader
from data.data import build_dataset
from models.model import SummarizationModel
from utils import *


def main(config):
    # fix random seeds for reproducibility
    SEED = 123
    pl.seed_everything(SEED)

    # generate experiment name
    exp_name = generate_exp_name(config)

    # Train
    # 1. Load dataset, build or load vocab file
    # if use pre-built vocab
    if os.path.exists(config.path.vocab) and config.load_vocab:
        load_vocab = config.path.vocab
    else:
        load_vocab = None

    # 1-1. Training dataset
    # If validation set is not specified, take 10% of training data as validation set
    split_dev = 0.1 if config.path.val == '' else 0.0
    (train_data, val_data), vocab = build_dataset(
        data_path=config.path.train,
        load_vocab=load_vocab,
        config=config,
        is_train=True,
        split_dev=split_dev,
    )

    # save vocab file
    vocab.save(os.path.join(f'logs/{exp_name}', config.path.vocab))

    train_loader = build_dataloader(
        dataset=train_data,
        vocab=vocab,
        batch_size=config.data_loader.batch_size.train,
        max_decode=config.data.tgt_max_train,
        is_train=True,
        num_workers=config.data_loader.num_workers,
    )

    # 1-2. Load validation dataset
    # For validation, we use NIKL
    if val_data is None:
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

    # 2. Build model instance

    model = SummarizationModel(
        config=config,
        vocab=vocab,
    )

    # 3. Set logger, trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir='logs/',
        name=exp_name,
    )
    tb_logger.log_hyperparams(config)

    if config.stop_with == 'loss':
        monitor = 'val_loss_avg'
        mode = 'min'
    else:
        which_rouge = config.stop_with[-1]  # 1, 2 or l
        monitor = f'rouge_{which_rouge}_avg'
        mode = 'max'

    filepath = '{epoch}-{' + monitor + ':.2f}'
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(f'logs/{exp_name}', filepath),
        verbose=False,
        monitor=monitor,
        mode=mode,
        save_top_k=5,
    )

    # early stopping
    early_stop_callback = EarlyStopping(
        monitor=monitor,
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode=mode,
    )

    if config.device == -1:
        gpus = None
    else:
        gpus = [config.device]
        
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[early_stop_callback],
        gpus=gpus,
        resume_from_checkpoint=config.model_path,
        max_epochs=config.trainer.epochs,
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=config.trainer.max_grad_norm,
        log_every_n_steps=500,
    )

    # 4. Train!
    trainer.fit(model, train_loader, val_loader)

    # 5. Evaluation
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

    test_outputs = trainer.test(model, test_loader)
    output_name = generate_output_name(config)
    write_output(test_loader=test_loader,
                 test_outputs=test_outputs,
                 fname=os.path.join(f'logs/{exp_name}', output_name))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Pointer-generator network')
    args.add_argument(
        '-cp', '--config-path',
        default='config.json',
        type=str,
        help='path to config file'
    )

    args.add_argument(
        '--mds',
        default=None,
        type=str,
        help='multi-news labeling method to employ. if None, nikl dataset is used.'
    )

    args.add_argument(
        '-m', '--model-path',
        default=None,
        type=str,
        help='path to load model'
    )

    args.add_argument(
        '--load-vocab',
        action='store_true',
        default=False,
        help='whether to load pre-built vocab file'
    )

    args.add_argument(
        '--stop-with',
        default='rl',
        type=str,
        choices=['loss', 'r1', 'r2', 'rl'],
        help='validation evaluation metric to perform early stopping'
    )

    args.add_argument(
        '-e', '--exp-name',
        default='',
        type=str,
        help='suffix to specify experiment name'
    )

    args.add_argument(
        '-d', '--device',
        default=-1,
        type=int,
        help='gpu device number to use. if cpu, set this argument to -1'
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
