import os
import torch
import logging
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import switch_functions
from utils.argparse_init import default_arg_parser, init_parse_argparse_default_params
from utils.pytorch_lightning_utils import load_params_from_checkpoint
from utils.model_checkpoint_utils import extract_model_path_for_hyperparams


def main_train(model_class_pointer, hparams, parser):
    """Initialize the model, call training loop."""

    torch.backends.cudnn.benchmark = False
    logging.basicConfig(level=logging.INFO)
    pl.seed_everything(seed=42)

    if hparams.resume_from_checkpoint is not None:
        hparams = load_params_from_checkpoint(hparams, parser)

    model = model_class_pointer(hparams)
    model.hparams.display_id = 0
    model.hparams.log_to_dir = extract_model_path_for_hyperparams(
        model.hparams.default_root_dir,
        model,
        model.hparams,
    )
    logger = TensorBoardLogger(
        save_dir=model.hparams.log_to_dir, name="", default_hp_metric=False
    )
    logger.log_hyperparams(model.hparams)

    print(f"\nLog directory:\n{model.hparams.log_to_dir}\n")
    print(f"\n\n\n{model.hparams.log_to_dir}\n\n\n")

    checkpoint_callback = ModelCheckpoint(
        dirpath=model.hparams.log_to_dir,
        filename="{epoch:02d}",
        verbose=True,
        save_top_k=-1,  # saves all checkpoints
        every_n_epochs=model.hparams.ckpt_period,
    )
    hparams.max_epochs = 600
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_gpu_memory="all",
        weights_summary="top",
        logger=logger,
        max_epochs=hparams.max_epochs,
        precision=hparams.precision,
        auto_lr_find=False,
        gradient_clip_val=hparams.gradient_clip_val,
        benchmark=True,
        gpus=str(hparams.gpus)
        if str(hparams.gpus) != "-1"
        else None,  # if not hparams.DEBUG_MODE else 1,
        strategy="dp"
        if hparams.gpus != "-1"
        else None,  # if not hparams.DEBUG_MODE else 'sp',
        num_sanity_val_steps=hparams.num_sanity_val_steps,
        val_check_interval=hparams.val_check_interval,  # how many times(0.25=4) to run validation each training loop
        limit_train_batches=hparams.limit_train_batches,  # how much of the training data to train on
        limit_val_batches=hparams.limit_val_batches,  # how much of the validation data to train on
        limit_test_batches=hparams.limit_test_batches,  # how much of the validation data to train on
        detect_anomaly=True,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        # load
        resume_from_checkpoint=hparams.resume_from_checkpoint,
        replace_sampler_ddp=False,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        flush_logs_every_n_steps=hparams.flush_logs_every_n_steps,
        log_every_n_steps=hparams.log_every_n_steps,
        reload_dataloaders_every_epoch=False,
    )

    if hparams.do_train:
        trainer.fit(model)
    else:
        if hparams.resume_from_checkpoint is not None:
            model = model.load_from_checkpoint(
                hparams.resume_from_checkpoint,
                hparams=model.hparams,
                map_location=torch.device(f"cpu"),
            )
    test_out = trainer.test(model)
    return test_out, model


if __name__ == "__main__":
    """Initialize all the parsers, before training init."""
    parser = default_arg_parser(description="Point correspondence")
    parser = pl.Trainer.add_argparse_args(parser)
    eager_flags = init_parse_argparse_default_params(parser)

    """Initialize model parsers, before training init."""
    model_class_pointer = switch_functions.model_class_pointer(
        eager_flags["task_name"], eager_flags["arch"]
    )
    parser = model_class_pointer.add_model_specific_args(
        parser, eager_flags["task_name"], eager_flags["dataset_name"]
    )
    hparams = parser.parse_args()

    print("dataset: ", eager_flags["dataset_name"])
    main_train(model_class_pointer, hparams, parser)
