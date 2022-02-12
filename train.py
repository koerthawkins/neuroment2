import glob
import hydra
import logging as log
from omegaconf import DictConfig
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from neuroment2 import *


@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    # use GPU if abailable
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(cfg.train.gpu_index))
    else:
        device = "cpu"

    log.info("Selected device: %s" % device)

    # read dataset statistics to get model size
    # we always derive statistics from training set
    # TODO check against validation and testing
    resolved_dataset_dir = os.path.join(
        hydra.utils.to_absolute_path("."), cfg.train.dataset_dir
    )
    with open(os.path.join(resolved_dataset_dir, "training_statistics.yml"), "r") as f:
        dataset_stats = yaml.load(f, Loader=yaml.SafeLoader)

    # create model
    model = NeuromentModel(
        num_instruments=dataset_stats["num_instruments"],
        num_input_features=dataset_stats["num_features_per_observation"],
        num_input_frames=dataset_stats["num_frames_per_observation"],
        use_batch_norm=cfg.train.use_batch_norm,
        dropout_rate=cfg.train.dropout_rate,
    )
    model.to(device)

    # create model directory
    os.makedirs(cfg.train.model_save_dir, exist_ok=True)
    log.info("Model saving directory: %s" % os.path.abspath(cfg.train.model_save_dir))

    # if we want to continue training and there are saved models already get the one with the most steps
    # we also need to resolve the model_dir because it is not in hydra CWD
    if cfg.train.continue_training and cfg.train.model_load_dir:
        resolved_model_load_dir = os.path.join(
            hydra.utils.to_absolute_path("."), cfg.train.model_load_dir
        )
        pattern = os.path.join(resolved_model_load_dir, "neuroment2_????????.model")

        # get all checkpoints and sort out the one with the most steps
        checkpoint_list = glob.glob(pattern)
        if len(checkpoint_list) > 0:
            model_checkpoint = sorted(checkpoint_list)[-1]
        else:
            model_checkpoint = None
    else:
        model_checkpoint = None

    # load checkpoint if we found one
    if model_checkpoint:
        state_dict_model = torch.load(model_checkpoint, map_location=device)
        step = state_dict_model["step"]
        epoch = state_dict_model["epoch"]

        log.info("Loaded model checkpoint '%s'." % os.path.abspath(model_checkpoint))
    else:
        state_dict_model = None

        # these vars will be incremented at beginning of training loop,
        # so we set them to -1 here
        step = -1
        epoch = -1

        log.info(
            "No model checkpoint found in '%s', starting training from scratch."
            % os.path.abspath(cfg.train.model_save_dir)
        )

    # init optimizer. we use Adam with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        cfg.optimizer.learning_rate,
        betas=(
            cfg.optimizer.beta[0],
            cfg.optimizer.beta[1],
        ),
        weight_decay=cfg.optimizer.weight_decay,
    )

    # init scheduler
    # it reduces the learning rate if the training reaches a plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.scheduler.mode,
        factor=cfg.scheduler.factor,
        patience=cfg.scheduler.patience,
        threshold=cfg.scheduler.threshold,
        min_lr=cfg.scheduler.min_lr,
        verbose=False,  # we log learning rate to TensorBoard
    )

    # load optimizer and scheduler state if a model checkpoint was loaded
    if state_dict_model:
        optimizer.load_state_dict(state_dict_model["optimizer"])
        scheduler.load_state_dict(state_dict_model["scheduler"])

    # create training dataset and loader
    train_dataset = Neuroment2Dataset(
        resolved_dataset_dir,
        "training",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )

    # create validation dataset and loader
    val_dataset = Neuroment2Dataset(
        resolved_dataset_dir,
        "validation",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    # create test dataset and loader
    test_dataset = Neuroment2Dataset(
        resolved_dataset_dir,
        "test",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    # init SummaryWriter to log TensorBoard events
    train_writer = SummaryWriter(os.path.join(cfg.train.tensorboard_dir, "train"))
    val_writer = SummaryWriter(os.path.join(cfg.train.tensorboard_dir, "val"))
    test_writer = SummaryWriter(os.path.join(cfg.train.tensorboard_dir, "test"))

    # create loss function
    loss_fn_1, loss_fn_2, loss_fn_3, loss_fn_4 = _get_loss_functions(cfg)

    # set model to training mode
    model.train()

    # we want to collect the latest 100 losses for averaging
    loss_list = []

    # check that stopping criteria are valid
    assert cfg.train.training_steps >= 1 or cfg.train.training_epochs >= 1

    while step < cfg.train.training_steps or epoch < cfg.train.training_epochs:
        # we hit a new epoch if we land here
        epoch += 1

        # create new progress bar
        prog_bar = tqdm(
            total=len(train_loader), desc="Training epoch: %d, step: %d" % (epoch, step)
        )

        for i_batch, (x, y_ref) in enumerate(train_loader):
            # increase step counter
            step += 1

            # tell pytorch to attach gradients to our features and labels
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y_ref = torch.autograd.Variable(y_ref.to(device, non_blocking=True))

            # reset gradient
            optimizer.zero_grad()

            # predict
            y_pred = model(x)

            # compute losses
            loss_1 = loss_fn_1(y_pred, y_ref)
            loss_2 = loss_fn_2(y_pred, y_ref)
            loss_3 = loss_fn_3(y_pred, y_ref)
            loss_4 = loss_fn_4(y_pred, y_ref)
            total_loss = (
                loss_1 * cfg.train.loss_weights["bce"]
                + loss_2 * cfg.train.loss_weights["mse"]
                + loss_3 * cfg.train.loss_weights["bce_per_instrument"]
                + loss_4 * cfg.train.loss_weights["frobenius"]
            )

            # backwards-propagate loss and update weights in model
            total_loss.backward()
            optimizer.step()

            # compute average loss over the last n_batches_per_average batches
            loss_list.append(float(total_loss))
            if len(loss_list) > cfg.train.n_batches_per_average:
                loss_list = loss_list[-cfg.train.n_batches_per_average :]
            avg_total_loss = np.mean(loss_list)

            # update progress bar. don't force-refresh because that will create a new line
            prog_bar.set_postfix(
                {
                    "Batch": i_batch + 1,
                    "Loss (avg)": avg_total_loss,
                },
                refresh=False,
            )
            prog_bar.update(1)

            # log scalar values to TensorBoard SummaryWriter
            train_writer.add_scalar("losses/loss_1", loss_1, global_step=step)
            train_writer.add_scalar("losses/loss_2", loss_2, global_step=step)
            train_writer.add_scalar("losses/loss_3", loss_3, global_step=step)
            train_writer.add_scalar("losses/loss_4", loss_4, global_step=step)
            train_writer.add_scalar("losses/total_loss", total_loss, global_step=step)
            train_writer.add_scalar(
                "losses/avg_total_loss", avg_total_loss, global_step=step
            )
            train_writer.add_scalar(
                "misc/learning_rate", optimizer.param_groups[0]["lr"], global_step=step
            )
            train_writer.add_scalar("misc/epoch", epoch, global_step=step)

            # save checkpoint
            if (cfg.train.model_checkpoint_interval != -1) and (
                step % cfg.train.model_checkpoint_interval == 0
            ):
                checkpoint_path = "%s/neuroment2_%.8d.model" % (
                    cfg.train.model_save_dir,
                    step,
                )
                _save_model(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    step,
                    epoch,
                    dataset_stats,
                    cfg,
                )

        # run validation loop
        val_loss = validation(
            cfg,
            model,
            val_loader,
            val_writer,
            step,
            epoch,
            device,
        )

        # end of an epoch, call learning rate scheduler
        scheduler.step(val_loss)

        # close progress bar s.t. it is flushed
        prog_bar.close()

    # save final model state
    checkpoint_path = "%s/neuroment2_%.8d.model" % (cfg.train.model_save_dir, step)
    _save_model(
        checkpoint_path, model, optimizer, scheduler, step, epoch, dataset_stats, cfg
    )

    # benchmark the test set
    test_loss = validation(
        cfg,
        model,
        test_loader,
        test_writer,
        step,
        epoch,
        device,
    )
    log.info("Test loss: %.5f" % test_loss)

    # print
    log.info("Finished.")


def validation(
    cfg: DictConfig,
    model: NeuromentModel,
    val_loader: DataLoader,
    val_writer: SummaryWriter,
    step: int,
    epoch: int,
    device: torch.device,
):
    # get loss function objects
    loss_fn_1, loss_fn_2, loss_fn_3, loss_fn_4 = _get_loss_functions(cfg)

    with torch.no_grad():
        # create new progress bar
        prog_bar = tqdm(
            total=len(val_loader), desc="Validation epoch: %d, step: %d" % (epoch, step)
        )

        # init loss_list
        loss_list = []

        for i_batch, (x, y_ref) in enumerate(val_loader):
            # move tensors to right device
            x = x.to(device)
            y_ref = y_ref.to(device)

            # predict
            y_pred = model(x)

            # compute losses
            loss_1 = loss_fn_1(y_pred, y_ref)
            loss_2 = loss_fn_2(y_pred, y_ref)
            loss_3 = loss_fn_3(y_pred, y_ref)
            loss_4 = loss_fn_4(y_pred, y_ref)
            total_loss = (
                loss_1 * cfg.train.loss_weights["bce"]
                + loss_2 * cfg.train.loss_weights["mse"]
                + loss_3 * cfg.train.loss_weights["bce_per_instrument"]
                + loss_4 * cfg.train.loss_weights["frobenius"]
            )

            # compute average loss over the last n_batches_per_average batches
            loss_list.append(float(total_loss))
            if len(loss_list) > cfg.train.n_batches_per_average:
                loss_list = loss_list[-cfg.train.n_batches_per_average :]
            avg_total_loss = np.mean(loss_list)

            # update progress bar. don't force-refresh because that will create a new line
            prog_bar.set_postfix(
                {
                    "Batch": i_batch + 1,
                    "Loss (avg)": avg_total_loss,
                },
                refresh=False,
            )
            prog_bar.update(1)

            # log losses to TensorBoard SummaryWriter
            val_writer.add_scalar("losses/loss_1", loss_1, global_step=step)
            val_writer.add_scalar("losses/loss_2", loss_2, global_step=step)
            val_writer.add_scalar("losses/loss_3", loss_3, global_step=step)
            val_writer.add_scalar("losses/loss_4", loss_4, global_step=step)
            val_writer.add_scalar("losses/total_loss", total_loss, global_step=step)
            val_writer.add_scalar(
                "losses/avg_total_loss", avg_total_loss, global_step=step
            )

    # close progress bar s.t. it is flushed
    prog_bar.close()

    # return the total avg val_loss
    return np.mean(loss_list)


def _save_model(
    checkpoint_path, model, optimizer, scheduler, step, epoch, dataset_stats, cfg
):
    """A simple utility function to save a model checkpoint."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "dataset_stats": dataset_stats,  # samplerate, hopsize, feature_generator.cfg, etc.
            "use_batch_norm": cfg.train.use_batch_norm,
            "dropout_rate": cfg.train.dropout_rate,
        },
        checkpoint_path,
    )
    log.info("Saved model checkpoint '%s'." % checkpoint_path)


def _get_loss_functions(cfg: DictConfig):
    """Returns new loss function objects"""
    weight_per_instrument = torch.zeros(size=[8])

    # we assume that it's possible that the order of instruments in cfg["class_labels"] is different
    # then the order in cfg["train"]["bce_weight_per_instrument"]
    # due to that we dynamically search for the corresponding bce weight
    for i_class, class_label in enumerate(cfg["class_labels"]):
        for instrument in cfg["train"]["bce_weight_per_instrument"].keys():
            # the list of instruments must always be the same
            assert instrument in cfg["class_labels"]

            if instrument == class_label:
                weight_per_instrument[i_class] = cfg["train"][
                    "bce_weight_per_instrument"
                ][class_label]

    return (
        torch.nn.BCELoss(),
        torch.nn.MSELoss(),
        BinaryCrossentropyPerInstrument(weight_per_instrument=weight_per_instrument),
        FrobeniusLoss(),
        # torch.nn.KLDivLoss(),
    )


if __name__ == "__main__":
    train()
