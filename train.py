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
    # we want CWD to be the root directory
    os.chdir(hydra.utils.to_absolute_path("."))

    # use GPU if abailable
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(cfg.train.gpu_index))
    else:
        device = "cpu"

    log.info("Selected device: %s" % device)

    # read dataset statistics to get model size
    # we always derive statistics from training set
    # TODO check against validation and testing
    with open(os.path.join(cfg.train.dataset_dir, "training_statistics.yml"), "r") as f:
        dataset_stats = yaml.load(f, Loader=yaml.SafeLoader)

    # create model
    model = NeuromentModel(
        num_instruments=dataset_stats["num_instruments"],
        num_input_features=dataset_stats["num_features_per_observation"],
        num_input_frames=dataset_stats["num_frames_per_observation"],
    )
    model.to(device)

    # create model directory
    os.makedirs(cfg.train.model_dir, exist_ok=True)
    log.info("Model directory: %s" % cfg.train.model_dir)

    # if we want to continue training and there are saved models already get the one with the most steps
    if cfg.train.continue_training and os.path.isdir(cfg.train.model_dir):
        pattern = os.path.join(cfg.train.model_dir, "neuroment2_????????.model")
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

        log.info("Loaded model checkpoint '%s'." % model_checkpoint)
    else:
        state_dict_model = None

        # these vars will be incremented at beginning of training loop,
        # so we set them to -1 here
        step = -1
        epoch = -1

        log.info("No model checkpoint found in '%s', starting training from scratch."
                 % cfg.train.model_dir)

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

    # load optimizer if we found a checkpoint
    if state_dict_model:
        optimizer.load_state_dict(state_dict_model["optimizer"])

    # create training dataset and loader
    train_dataset = Neuroment2Dataset(
        cfg.train.dataset_dir,
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
        cfg.train.dataset_dir,
        "validation",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    # init SummaryWriter to log TensorBoard events
    train_writer = SummaryWriter(os.path.join(cfg.train.tensorboard_dir, "train"))
    val_writer = SummaryWriter(os.path.join(cfg.train.tensorboard_dir, "val"))

    # create loss function
    loss_fn = torch.nn.MSELoss()

    # set model to training mode
    model.train()

    # we want to collect the latest 100 losses for averaging
    loss_list = []

    while step < cfg.train.training_steps:
        # we hit a new epoch if we land here
        epoch += 1

        # create new progress bar
        prog_bar = tqdm(total=len(train_loader), desc="Training epoch: %d, step: %d" % (epoch, step))

        for i_batch, (x, y) in enumerate(train_loader):
            # increase step counter
            step += 1

            # tell pytorch to attach gradients to our features and labels
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))

            # add channel dimension
            x = torch.unsqueeze(x, 1)

            # reset gradient
            optimizer.zero_grad()

            # predict
            y_pred = model(x)

            # compute losses
            loss = loss_fn(y_pred, y)

            # backwards-propagate loss and update weights in model
            loss.backward()
            optimizer.step()

            # compute average loss over the last n_batches_per_average batches
            loss_list.append(float(loss))
            if len(loss_list) > cfg.train.n_batches_per_average:
                loss_list = loss_list[-cfg.train.n_batches_per_average:]
            avg_loss = np.mean(loss_list)

            # update progress bar. don't force-refresh because that will create a new line
            prog_bar.set_postfix(
                {
                    "Batch": i_batch + 1,
                    'Loss (avg)': avg_loss,
                },
                refresh=False,
            )
            prog_bar.update(1)

            # log losses to TensorBoard SummaryWriter
            train_writer.add_scalar("loss", loss, global_step=step)
            train_writer.add_scalar("avg_loss", avg_loss, global_step=step)

            # save checkpoint
            if step % cfg.train.model_checkpoint_interval == 0:
                checkpoint_path = "%s/neuroment2_%.8d.model" % (cfg.train.model_dir, step)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
                log.info("Saved model checkpoint '%s'." % checkpoint_path)

            # run validation loop
            if step % cfg.train.validation_interval == 0:
                validation(
                    cfg,
                    model,
                    val_loader,
                    val_writer,
                    step,
                    epoch,
                )


def validation(
    cfg: DictConfig,
    model: NeuromentModel,
    val_loader: DataLoader,
    val_writer: SummaryWriter,
    step: int,
    epoch: int,
):
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        # create new progress bar
        prog_bar = tqdm(total=len(val_loader), desc="Validation epoch: %d" % epoch)

        # init loss_list
        loss_list = []

        for i_batch, (x, y) in enumerate(val_loader):
            # add channel dimension
            x = torch.unsqueeze(x, 1)
            
            # predict
            y_pred = model(x)

            # compute loss
            loss = loss_fn(y_pred, y)

            # compute average loss over the last n_batches_per_average batches
            loss_list.append(float(loss))
            if len(loss_list) > cfg.train.n_batches_per_average:
                loss_list = loss_list[-cfg.train.n_batches_per_average:]
            avg_loss = np.mean(loss_list)

            # update progress bar. don't force-refresh because that will create a new line
            prog_bar.set_postfix(
                {
                    "Batch": i_batch + 1,
                    'Loss (avg)': avg_loss,
                },
                refresh=False,
            )
            prog_bar.update(1)

            # log losses to TensorBoard SummaryWriter
            val_writer.add_scalar("loss", loss, global_step=step)
            val_writer.add_scalar("avg_loss", avg_loss, global_step=step)


if __name__ == "__main__":
    train()
