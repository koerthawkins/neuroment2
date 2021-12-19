import glob
import hydra
import logging as log
from omegaconf import DictConfig
import os
import torch

from neuroment2 import *


@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    # we want CWD to be the root directory
    os.chdir(hydra.utils.to_absolute_path("."))

    # use GPU if abailable
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(cfg.gpu_index))
    else:
        device = "cpu"

    log.info("Selected device: %s" % device)

    # create model
    # TODO read me dynamically
    model = NeuromentModel(
        num_instruments=8,
        num_input_features=128,
        num_input_frames=14,
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


def validation():
    pass


if __name__ == "__main__":
    train()
