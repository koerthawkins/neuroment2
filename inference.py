import glob
import hydra
import logging as log
import librosa as lb
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
def inference(cfg: DictConfig) -> None:
    # we switch to repository root dir to simplify path handling
    # (inference doesn't have intermediate results, so no need to introduce an extra hydra dir)
    os.chdir(hydra.utils.to_absolute_path("."))

    # use GPU if available
    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(cfg.train.gpu_index))
    else:
        device = "cpu"

    log.info("Selected device: %s" % device)

    # load state dict s.t. we get parameters to create model
    state_dict_model = torch.load(cfg.inference.model_path, map_location=device)
    dataset_stats = state_dict_model["dataset_stats"]
    log.info("Loaded model: '%s'." % cfg.inference.model_path)

    # create model
    model = NeuromentModel(
        num_instruments=dataset_stats["num_instruments"],
        num_input_features=dataset_stats["num_input_features"],
        num_input_frames=dataset_stats["num_input_frames"],
    )
    model.to(device)

    # create feature generator
    feature_gen = FeatureGenerator(dataset_stats["feature_generator_cfg"])

    # create predictions directory
    os.makedirs(cfg.inference.predictions_dir, exist_ok=True)
    log.info("Saving predictions to: %s" % os.path.abspath(cfg.inference.predictions_dir))

    # set model to evaluation mode
    model.eval()

    # get list of audio files
    inference_file_list = glob.glob(os.path.join(cfg.inference.audio_dir, ".{wav,flac,mp3,WAV,FLAC,MP3}"))

    # make progbar
    prog_bar = tqdm(total=len(inference_file_list), desc="Predicting...")

    for input_file in inference_file_list:
        # load raw audio
        audio, _ = lb.load(input_file, sr=dataset_stats.sr, mono=True,)

        # generate features
        features, envelope_ref = feature_gen.generate(audio)

        # predict
        envelope_pred = model(features)

        # create new progress bar
        prog_bar.update(1)



def validation(
    cfg: DictConfig,
    model: NeuromentModel,
    val_loader: DataLoader,
    val_writer: SummaryWriter,
    step: int,
    epoch: int,
    device: torch.device,
):
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        # create new progress bar
        prog_bar = tqdm(total=len(val_loader), desc="Validation epoch: %d, step: %d" % (epoch, step))

        # init loss_list
        loss_list = []

        for i_batch, (x, y) in enumerate(val_loader):
            # move tensors to right device
            x = x.to(device)
            y = y.to(device)

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
    inference()
