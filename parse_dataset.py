import glob
import hydra
import logging as log
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, read_write
import os
from shutil import copyfile
from tqdm import tqdm


# number of instruments in the dataset
N_INSTRUMENTS = 8


@hydra.main(config_path="conf", config_name="config")
def parse(cfg: DictConfig) -> None:
    # switch to root directory to make path handling easier
    os.chdir(hydra.utils.to_absolute_path("."))

    # append max_oversampling_factor to output_dataset_dir
    with read_write(cfg):
        cfg.parsing.output_dataset_dir += "-%.1f" % cfg.parsing.max_oversampling_factor

    # create output directories
    os.makedirs(cfg.parsing.output_dataset_dir, exist_ok=True)
    os.makedirs(cfg.parsing.output_plot_dir, exist_ok=True)

    # run parsing for all dataset types at once
    for dataset_type in cfg.parsing.dataset_types:
        # example file path:
        # 'data/Medley-solos-DB/Medley-solos-DB_training-7_0244a60b-91bc-5202-fef2-548395a4ce86.wav'

        # initialize the array with the number of instruments per class
        num_samples_per_class = np.zeros(N_INSTRUMENTS,)

        # get number of instruments per class
        for instrument in range(N_INSTRUMENTS):
            # get the list of files with current dataset type and instrument
            file_list = glob.glob(
                os.path.join(cfg.parsing.input_dataset_dir, "M*_%s-%d_*.wav" % (dataset_type, instrument)),
            )
            num_samples_per_class[instrument] = len(file_list)

        if np.sum(num_samples_per_class) < 0.1:
            log.warning(f"No samples detected in {cfg.parsing.input_dataset_dir} for dataset type '{dataset_type}'.")
            continue

        # define the maximum number of samples per instrument that will be taken
        # don't take 0 into account
        n_samples_per_instrument_to_take = int(
            np.min(num_samples_per_class[num_samples_per_class > 0]) * cfg.parsing.max_oversampling_factor,
        )

        # validity check
        if n_samples_per_instrument_to_take == 0:
            raise ValueError("n_samples_per_instrument should not become 0!")

        # copy old files to new direction
        for instrument in range(N_INSTRUMENTS):
            # check if there are any files in current class
            if num_samples_per_class[instrument] == 0:
                log.warning("No samples in class '%s' of '%s' train set. Won't copy any samples!"
                            % (cfg.class_labels[instrument], dataset_type))
                continue

            # init progress bar
            progbar = tqdm(total=n_samples_per_instrument_to_take)
            progbar.set_description("%s | %s" % (dataset_type, cfg.class_labels[instrument]))

            # get the list of files with current dataset type and instrument
            file_list = glob.glob(
                os.path.join(cfg.parsing.input_dataset_dir, "M*_%s-%d_*.wav" % (dataset_type, instrument)),
            )

            # randomize the list s.t. we get files from different recordings
            np.random.shuffle(file_list)

            if len(file_list) > n_samples_per_instrument_to_take:
                # undersample the file list
                file_list = file_list[:n_samples_per_instrument_to_take]
            elif len(file_list) < n_samples_per_instrument_to_take:
                # oversample the file list
                while len(file_list) < n_samples_per_instrument_to_take:
                    file_list += file_list

                # remove the number of files that are too much
                file_list = file_list[:n_samples_per_instrument_to_take]

            # now go through file list of proper length and copy files from old to new dir
            for file in file_list:
                file_name_without_dir = os.path.basename(file)
                copyfile(file, os.path.join(cfg.parsing.output_dataset_dir, file_name_without_dir))
                progbar.update(1)

            # close the progress bar
            progbar.close()

        # now generate a bar plot of the dataset distribution
        fig = plt.figure(figsize=cfg.figsize)
        plt.bar(cfg.class_labels, num_samples_per_class)
        plt.title("Number of instruments in %s set" % dataset_type)
        plt.grid()
        plt.xticks(rotation="45")
        plt.tight_layout()

        plot_file_path = os.path.join(cfg.parsing.output_plot_dir, "class_distribution_%s.png" % dataset_type)
        fig.savefig(plot_file_path)
        log.info("Saved plot '%s'." % plot_file_path)


if __name__ == "__main__":
    parse()
