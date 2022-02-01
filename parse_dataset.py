import glob
import numpy as np
import os
from shutil import copyfile


##### PARAMETERS ###################
INPUT_DATASET_DIR = "/media/WindowsPartition/Users/maier/Desktop/datasets/MedleyDB/full_dataset_v1p2"
OUTPUT_DATASET_DIR = "data/fixed"
DATASET_TYPES = ["training", "validation", "testing"]
N_INSTRUMENTS = 8

# we oversample the underrepresented classes by this factor
# overrepresented classes will be undersampled s.t. in the end each class has the same number of files
MAX_OVERSAMPLING_FACTOR = 2.0
####################################

# create output directory
os.makedirs(OUTPUT_DATASET_DIR, exist_ok=True)

# run parsing for all dataset types at once
for dataset_type in DATASET_TYPES:
    # example file path:
    # 'data/Medley-solos-DB/Medley-solos-DB_training-7_0244a60b-91bc-5202-fef2-548395a4ce86.wav'

    # initialize the array with the number of instruments per class
    num_instruments_per_class = np.zeros(N_INSTRUMENTS,)

    # get number of instruments per class
    for instrument in range(N_INSTRUMENTS):
        # get the list of files with current dataset type and instrument
        file_list = glob.glob(
            os.path.join(INPUT_DATASET_DIR, "M*_%s-%d_*.wav" % (dataset_type, instrument)),
        )
        num_instruments_per_class[instrument] = len(file_list)

    # define the maximum number of samples per instrument that will be taken
    n_samples_per_instrument = int(np.min(num_instruments_per_class) * MAX_OVERSAMPLING_FACTOR)

    # validity check
    if n_samples_per_instrument == 0:
        raise ValueError("n_samples_per_instrument should not become 0!")

    # copy old files to new direction
    for instrument in range(N_INSTRUMENTS):
        # get the list of files with current dataset type and instrument
        file_list = glob.glob(
            os.path.join(INPUT_DATASET_DIR, "M*_%s-%d_*.wav" % (dataset_type, instrument)),
        )

        # randomize the list s.t. we get files from different recordings
        np.random.shuffle(file_list)

        if len(file_list) > n_samples_per_instrument:
            # undersample the file list
            file_list = file_list[:n_samples_per_instrument]
        elif len(file_list) < n_samples_per_instrument:
            # oversample the file list
            while len(file_list) < n_samples_per_instrument:
                file_list += file_list

            # remove the number of files that are too much
            file_list = file_list[:n_samples_per_instrument]

        # now go through file list of proper length and copy files from old to new dir
        for file in file_list:
            file_name_without_dir = os.path.basename(file)
            copyfile(file, os.path.join(OUTPUT_DATASET_DIR, file_name_without_dir))
