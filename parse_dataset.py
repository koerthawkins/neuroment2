import glob
import os
import pickle as pk
import yaml

import numpy as np
from librosa.core import audio, load
import librosa as lb

from neuroment2.utils import delete_by_indices
import matplotlib.pyplot as plt
from shutil import copyfile

os.chdir("data/Medley-solos-DB/")
type = "validation"
num_instruments = np.zeros(
    8,
)
for file in glob.glob("*.wav"):
    if type in file:
        file_cur = file[16::]
        instr = int(file_cur.split("-")[1][0])
        num_instruments[instr] += 1

n_samples_per_instrument = int(np.min(num_instruments))

files_instruments = [[] for x in range(8)]
files = []
files_new = []
for file in glob.glob("*.wav"):
    if type in file:
        file_cur = file[16::]
        instr = int(file_cur.split("-")[1][0])
        num_instruments[instr] += 1
        if len(files_instruments[instr]) < n_samples_per_instrument:
            files_instruments[instr].append(file)
            copyfile(file, "../dataset_new/" + file[16::])


print(num_instruments)
