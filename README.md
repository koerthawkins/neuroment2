# neuroment2

Instrument detection using convolutional neural networks.

## Structure

- `cluster/`: Configuration files and scripts for slurm simulation clusters
- `conf/config.yaml`: Configuration file for the main application scripts
- `data/`: Directory where data may saved to and read from
- `docs/Neuroment2_Haeusler-Maier.pdf`: Official documentation
- `generate_data.py`: Main script to generate model training data
  - Configured via `conf/config.yaml`
- `inference.py`: Main script to generate predictions and output plots with a given model checkpoint for specified audio files
  - Configured via `conf/config.yaml`
- `models/`: Model checkpoints
- `neuroment2/`: Main python module for the neuroment2 project
- `parse_dataset.py`: Parses the original, unbalanced Medley-Solos-DB and generates a new, balanced version out of it
- `scripts/`: Some small scripts
- `train.py`: Main script for model training
  - Configured via `conf/config.yaml`

## Dataset

For our experiments we used the [Medley-solos-DB](https://zenodo.org/record/1344103#.YczLvNso9hE) dataset (v1.2). It contains 8 different instrument classes, sampled at 44.1kHz and with ~21k files in total. The instrument classes are (in this order):

0. clarinet
1. distorted electric guitar
2. female singer
3. flute
4. piano
5. tenor saxophone
6. trumpet
7. violin

## Recommended settings

- `use_batch_norm`: If we use BatchNorm after Conv2D layers the network predicts silence a bit better, but it also kind of "generalizes" the amplitude. This means, that most frames are the same amplitude.

## How to run (with examples)

### Environment setup

We use the [anaconda](https://docs.anaconda.com/anaconda-repository/user-guide/tasks/pkgs/use-pkg-managers/) package manager.

1. Install anaconda/[miniconda](https://docs.conda.io/en/latest/miniconda.html) package manager
2. Install the conda environment via `conda env create -f neuroment2-cpu.yaml` (or the GPU-ready environment `neuroment2-gpu.yaml`)
3. Activate the environment via `conda activate neuroment2-cpu` (or `neuroment2-gpu`)
4. Run a script (see below)

### Data generation

Before data generation you need to download raw audio data. We recommend to download v1.2 of the MedleySolos-DB dataset to `data/` and then specify the command line argument `Mixer.raw_data_path` with the extraction path. For testing purposes though we added a small, balanced version of the dataset in `data/medley-solos-db-small/`.


- generate data: `python generate_data.py -m Mixer.raw_data_path=clean_dataset Mixer.num_epochs=10 Mixer.pickle_path=data/pickle_10epochs Mixer.data_type=validation,test,training`
  - generates data for training+test+validation all at once

### Training

### Inference

- Using the provided checkpoint: `python inference.py inference.audio_dir=data/test/ inference.model_path=models/neuroment2_cqt_00125782.model inference.predictions_dir=predictions/`
  - All predictions will be written to `predictions/`

## Examples

### Complete run

- parse the original raw audio file dataset s.t. it is balanced

```bash
python parse_dataset.py parsing.input_dataset_dir=input/ parsing.output_dataset_dir=output/ parsing.max_oversampling_factor=2.0
```

- generate the training dataset

```bash
python generate_data.py -m Mixer.raw_data_path=output/ Mixer.num_epochs=1 Mixer.pickle_path=output_dataset/ Mixer.data_type=validation,test,training
```

- train with the training dataset:

```bash
python train.py train.training_epochs=25 train.batch_size=32 train.gpu_index=0 train.continue_training=False train.dataset_dir=output_dataset/ train.use_batch_norm=True
```

- predict envelopes of test audio files in `data/test`

```bash
python inference.py inference.model_path=neuroment2_00010000.model inference.audio_dir=data/test/ inference.predictions_dir="predictions"
```

### Single

- generate data: `python generate_data.py -m Mixer.raw_data_path=clean_dataset Mixer.num_epochs=10 Mixer.pickle_path=data/pickle_10epochs Mixer.data_type=validation,test,training`
  - generates data for training+test+validation all at once
- training: `python train.py train.training_epochs=25 train.batch_size=32 train.gpu_index=0 train.continue_training=False train.dataset_dir=data/pickle_10epochs train.use_batch_norm=True`
- inference: `python inference.py inference.model_path=outputs/2022-01-12/23-18-58/models/neuroment2_00000149.model inference.audio_dir=data/test/ inference.predictions_dir="predictions"`
