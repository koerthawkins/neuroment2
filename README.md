# neuroment2

A complete rewrite of "Neuroment - Instrument Detection using Neural Networks".

## Structure

- `parse_dataset.py`: Parses the original, unbalanced Medley-Solos-DB and generates a new, balanced version out of it

## Dataset

We use the [Medley-solos-DB](https://zenodo.org/record/1344103#.YczLvNso9hE). It contains 8 different instrument classes, sampled at 44.1kHz and with ~21k files in total. The instrument classes are (in this order):

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
