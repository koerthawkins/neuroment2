# neuroment2

A complete rewrite of "Neuroment - Instrument Detection using Neural Networks".

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

- generate data: `python generate_data.py -m Mixer.raw_data_path=clean_dataset Mixer.num_epochs=10 Mixer.pickle_path=data/pickle_10epochs Mixer.data_type=validation,test,training`
  - generates data for training+test+validation all at once
- training: `python train.py train.training_epochs=25 train.batch_size=32 train.gpu_index=0 train.continue_training=False train.dataset_dir=data/pickle_10epochs train.use_batch_norm=True`
- inference: `python inference.py inference.model_path=outputs/2022-01-12/23-18-58/models/neuroment2_00000149.model inference.audio_dir=data/test/ inference.predictions_dir="predictions"`
