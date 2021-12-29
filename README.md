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

## Examples

- generate data: `python generate_data.py -m Mixer.raw_data_path=clean_dataset_small Mixer.pickle_path=data/pickle Mixer.data_type=validation,test,training`
  - generates data for training+test+validation all at once
- training: `python train.py train.dataset_dir=data/pickle/ train.training_epochs=50 train.batch_size=32 train.gpu_index=0 train.continue_training=False`
