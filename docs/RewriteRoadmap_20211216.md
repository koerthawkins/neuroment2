# Rewrite roadmap

## General ideas

- split structure into 3 files
  - generate_dataset.py
  - train.py
  - inference.py
  - this way we won't have that many dependencies in one file, which is less error-prone
- first compute CQT, then split into frames
- we should implement the structure for CQT **and STFT**
  - we can use the STFT as a comparison method
  - even though it shouldn't work that well in theory, if it does not work that indicates an error in our code
  
## generate_dataset.py

- generate file lists
  - get list of base files
  - generate list with mix files
  - the following steps then are basically the same for base and mix files
- read audio
- generate stacked CQT **for whole file at once**
- generate CQT envelope **for whole file at once**
  - check if envelope makes sense for 1-beat sample!
- if audio is from a mix, mix multiple CQT frames (and envelopes)
- save training data to disk (each n frames)
  - also save start indices for new files
  
### Programming details

- save generator config in pickle directory
  - this way we can use the same CQT generation config for inference

## train.py

- get list of training files
- shuffle the training file list
- read a training file
- shuffle the frames in the training file!
  - we want to make sure that each consecutive training data frame is from another audio sample
  - otherwise our network gets biased a little bit for each audio file
- train using the frames in the training file
- save prediction results

## inference.py

- read audio file
- compute tatum for whole file
- compute CQT frames for whole file
- align CQT frames to *tatum bins/indices*
- predict envelopes using aligned CQT frames
- backward-transform predicted envelopes from tatum bins/indices to full time domain
- plot envelopes
  
## Things to prototype

- patching beat-aligned CQT frames back to original time domain
- CQT envelope vs. STFT envelope vs. RMS envelope
  - done already
  
## Data

- `frame_size = 2048` s.t. we generate reasonable spectral data down to 35Hz
- `hop_size = 512` s.t. the temporal resolution is OK
- don't search for onsets for now
  - we keep it simple and generate training data for fixed-length audio files
  - this means that as for now we aim to avoid using tatums (and beat-aligned prediction) in our inference!
- all files are mixes
  - 1-instrument mix is essentially a base file

### Feature transformations

- CQT
- STFT
- STFT with Mel filterbank (or other filterbank)
- **no** stacked CQT for now

### Mixes

- Levels drawn from normal distribution (mean = 0.5, minmax = [0.0, 1.0])
  - levels are then normalized to sum 1.0
- n_instruments drawn from uniform distribution (1 - 4)
- generate list with **all** base files
  - draw files from list until list is empty (when it is empty we ran through 1 dataset epoch)
  - if empty generate the list anew
  - generate until number of dataset epochs is through
  
### Labels

- instrument
- original file (name/path)
- offset (in s)
- length (in s) --> for librosa to load
- labels (i.e. training envelopes)
- features (i.e. spectral training data)
- frame index
- UID
  
  
