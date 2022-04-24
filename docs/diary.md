## Models

### 1 dataset epoch, 150 files per instrument, mel-features, rms-envelope

- `neuroment2_00002000_20211227.model`
  - 50 epochs
  - **predicted only zeros**
  - I think it's because it was trained with mel-features, where also the **envelope** was generated via mel spectrum
  - I looked into the data and the mel envelope rarely goes higher than 1e-4 (way too small)
- `neuroment2_00004997_202112292100.model`
  - 50 epochs
  - here I used RMS envelope with mel features (s.t. the envelope isn't too low)
  - still the network only predicts 0 values ...
  - **the problem seems to be MSELoss**
    - it gravitates towards 0 if a lot of data points are near 0
- `neuroment2_00002547_202112292237.model`
  - 25 epochs
  - BCELoss
  - works! predictions are reasonable (their range and their values)
  - however, they are not pretty accurate
  - batch norm after conv layers activated
- `neuroment2_00002547_202112292320.model`
  - 25 epochs
  - here I used **no** batch norm
  - the silence is predicted less accurate
  - however, the amplitude changes between parts of the songs are way more accurate

### 10 dataset epochs, 150 files per instrument, mel-features, rms-envelope

- `neuroment2_00025557_202112292325.model`
  - 25 epochs
  - no batch norm
  - network recognizes silence way better (non-appearant instruments are <= -100dB)
  - however, it also often not recognizes appearant instruments
- `neuroment2_00025557_202112292333.model`
  - 25 epochs
  - batch norm

## Losses

- MSELoss drew all predictions towards 0 after a few training epochs
  - I read that it can't cope with labels with many 0s...
  - same for L1Loss
- BCELoss worked then!
