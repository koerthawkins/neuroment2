## 2022-02-13
## train with CQT+MEL dataset
# 1. oversampling factor 0.2, CQT+MEL, 1 epoch data generation, 150 epochs training, only BCE loss
sbatch --job-name=oversampling-0p2_cqt+mel_1epoch-data_150epochs-only-bce cluster/train_on_slurm.sh hydra.job.name="oversampling-0p2_cqt+mel_1epoch-data_150epochs-training_only-bce" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-0p2_cqt+mel_1epoch/ train.use_batch_norm=True
## 2. oversampling factor 2.0, CQT+MEL, 5 epochs data generation, 150 epochs training, only BCE loss
#sbatch --job-name=oversampling-2p0_cqt+mel_1epoch-data_150epochs-training_only-bce cluster/train_on_slurm.sh hydra.job.name="oversampling-2p0_cqt+mel_5epochs-data_150epochs-training_only-bce" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_cqt+mel_5epochs/ train.use_batch_norm=True

## generate datasets for CQT+MEL features
## 1. oversampling factor 0.2, CQT+MEL, 1 epoch
#sbatch --job-name=oversampling-0p2_cqt+mel_1epoch cluster/generate_data_on_slurm.sh -m hydra.job.name="oversampling-0p2_cqt+mel_1epoch" Mixer.raw_data_path=data/parsed_orig_dataset_oversampling-factor-0.2/ Mixer.pickle_path=data/oversampling-0p2_cqt+mel_1epoch/ Mixer.data_type=validation,test,training Mixer.feature=CQT+MEL Mixer.num_epochs.training=1
## 2. oversampling factor 2.0, CQT+MEL, 5 epochs
#sbatch --job-name=oversampling-2p0_cqt+mel_1epoch cluster/generate_data_on_slurm.sh -m hydra.job.name="oversampling-2p0_cqt+mel_5epochs" Mixer.raw_data_path=data/parsed_orig_dataset_oversampling-factor-2.0/ Mixer.pickle_path=data/oversampling-2p0_cqt+mel_5epochs/ Mixer.data_type=validation,test,training Mixer.feature=CQT+MEL Mixer.num_epochs.training=5

## 2022-02-12
## compare different losses with MEL data
## 1. BCE, BCE per instrument
#sbatch --job-name=bce_bce-per-instr cluster/train_on_slurm.sh hydra.job.name="bce_bce-per-instr" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_mel_5epochs/ train.use_batch_norm=True train.model_checkpoint_interval=5000 train.loss_weights.bce=0.5 train.loss_weights.bce_per_instrument=0.5
## 2. BCE, frobenius
#sbatch --job-name=bce_frobenius cluster/train_on_slurm.sh hydra.job.name="bce_frobenius" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_mel_5epochs/ train.use_batch_norm=True train.model_checkpoint_interval=5000 train.loss_weights.bce=0.5 train.loss_weights.frobenius=0.5
## 3. BCE, compressed_spectral
#sbatch --job-name=bce_compressed_spectral cluster/train_on_slurm.sh hydra.job.name="bce_compressed_spectral" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_mel_5epochs/ train.use_batch_norm=True train.model_checkpoint_interval=5000 train.loss_weights.bce=0.5 train.loss_weights.compressed_spectral=0.5


# train with binary-crossentropy per instrument loss
### 1. oversampling factor 2.0, MEL, 5 epochs data generation, 100 epochs training, BCE loss + per-instr loss
#sbatch --job-name=oversampling-2p0_mel_5epochs-data_100epochs-training_equal-loss-weights cluster/train_on_slurm.sh hydra.job.name="oversampling-2p0_mel_5epochs-data_100epochs-training_equal-loss-weights" train.training_epochs=100 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_mel_5epochs/ train.use_batch_norm=True train.model_checkpoint_interval=5000
### 2. oversampling factor 2.0, CQT, 5 epochs data generation, 100 epochs training, BCE loss + per-instr loss
#sbatch --job-name=oversampling-2p0_cqt_5epochs-data_100epochs-training_equal-loss-weights cluster/train_on_slurm.sh hydra.job.name="oversampling-2p0_cqt_5epochs-data_100epochs-training_equal-loss-weights" train.training_epochs=100 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_cqt_5epochs/ train.use_batch_norm=True train.model_checkpoint_interval=5000
### 3. oversampling factor 2.0, MEL, 5 epochs data generation, 100 epochs training, only per-instr loss
#sbatch --job-name=oversampling-2p0_mel_5epochs-data_100epochs-training_only-per-inst-weights cluster/train_on_slurm.sh hydra.job.name="oversampling-2p0_mel_5epochs-data_100epochs-training_only-per-inst-weights" train.training_epochs=100 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_mel_5epochs/ train.use_batch_norm=True train.model_checkpoint_interval=5000 train.loss_weights.bce=0.0 train.loss_weights.bce_per_instrument=1.0
### 4. oversampling factor 2.0, CQT, 5 epochs data generation, 100 epochs training, only per-instr loss
#sbatch --job-name=oversampling-2p0_cqt_5epochs-data_100epochs-training_only-per-inst-weights cluster/train_on_slurm.sh hydra.job.name="oversampling-2p0_cqt_5epochs-data_100epochs-training_only-per-inst-weights" train.training_epochs=100 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_cqt_5epochs/ train.use_batch_norm=True train.model_checkpoint_interval=5000 train.loss_weights.bce=0.0 train.loss_weights.bce_per_instrument=1.0


## 2022-02-05
## train with all datasets, 0.5 BCE loss + 0.5 MSE loss
## 1. oversampling factor 0.2, MEL, 1 epoch data generation, 150 epochs training
#sbatch --job-name=oversampling-0p2_mel_1epoch-data_150epochs-training_0p5-bce-0p5-mse cluster/train_on_slurm.sh hydra.job.name="oversampling-0p2_mel_1epoch-data_150epochs-training_0p5-bce-0p5-mse" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-0p2_mel_1epoch/ train.use_batch_norm=True train.loss_weights=\[0.5,0.5\]
## 2. oversampling factor 0.2, CQT, 1 epoch data generation, 150 epochs training
#sbatch --job-name=oversampling-0p2_cqt_1epoch-data_150epochs-training_0p5-bce-0p5-mse cluster/train_on_slurm.sh hydra.job.name="oversampling-0p2_cqt_1epoch-data_150epochs-training_0p5-bce-0p5-mse" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-0p2_cqt_1epoch/ train.use_batch_norm=True train.loss_weights=\[0.5,0.5\]
## 3. oversampling factor 0.2, STFT, 1 epoch data generation, 150 epochs training
#sbatch --job-name=oversampling-0p2_stft_1epoch-data_150epochs-training_0p5-bce-0p5-mse cluster/train_on_slurm.sh hydra.job.name="oversampling-0p2_stft_1epoch-data_150epochs-training_0p5-bce-0p5-mse" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-0p2_stft_1epoch/ train.use_batch_norm=True train.loss_weights=\[0.5,0.5\]
## 4. oversampling factor 2.0, MEL, 5 epochs data generation, 150 epochs training
#sbatch --job-name=oversampling-2p0_mel_1epoch-data_150epochs-training_0p5-bce-0p5-mse cluster/train_on_slurm.sh hydra.job.name="oversampling-2p0_mel_5epochs-data_150epochs-training_0p5-bce-0p5-mse" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_mel_5epochs/ train.use_batch_norm=True train.loss_weights=\[0.5,0.5\]
## 5. oversampling factor 2.0, CQT, 5 epochs data generation, 150 epochs training
#sbatch --job-name=oversampling-2p0_cqt_1epoch-data_150epochs-training_0p5-bce-0p5-mse cluster/train_on_slurm.sh hydra.job.name="oversampling-2p0_cqt_5epochs-data_150epochs-training_0p5-bce-0p5-mse" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_cqt_5epochs/ train.use_batch_norm=True train.loss_weights=\[0.5,0.5\]
## 6. oversampling factor 2.0, STFT, 5 epochs data generation, 150 epochs training
#sbatch --job-name=oversampling-2p0_stft_1epoch-data_150epochs-training_0p5-bce-0p5-mse cluster/train_on_slurm.sh hydra.job.name="oversampling-2p0_stft_5epochs-data_150epochs-training_0p5-bce-0p5-mse" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_stft_5epochs/ train.use_batch_norm=True train.loss_weights=\[0.5,0.5\]


## 2022-02-03
## train with all datasets, only BCE loss
## 1. oversampling factor 0.2, MEL, 1 epoch data generation, 150 epochs training, only BCE loss
#sbatch --job-name=oversampling-0p2_mel_1epoch-data_150epochs-training_only-bce cluster/train_on_slurm.sh hydra.job.name="oversampling-0p2_mel_1epoch-data_150epochs-training_only-bce" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-0p2_mel_1epoch/ train.use_batch_norm=True train.loss_weights=\[1.0,0.0\]
## 2. oversampling factor 0.2, CQT, 1 epoch data generation, 150 epochs training, only BCE loss
#sbatch --job-name=oversampling-0p2_cqt_1epoch-data_150epochs-training_only-bce cluster/train_on_slurm.sh hydra.job.name="oversampling-0p2_cqt_1epoch-data_150epochs-training_only-bce" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-0p2_cqt_1epoch/ train.use_batch_norm=True train.loss_weights=\[1.0,0.0\]
## 3. oversampling factor 0.2, STFT, 1 epoch data generation, 150 epochs training, only BCE loss
#sbatch --job-name=oversampling-0p2_stft_1epoch-data_150epochs-training_only-bce cluster/train_on_slurm.sh hydra.job.name="oversampling-0p2_stft_1epoch-data_150epochs-training_only-bce" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-0p2_stft_1epoch/ train.use_batch_norm=True train.loss_weights=\[1.0,0.0\]
## 4. oversampling factor 2.0, MEL, 5 epochs data generation, 150 epochs training, only BCE loss
#sbatch --job-name=oversampling-2p0_mel_1epoch-data_150epochs-training_only-bce cluster/train_on_slurm.sh hydra.job.name="oversampling-2p0_mel_5epochs-data_150epochs-training_only-bce" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_mel_5epochs/ train.use_batch_norm=True train.loss_weights=\[1.0,0.0\]
## 5. oversampling factor 2.0, CQT, 5 epochs data generation, 150 epochs training, only BCE loss
#sbatch --job-name=oversampling-2p0_cqt_1epoch-data_150epochs-training_only-bce cluster/train_on_slurm.sh hydra.job.name="oversampling-2p0_cqt_5epochs-data_150epochs-training_only-bce" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_cqt_5epochs/ train.use_batch_norm=True train.loss_weights=\[1.0,0.0\]
## 6. oversampling factor 2.0, STFT, 5 epochs data generation, 150 epochs training, only BCE loss
#sbatch --job-name=oversampling-2p0_stft_1epoch-data_150epochs-training_only-bce cluster/train_on_slurm.sh hydra.job.name="oversampling-2p0_stft_5epochs-data_150epochs-training_only-bce" train.training_epochs=150 train.batch_size=32 train.gpu_index=0 train.dataset_dir=data/oversampling-2p0_stft_5epochs/ train.use_batch_norm=True train.loss_weights=\[1.0,0.0\]


## 2022-02-02
## generate datasets
## 1. oversampling factor 0.2, MEL, 1 epoch
#sbatch --job-name=oversampling-0p2_mel_1epoch cluster/generate_data_on_slurm.sh -m hydra.job.name="oversampling-0p2_mel_1epoch" Mixer.raw_data_path=data/parsed_orig_dataset_oversampling-factor-0.2/ Mixer.pickle_path=data/oversampling-0p2_mel_1epoch/ Mixer.data_type=validation,test,training Mixer.feature=MEL
## 2. oversampling factor 0.2, CQT, 1 epoch
#sbatch --job-name=oversampling-0p2_cqt_1epoch cluster/generate_data_on_slurm.sh -m hydra.job.name="oversampling-0p2_cqt_1epoch" Mixer.raw_data_path=data/parsed_orig_dataset_oversampling-factor-0.2/ Mixer.pickle_path=data/oversampling-0p2_cqt_1epoch/ Mixer.data_type=validation,test,training Mixer.feature=CQT
## 3. oversampling factor 0.2, STFT, 1 epoch
#sbatch --job-name=oversampling-0p2_stft_1epoch cluster/generate_data_on_slurm.sh -m hydra.job.name="oversampling-0p2_stft_1epoch" Mixer.raw_data_path=data/parsed_orig_dataset_oversampling-factor-0.2/ Mixer.pickle_path=data/oversampling-0p2_stft_1epoch/ Mixer.data_type=validation,test,training Mixer.feature=STFT
## 4. oversampling factor 2.0, MEL, 5 epochs
#sbatch --job-name=oversampling-2p0_mel_1epoch cluster/generate_data_on_slurm.sh -m hydra.job.name="oversampling-2p0_mel_5epochs" Mixer.raw_data_path=data/parsed_orig_dataset_oversampling-factor-2.0/ Mixer.pickle_path=data/oversampling-2p0_mel_5epochs/ Mixer.data_type=validation,test,training Mixer.feature=MEL Mixer.num_epochs.training=5
## 5. oversampling factor 2.0, CQT, 5 epochs
#sbatch --job-name=oversampling-2p0_cqt_1epoch cluster/generate_data_on_slurm.sh -m hydra.job.name="oversampling-2p0_cqt_5epochs" Mixer.raw_data_path=data/parsed_orig_dataset_oversampling-factor-2.0/ Mixer.pickle_path=data/oversampling-2p0_cqt_5epochs/ Mixer.data_type=validation,test,training Mixer.feature=CQT Mixer.num_epochs.training=5
## 6. oversampling factor 2.0, STFT, 5 epochs
#sbatch --job-name=oversampling-2p0_stft_1epoch cluster/generate_data_on_slurm.sh -m hydra.job.name="oversampling-2p0_stft_5epochs" Mixer.raw_data_path=data/parsed_orig_dataset_oversampling-factor-2.0/ Mixer.pickle_path=data/oversampling-2p0_stft_5epochs/ Mixer.data_type=validation,test,training Mixer.feature=STFT Mixer.num_epochs.training=5
