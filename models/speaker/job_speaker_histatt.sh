#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sppb
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --partition=gpu

#activating the virtual environment
echo "Activating the virtual environment..."
source ${HOME}/venvp/bin/activate

#create output directory
echo "Creating output directory..."
mkdir "${HOME}"/PBSL/speaker_old/buffer_output_hist_att

#running the actual code
echo "Starting the process..."
CUDA_VISIBLE_DEVICES=0 python -u ${HOME}/PBSL/speaker_old/train_speaker_generic.py -metric bert -batch_size 32 -model_type hist_att -reduction sum -subset_size -1 -seed 42 -learning_rate 0.0001 -beam_size 3 -embedding_dim 1024 -shuffle\
	&> ${HOME}/PBSL/speaker_old/buffer_output_hist_att/log_speaker_hist_att_01 &
sleep 60
CUDA_VISIBLE_DEVICES=1 python -u ${HOME}/PBSL/speaker_old/train_speaker_generic.py -metric bert -batch_size 16 -model_type hist_att -reduction sum -subset_size -1 -seed 42 -learning_rate 0.0001 -beam_size 3 -embedding_dim 1024 -shuffle\
	&> ${HOME}/PBSL/speaker_old/buffer_output_hist_att/log_speaker_hist_att_02 &
sleep 60
CUDA_VISIBLE_DEVICES=2 python -u ${HOME}/PBSL/speaker_old/train_speaker_generic.py -dropout 0.3 -metric bert -batch_size 32 -model_type hist_att -reduction sum -subset_size -1 -seed 42 -learning_rate 0.0001 -beam_size 3 -embedding_dim 1024 -shuffle\
	&> ${HOME}/PBSL/speaker_old/buffer_output_hist_att/log_speaker_hist_att_01_DP &
sleep 60
CUDA_VISIBLE_DEVICES=3 python -u ${HOME}/PBSL/speaker_old/train_speaker_generic.py -dropout 0.3 -metric bert -batch_size 16 -model_type hist_att -reduction sum -subset_size -1 -seed 42 -learning_rate 0.0001 -beam_size 3 -embedding_dim 1024 -shuffle\
	&> ${HOME}/PBSL/speaker_old/buffer_output_hist_att/log_speaker_hist_att_02_DP
wait
