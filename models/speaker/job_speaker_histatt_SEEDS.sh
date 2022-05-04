#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=4hist_att
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --reservation=condo_raquel

#activating the virtual environment
echo "Activating the virtual environment..."
source ${HOME}/venvp/bin/activate

#create output directory
echo "Creating output directory..."
mkdir "${HOME}"/PBSL/speaker_old/buffer_output_SELECTED_hist_att

#running the actual code
echo "Starting the process..."
CUDA_VISIBLE_DEVICES=0 python -u ${HOME}/PBSL/speaker_old/train_speaker_generic.py -dropout 0.3 -metric bert -batch_size 16 -model_type hist_att -reduction sum -subset_size -1 -seed 1 -learning_rate 0.0001 -beam_size 3 -embedding_dim 1024 -shuffle\
	&> ${HOME}/PBSL/speaker_old/buffer_output_SELECTED_hist_att/log_speaker_hist_att_SELECTED_1 &
sleep 60
CUDA_VISIBLE_DEVICES=1 python -u ${HOME}/PBSL/speaker_old/train_speaker_generic.py -dropout 0.3 -metric bert -batch_size 16 -model_type hist_att -reduction sum -subset_size -1 -seed 2 -learning_rate 0.0001 -beam_size 3 -embedding_dim 1024 -shuffle\
	&> ${HOME}/PBSL/speaker_old/buffer_output_SELECTED_hist_att/log_speaker_hist_att_SELECTED_2 &
sleep 60
CUDA_VISIBLE_DEVICES=2 python -u ${HOME}/PBSL/speaker_old/train_speaker_generic.py -dropout 0.3 -metric bert -batch_size 16 -model_type hist_att -reduction sum -subset_size -1 -seed 3 -learning_rate 0.0001 -beam_size 3 -embedding_dim 1024 -shuffle\
	&> ${HOME}/PBSL/speaker_old/buffer_output_SELECTED_hist_att/log_speaker_hist_att_SELECTED_3 &
sleep 60
CUDA_VISIBLE_DEVICES=3 python -u ${HOME}/PBSL/speaker_old/train_speaker_generic.py -dropout 0.3 -metric bert -batch_size 16 -model_type hist_att -reduction sum -subset_size -1 -seed 4 -learning_rate 0.0001 -beam_size 3 -embedding_dim 1024 -shuffle\
	&> ${HOME}/PBSL/speaker_old/buffer_output_SELECTED_hist_att/log_speaker_hist_att_SELECTED_4
wait
