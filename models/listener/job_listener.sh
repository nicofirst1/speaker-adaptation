#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=listener
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4

#activating the virtual environment
echo "Activating the virtual environment..."
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate uvapb

#create output directory
echo "Creating output directory..."
mkdir "${HOME}"/pb_listener_outputs

#running the actual code
echo "Starting the process..."
CUDA_VISIBLE_DEVICES=0 python -u ${HOME}/pb_speaker_adaptation/listener/train_listener_bert_CE_generic.py -dropout 0.5 -batch_size 32 -bert_type base -model_type bert_att_ctx_hist -reduction sum -subset_size -1 -seed 1 -learning_rate 0.0001 -shuffle\
	&> ${HOME}/pb_listener_outputs/log_listener_bert_att_ctx_hist1DP05 &
sleep 60
CUDA_VISIBLE_DEVICES=1 python -u ${HOME}/pb_speaker_adaptation/listener/train_listener_bert_CE_generic.py -dropout 0.5 -batch_size 32 -bert_type base -model_type bert_att_ctx_hist -reduction sum -subset_size -1 -seed 2 -learning_rate 0.0001 -shuffle\
	&> ${HOME}/pb_listener_outputs/log_listener_bert_att_ctx_hist2DP05 &
sleep 60
CUDA_VISIBLE_DEVICES=2 python -u ${HOME}/pb_speaker_adaptation/listener/train_listener_bert_CE_generic.py -dropout 0.5 -batch_size 32 -bert_type base -model_type bert_att_ctx_hist -reduction sum -subset_size -1 -seed 3 -learning_rate 0.0001 -shuffle\
	&> ${HOME}/pb_listener_outputs/log_listener_bert_att_ctx_hist3DP05 &
sleep 60
CUDA_VISIBLE_DEVICES=3 python -u ${HOME}/pb_speaker_adaptation/listener/train_listener_bert_CE_generic.py -dropout 0.5 -batch_size 32 -bert_type base -model_type bert_att_ctx_hist -reduction sum -subset_size -1 -seed 4 -learning_rate 0.0001 -shuffle\
	&> ${HOME}/pb_listener_outputs/log_listener_bert_att_ctx_hist4DP05
wait

