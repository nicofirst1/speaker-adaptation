#!/bin/bash
#SBATCH --reservation=condo_2204047_01
#SBATCH --nodes=1
#SBATCH --job-name=speak_train
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gpus-per-node=4



#activating the virtual environment
echo "Activating the virtual environment..."
module load 2021
module load Anaconda3/2021.05
source activate uvapb

#create output directory
echo "Creating output directory..."
common_args=( --dropout 0.3 --batch_size 32 --model_type hist_att --metric bert --beam_size 3 --reduction sum --subset_size -1 --seed 33 --learning_rate 0.0001 -shuffle --embedding_dim 1024)
restore_arg=( --resume_train "adaptive-speaker/speaker/SpeakerModelHistAtt:latest" )

#running the actual code
echo "Starting the process..."
python -u ${HOME}/pb_speaker_adaptation/src/trainers/speaker_train.py  "${common_args[@]}" > "speaker_train.log"
