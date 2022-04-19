#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=listDM
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4

#activating the virtual environment
echo "Activating the virtual environment..."
source /sw/arch/Debian10/EB_production/2021/software/Anaconda3/2021.05/etc/profile.d/conda.sh
conda activate uvapb

#create output directory
echo "Creating output directory..."
mkdir "${HOME}"/outputs

common_args=( -dropout 0.5 -batch_size 32 -model_type scratch_rrr -embed_type scratch -vectors_file clip.json -reduction sum -subset_size -1 -seed 1 -learning_rate 0.0001 -shuffle)

#running the actual code
echo "Starting the process..."
echo "Launching appliances"

CUDA_VISIBLE_DEVICES=0 python -u ${HOME}/pb_speaker_adaptation/models/listener/train_listener_domains.py -train_domain appliances "${common_args[@]}" \
&> ${HOME}/outputs/appliances.log &
sleep 60

echo "Launching food"

CUDA_VISIBLE_DEVICES=1 python -u ${HOME}/pb_speaker_adaptation/models/listener/train_listener_domains.py -train_domain food  "${common_args[@]}" \
&> ${HOME}/outputs/food.log &
sleep 60

echo "Launching indoor"

CUDA_VISIBLE_DEVICES=2 python -u ${HOME}/pb_speaker_adaptation/models/listener/train_listener_domains.py -train_domain indoor "${common_args[@]}" \
&> ${HOME}/outputs/indoor.log &
sleep 60

echo "Launching outdoor"

CUDA_VISIBLE_DEVICES=3 python -u ${HOME}/pb_speaker_adaptation/models/listener/train_listener_domains.py -train_domain outdoor "${common_args[@]}" \
&> ${HOME}/outputs/outdoor.log


