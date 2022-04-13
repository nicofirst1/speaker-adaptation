#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=listDM
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=5

#activating the virtual environment
echo "Activating the virtual environment..."
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate uvapb

#create output directory
echo "Creating output directory..."
mkdir "${HOME}"/pb_listener_outputs

common_args="-dropout 0.5 -batch_size 32 -model_type scratch_rrr -embed_type scratch -vectors_file clip.json -reduction sum -subset_size -1 -seed 1 -learning_rate 0.0001 -shuffle"

#running the actual code
echo "Starting the process..."
echo "Launching appliances"

CUDA_VISIBLE_DEVICES=0 python -u ${HOME}/pb_speaker_adaptation/listener/train_listener_domains.py -train_domain appliances \
"$common_args" \
&> ${HOME}/pb_listener_outputs/log_listener_scratch_rrr_clip__1_appliances &
sleep 60

echo "Launching food"

CUDA_VISIBLE_DEVICES=1 python -u ${HOME}/pb_speaker_adaptation/listener/train_listener_domains.py -train_domain food \
  "$common_args" \
&> ${HOME}/pb_listener_outputs/log_listener_scratch_rrr_clip__2_food &
sleep 60

echo "Launching indoor"

CUDA_VISIBLE_DEVICES=2 python -u ${HOME}/pb_speaker_adaptation/listener/train_listener_domains.py -train_domain indoor\
   "$common_args" \
&> ${HOME}/pb_listener_outputs/log_listener_scratch_rrr_clip__3_indoor &
sleep 60
echo "Launching outdoor"

CUDA_VISIBLE_DEVICES=3 python -u ${HOME}/pb_speaker_adaptation/listener/train_listener_domains.py -train_domain outdoor \
    "$common_args" \
&> ${HOME}/pb_listener_outputs/log_listener_scratch_rrr_clip__4_outdoor

echo "Launching vehicles"

CUDA_VISIBLE_DEVICES=4 python -u ${HOME}/pb_speaker_adaptation/listener/train_listener_domains.py -train_domain vehicles \
    "$common_args" \
&> ${HOME}/pb_listener_outputs/log_listener_scratch_rrr_clip__4_vehicles


