#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=listDM_2
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gpus-per-node=2

#activating the virtual environment
echo "Activating the virtual environment..."
source /sw/arch/Debian10/EB_production/2021/software/Anaconda3/2021.05/etc/profile.d/conda.sh
conda activate uvapb

#create output directory
echo "Creating output directory..."
out_dir="${HOME}"/outputs
mkdir "${out_dir}"

common_args=( -dropout 0.5 -batch_size 32 -model_type scratch_rrr -embed_type scratch -vectors_file clip.json -reduction sum -subset_size -1 -seed 1 -learning_rate 0.0001 -shuffle)

#running the actual code
echo "Starting the process..."
echo "Launching vehicles"

CUDA_VISIBLE_DEVICES=0 python -u ${HOME}/pb_speaker_adaptation/models/listener/train_listener_domains.py -train_domain vehicles "${common_args[@]}" \
&> ${out_dir}/vehicles.log &
sleep 60

echo "Launching all"

CUDA_VISIBLE_DEVICES=1 python -u ${HOME}/pb_speaker_adaptation/models/listener/train_listener_domains.py -train_domain all  "${common_args[@]}" \
&> ${out_dir}/all.log &
sleep 60

