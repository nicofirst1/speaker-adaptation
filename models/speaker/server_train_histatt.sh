#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sppb
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

#activating the virtual environment
echo "Activating the virtual environment..."
source /sw/arch/Debian10/EB_production/2021/software/Anaconda3/2021.05/etc/profile.d/conda.sh
conda activate uvapb

#create output directory
echo "Creating output directory..."
out_dir="${HOME}"/speaker_train
common_args=(-dropout 0.3 -batch_size 32 -model_type hist_att -metric bert -beam_size 3 -reduction sum -subset_size -1 -seed 42 -learning_rate 0.0001 -shuffle -embedding_dim 1024)


mkdir -p "${out_dir}"
out_file="${out_dir}"/speaker_hist_att_"${SLURM_ARRAY_TASK_ID}".log

#running the actual code
echo "Starting the process..."
python -u ${HOME}/pb_speaker_adaptation/models/speaker/train_speaker_generic.py  "${common_args[@]}" #&> "${out_file}"
