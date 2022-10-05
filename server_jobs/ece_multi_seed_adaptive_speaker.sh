#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=adapt_speak
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gpus-per-node=1



#activating the virtual environment
echo "Activating the virtual environment..."
source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda activate uvapb

#create output directory
common_args=(
--seed 1
--adapt_lr
0.12
--s_iter
20
--type_of_int
domain
--test_split
seen
--pretrain_loss
ce
--adaptive_loss
ce
--model_type
no_hist
--data_domain
all
--train_domain
food
)




trainers_file="${HOME}/adp/pb_speaker_adaptation/src/evals/adaptive_speaker.py"

#running the actual code
echo "Starting the process..."

python -u ${trainers_file} "${common_args[@]}"
