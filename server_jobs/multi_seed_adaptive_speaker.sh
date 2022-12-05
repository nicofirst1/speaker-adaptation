#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=adapt_speak
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gpus-per-node=1



#activating the virtual environment
echo "Activating the virtual environment..."
module load 2021
module load Anaconda3/2021.05
source activate uvapb

#create output directory
common_args=(
--attention_dim 1024
--hidden_dim 1024
--embedding_dim 1024
--adapt_lr
0.75
--s_iter
24
--type_of_int
domain
--test_split
seen
--data_domain
all
--train_domain
food
--sim_domain
food
)




trainers_file="${HOME}/pb_speaker_adaptation/src/evals/adaptive_speaker.py"

#running the actual code
echo "Starting the process..."

python -u ${trainers_file} --seed $SLURM_ARRAY_TASK_ID  "${common_args[@]}"
