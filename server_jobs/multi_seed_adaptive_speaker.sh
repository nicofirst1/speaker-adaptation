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




trainers_file="${HOME}/pb_speaker_adaptation/src/evals/adaptive_speaker.py"

#running the actual code
echo "Starting the process..."


for SEED in 1 2 3 4 5
do
    python -u ${trainers_file} --seed $SEED "${common_args[@]}"

done
