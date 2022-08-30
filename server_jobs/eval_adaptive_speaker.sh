#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=adapt_speak
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --partition=gpu_shared
#SBATCH --gpus-per-node=1



#activating the virtual environment
echo "Activating the virtual environment..."
module load 2021
module load Anaconda3/2021.05
source activate uvapb

#create output directory
common_args=( --seed 69
--adapt_lr
0.03
--s_iter
5
--type_of_int
domain
--test_split
seen
--pretrain_loss
fbce
--adaptive_loss
bce
--model_type
binary
--data_domain
all
)




trainers_file="${HOME}/pb_speaker_adaptation/src/evals/adaptive_speaker.py"

#running the actual code
echo "Starting the process..."


if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
  echo "Launching appliances"
  python -u ${trainers_file} --train_domain appliances "${common_args[@]}"

elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then

  echo "Launching food"

  python -u ${trainers_file} --train_domain food "${common_args[@]}"

elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then

  echo "Launching indoor"

  python -u ${trainers_file} --train_domain indoor "${common_args[@]}"

elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]; then

  echo "Launching outdoor"

  python -u ${trainers_file} --train_domain outdoor "${common_args[@]}"

elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]; then

  echo "Launching vehicles"

  python -u ${trainers_file} --train_domain vehicles "${common_args[@]}"

elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]; then
  echo "Launching all"

  python -u ${trainers_file} --train_domain all "${common_args[@]}"

else
  echo "No domain specified for id $SLURM_ARRAY_TASK_ID"
  exit 1

fi
