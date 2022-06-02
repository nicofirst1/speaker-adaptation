#!/bin/bash
#SBATCH --reservation=condo_2204047_01
#SBATCH --nodes=1
#SBATCH --job-name=listDM
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gpus-per-node=1

# array job can be launched with
# sbatch --array 1-6  server_train_listener.sh
# or if you want to specify a specific domain use
# sbatch --array=0,2,3 server_train_listener.sh
# the number mapping is the following:
# 1: appliances
# 2: food
# 3: indoor
# 4: outdoor
# 5: vehicles
# 6: all


#activating the virtual environment
echo "Activating the virtual environment..."
module load 2021
module load Anaconda3/2021.05
source activate uvapb

# define variables
common_args=( --dropout 0.5 --batch_size 64 --model_type scratch_rrr --embed_type scratch --vectors_file vectors.json --reduction sum --subset_size -1 --seed 42 --learning_rate 0.0001 --shuffle)

trainers_file="${HOME}/pb_speaker_adaptation/src/trainers/listener_train.py"

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