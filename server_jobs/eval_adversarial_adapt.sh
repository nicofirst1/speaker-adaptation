#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=adv_adapt
#SBATCH --cpus-per-task=1
#SBATCH --time=5:30:00
#SBATCH --partition=gpu_shared
#SBATCH --gpus-per-node=4



# array job can be launched with
# sbatch --array 1-6  eval_speaklist.sh
# or if you want to specify a specific domain use
# sbatch --array=0,2,3 eval_speaklist.sh
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

common_args=( --subset_size -1 --seed 42 --test_split seen)


#running the actual code
echo "Starting the process..."
trainers_file="${HOME}/pb_speaker_adaptation/src/evals/adversarial_adapt.py"

#running the actual code
echo "Starting the process..."

if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
  echo "Launching appliances"
  PYTHONIOENCODING=utf-8 python -u ${trainers_file} --train_domain appliances "${common_args[@]}" 

elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then

  echo "Launching food"

  PYTHONIOENCODING=utf-8 python -u ${trainers_file} --train_domain food "${common_args[@]}"

elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then

  echo "Launching indoor"

  PYTHONIOENCODING=utf-8 python -u ${trainers_file} --train_domain indoor "${common_args[@]}" 
elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]; then

  echo "Launching outdoor"

  PYTHONIOENCODING=utf-8 python -u ${trainers_file} --train_domain outdoor "${common_args[@]}" 
elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]; then

  echo "Launching vehicles"

  PYTHONIOENCODING=utf-8 python -u ${trainers_file} --train_domain vehicles "${common_args[@]}" 
elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]; then
  echo "Launching all"

  PYTHONIOENCODING=utf-8 python -u ${trainers_file} --train_domain all "${common_args[@]}" 
else
  echo "No domain specified for id $SLURM_ARRAY_TASK_ID"
  exit 1

fi
