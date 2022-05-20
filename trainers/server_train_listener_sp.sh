#!/bin/bash
#SBATCH --reservation=condo_2204047_01
#SBATCH --nodes=1
#SBATCH --job-name=listDM
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_shared_jupyter
#SBATCH --gpus-per-node=4



# array job can be launched with
# sbatch --array=0,1 server_train_listener.sh
# the number mapping is the following:
# 0: appliances, food,  indoor,  outdoor
# 1: vehicles, all


#activating the virtual environment
echo "Activating the virtual environment..."
module load 2021
module load Anaconda3/2021.05
source activate uvapb


# define variables
common_args=( --dropout 0.5 --batch_size 64 --model_type scratch_rrr --embed_type scratch --vectors_file vectors.json --reduction sum --subset_size -1 --seed 42 --learning_rate 0.0001 --shuffle)


#running the actual code
echo "Starting the process..."

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
  echo "Launching appliances"
  CUDA_VISIBLE_DEVICES=0  python -u ${HOME}/pb_speaker_adaptation/trainers/listener_train.py --train_domain appliances "${common_args[@]}" &
  sleep 60

  echo "Launching food"
  CUDA_VISIBLE_DEVICES=1  python -u ${HOME}/pb_speaker_adaptation/trainers/listener_train.py --train_domain food "${common_args[@]}" &
  sleep 60

  echo "Launching indoor"
  CUDA_VISIBLE_DEVICES=2  python -u ${HOME}/pb_speaker_adaptation/trainers/listener_train.py --train_domain indoor "${common_args[@]}" &
  sleep 60

  echo "Launching outdoor"
  CUDA_VISIBLE_DEVICES=3  python -u ${HOME}/pb_speaker_adaptation/trainers/listener_train.py --train_domain outdoor "${common_args[@]}" &
  sleep 60

elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
  echo "Launching vehicles"
  CUDA_VISIBLE_DEVICES=0  python -u ${HOME}/pb_speaker_adaptation/trainers/listener_train.py --train_domain vehicles "${common_args[@]}" &
  sleep 60

  echo "Launching all"
  CUDA_VISIBLE_DEVICES=1  python -u ${HOME}/pb_speaker_adaptation/trainers/listener_train.py --train_domain all "${common_args[@]}" &
  sleep 60

else
  echo "No domain specified for id $SLURM_ARRAY_TASK_ID"
  exit 1

fi