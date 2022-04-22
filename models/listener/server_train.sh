#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=listDM
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

# array job can be launched with
# sbatch --array 1-6 server_train.sh
# or if you want to specify a specific domain use
# sbatch --array=0,2,3 server_train.sh
# the number mapping is the following:
# 1: appliances
# 2: food
# 3: indoor
# 4: outdoor
# 5: vehicles
# 6: all


#activating the virtual environment
echo "Activating the virtual environment..."
source /sw/arch/Debian10/EB_production/2021/software/Anaconda3/2021.05/etc/profile.d/conda.sh
conda activate uvapb

# define variables
out_dir="${HOME}"/outputs
common_args=(-dropout 0.5 -batch_size 64 -model_type scratch_rrr -embed_type scratch -vectors_file vectors.json -reduction sum -subset_size -1 -seed 42 -learning_rate 0.0001 -shuffle)

#running the actual code
echo "Starting the process..."

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
  #create output directory
  echo "Creating output directory..."
  out_dir="${HOME}"/outputs
  rm -rf "${out_dir}"
  mkdir "${out_dir}"

elif [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
  echo "Launching appliances"
  python -u ${HOME}/pb_speaker_adaptation/models/listener/train_listener_domains.py -train_domain appliances "${common_args[@]}" \
    &>${out_dir}/appliances.log &

elif [[ $SLURM_ARRAY_TASK_ID -eq 2 ]]; then

  echo "Launching food"

  python -u ${HOME}/pb_speaker_adaptation/models/listener/train_listener_domains.py -train_domain food "${common_args[@]}" \
    &>${out_dir}/food.log &

elif [[ $SLURM_ARRAY_TASK_ID -eq 3 ]]; then

  echo "Launching indoor"

  python -u ${HOME}/pb_speaker_adaptation/models/listener/train_listener_domains.py -train_domain indoor "${common_args[@]}" \
    &>${out_dir}/indoor.log &

elif [[ $SLURM_ARRAY_TASK_ID -eq 4 ]]; then

  echo "Launching outdoor"

  python -u ${HOME}/pb_speaker_adaptation/models/listener/train_listener_domains.py -train_domain outdoor "${common_args[@]}" \
    &>${out_dir}/outdoor.log

elif [[ $SLURM_ARRAY_TASK_ID -eq 5 ]]; then

  echo "Launching vehicles"

  python -u ${HOME}/pb_speaker_adaptation/models/listener/train_listener_domains.py -train_domain vehicles "${common_args[@]}" \
    &>${out_dir}/vehicles.log &

elif [[ $SLURM_ARRAY_TASK_ID -eq 6 ]]; then
  echo "Launching all"

  python -u ${HOME}/pb_speaker_adaptation/models/listener/train_listener_domains.py -train_domain all "${common_args[@]}" \
    &>${out_dir}/all.log &

else
  echo "No domain specified for id $SLURM_ARRAY_TASK_ID"
  exit 1

fi
