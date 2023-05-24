#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=int_pre
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gpus-per-node=2



#activating the virtual environment
echo "Activating the virtual environment..."
module load 2021
module load Anaconda3/2021.05
source activate uvapb

# static arguments
common_args=(--metric accs --reduction sum --subset_size -1  --seed 42
-shuffle --test_split seen --data_domain all --type_of_int domain --mask_oov_embed unk --golden_data_perc 1)
# train arguments
common_args=("${common_args[@]}" --epochs 30 --patience 5 --learning_rate 0.0004 -shuffle )

# model arguments
common_args=("${common_args[@]}"  --dropout 0.0  --embedding_dim 1024 --hidden_dim 1024 --attention_dim 1024 )
# restore the simulator
#common_args=("${common_args[@]}" --resume_train true )

trainers_file="${HOME}/pb_speaker_adaptation/src/trainers/simulator_pretrain.py"

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
