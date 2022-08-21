#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=lrn2stir
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gpus-per-node=1



#activating the virtual environment
echo "Activating the virtual environment..."
module load 2021
module load Anaconda3/2021.05
source activate uvapb

# adaptive args
common_args=( --model_type no_hist  --s_iter 5
--pretrain_loss kl --adaptive_loss ce   )

# static arguments
common_args=("${common_args[@]}" --metric accs --reduction sum --subset_size -1  --seed 42
-shuffle --test_split seen --data_domain all --type_of_sim domain )


# train arguments
common_args=("${common_args[@]}" --epochs 200 --patience 50 )

# optimizer arguments
# mlt_type [ GradNorm ,DWA , DTP]
common_args=("${common_args[@]}" --learning_rate 0.001  --adapt_lr 0.3  --mtl_type GradNorm
--mtl_gamma_a 1.2 --mtl_gamma_p 0.8 --mtl_alpha 1.2 --mtl_temp 2.0 --focal_alpha 0.4 --focal_gamma 2.0)


# model arguments
common_args=("${common_args[@]}"  --dropout 0.25  --embedding_dim 1024 --hidden_dim 512 )

trainers_file="${HOME}/pb_speaker_adaptation/src/trainers/learning_to_stir.py"
out_file="learning2stir${SLURM_ARRAY_TASK_ID}.log"

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
