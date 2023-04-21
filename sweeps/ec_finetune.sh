#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=sweep_sim
#SBATCH --cpus-per-task=1
#SBATCH --time=35:00:00
#SBATCH --partition=gpu_shared
#SBATCH --gpus-per-node=1



#activating the virtual environment
echo "Activating the virtual environment..."
module load 2021
module load Anaconda3/2021.05
source activate uvapb

# perform sweep
common_args=( --sweep_file ./ec_finetune_sweep.json --episodes 2048 --batch_size 32 --patience 5 --epochs 400 --train_domain all)


trainers_file="${HOME}/speaker-adaptation/sweeps/array_sweep.py"

#running the actual code
echo "Starting the process..."

python -u ${trainers_file}  "${common_args[@]}"
