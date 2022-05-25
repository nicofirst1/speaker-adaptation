#!/bin/bash
#SBATCH --reservation=condo_2204047_01
#SBATCH --nodes=1
#SBATCH --job-name=sp-lst
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_shared_jupyter
#SBATCH --gpus-per-node=4



#activating the virtual environment
echo "Activating the virtual environment..."
module load 2021
module load Anaconda3/2021.05
source activate uvapb


#running the actual code
echo "Starting the process..."
python -u ${HOME}/pb_speaker_adaptation/evals/speaker_listener_domains.py
