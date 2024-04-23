#!/bin/bash
#SBATCH -J LlamaToxigen
#SBATCH -o LlamaToxigen.o%j
#SBATCH -e LlamaToxigen.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A CCR21016
source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate DkNN
cd /work/06782/ysu707/ls6/DkNN
deepspeed main.py ./trial_configurations/toxigen/llama/train_deepspeed_steps_diff_weights.json
