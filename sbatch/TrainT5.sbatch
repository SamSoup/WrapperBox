#!/bin/bash
#SBATCH -J T5ForENSLi
#SBATCH -o T5ForENSLi.o%j
#SBATCH -e T5ForENSLi.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 4:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A CCR21016
source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate DkNN
cd /work/06782/ysu707/ls6/DkNN
python3 main.py ./trial_configurations/esnli/flan-t5-large/train.json