#!/bin/bash
#SBATCH -J TestGPU
#SBATCH -o TestGPU.o%j
#SBATCH -e TestGPU.e%j
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:05:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A CCR21016
source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate DkNN
cd /work/06782/ysu707/ls6/DkNN
python3 test_gpu.py
