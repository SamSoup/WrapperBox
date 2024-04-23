#!/bin/bash
#SBATCH -J RunBoostrap
#SBATCH -o RunBoostrap.o%j
#SBATCH -e RunBoostrap.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 6:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A CCR21016
source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate DkNN
cd /work/06782/ysu707/ls6/DkNN/bootstrap
python3 run_bootstrap.py
