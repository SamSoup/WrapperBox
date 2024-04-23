#!/bin/bash
#SBATCH -J WrapperLlamaToxigen
#SBATCH -o WrapperLlamaToxigen.o%j
#SBATCH -e WrapperLlamaToxigen.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A CCR21016
source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate DkNN
cd /work/06782/ysu707/ls6/DkNN
python3 run_wrappers.py llama_results_toxigen_all.csv
