#!/bin/bash
#SBATCH -J LLamaWrapperBoxESNLI
#SBATCH -o LLamaWrapperBoxESNLI.o%j
#SBATCH -e LLamaWrapperBoxESNLI.e%j
#SBATCH -p normal
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
python3 run_wrappers.py llama_esnli_results.csv
