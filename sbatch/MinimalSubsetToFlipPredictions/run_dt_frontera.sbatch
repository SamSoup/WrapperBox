#!/bin/bash
#SBATCH -J deberta_large_dt_minimal_subset_frontera
#SBATCH -o deberta_large_dt_minimal_subset_frontera.o%j
#SBATCH -e deberta_large_dt_minimal_subset_frontera.e%j
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A CCR24018
source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate DkNN
cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
python3 main.py --config config/toxigen/deberta_mean_with_attention_dt.json
