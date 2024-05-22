#!/bin/bash
#SBATCH -J deberta_large_lmeans_subset_esnli_2100_2800_normal_4
#SBATCH -o deberta_large_lmeans_subset_esnli_2100_2800_normal_4.o%j
#SBATCH -e deberta_large_lmeans_subset_esnli_2100_2800_normal_4.e%j
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 48:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A CCR24018
source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate wrapperbox
cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
python3 main.py --config config/esnli/lmeans/deberta_mean_with_attention_LMeans_2100_2800.json
