#!/bin/bash
#SBATCH -J deberta_large_lgbm_subset_esnli_7700_8400_normal_12
#SBATCH -o deberta_large_lgbm_subset_esnli_7700_8400_normal_12.o%j
#SBATCH -e deberta_large_lgbm_subset_esnli_7700_8400_normal_12.e%j
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
python3 main.py --config config/esnli/lgbm/deberta_mean_with_attention_lgbm_7700_8400.json
