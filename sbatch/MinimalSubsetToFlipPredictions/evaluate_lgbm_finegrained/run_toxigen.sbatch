#!/bin/bash
#SBATCH -J toxigen_deberta_large_lgbm_finegrained_st3
#SBATCH -o toxigen_deberta_large_lgbm_finegrained_st3.o%j
#SBATCH -e toxigen_deberta_large_lgbm_finegrained_st3.e%j
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH -A CCR24018
source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate wrapperbox
cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
python3 main.py --config config/toxigen/deberta_mean_with_attention_lgbm_finegrained.json
