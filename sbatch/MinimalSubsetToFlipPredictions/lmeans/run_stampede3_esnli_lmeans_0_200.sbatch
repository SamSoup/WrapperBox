#!/bin/bash
#SBATCH -J esnli_lmeans_0_200
#SBATCH -o esnli_lmeans_0_200.o%j
#SBATCH -e esnli_lmeans_0_200.e%j
#SBATCH -p skx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -A CCR24018

source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate wrapperbox
cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
python3 main.py --config config/esnli/lmeans/deberta_mean_with_attention_LMeans_0_200.json
