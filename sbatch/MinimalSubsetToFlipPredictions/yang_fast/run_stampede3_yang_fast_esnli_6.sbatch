#!/bin/bash
#SBATCH -J esnli_yang_fast_2700_3150
#SBATCH -o esnli_yang_fast_2700_3150.o%j
#SBATCH -e esnli_yang_fast_2700_3150.e%j
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
python3 main.py --config config/esnli/yang_fast/deberta_mean_with_attention_yang_fast_2700_3150.json
