#!/bin/bash
#SBATCH -J esnli_yang_slow_struggler
#SBATCH -o esnli_yang_slow_struggler.o%j
#SBATCH -e esnli_yang_slow_struggler.e%j
#SBATCH -p icx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -A CCR24018

source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate wrapperbox
cd /work/06782/ysu707/ls6/WrapperBox/MinimalSubsetToFlipPredictions
python3 main.py --config config/esnli/yang_slow/2184_2223.json
