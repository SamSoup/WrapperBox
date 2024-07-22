#!/bin/bash
#SBATCH -J esnli_stella_instruct_embeddings
#SBATCH -o esnli_stella_instruct_embeddings.o%j
#SBATCH -e esnli_stella_instruct_embeddings.e%j
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH --mail-user=sam.su@utexas.edu
#SBATCH --mail-type=ALL
#SBATCH -A CCR24018

source /work/06782/ysu707/ls6/conda/etc/profile.d/conda.sh
conda activate wrapperbox
cd /work/06782/ysu707/ls6/WrapperBox/ComputeRepresentations/SentenceTransformers
python3 main.py --config configs/esnli_stella_instruct.json
