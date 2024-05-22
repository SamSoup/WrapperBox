#!/bin/bash

# Find and submit all .sbatch files in the current directory
for file in *.sbatch; do
  if [ -f "$file" ]; then
    sbatch "$file"
  fi
done
