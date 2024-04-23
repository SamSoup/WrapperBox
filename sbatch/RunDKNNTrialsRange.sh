#!/bin/bash
: << 'COMMENT'
This script is meant to run multiple DKNN trial configurations, based on 
whatever diretory is passed in for processing

Usage: run <start> <end> <directory of trials>
COMMENT

counter=0
# assume first argument is the search directory
# assume second argument is the threshold
for (( c=$1; c<=$2; c++ ))
do
    python3 main.py $3trial-$c.json
	# echo $3trial-$c.json
done
