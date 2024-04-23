#!/bin/bash
: << 'COMMENT'
This script is meant to run multiple DKNN trial configurations, based on 
whatever diretory is passed in for processing
COMMENT

counter=0
# assume first argument is the search directory
# assume second argument is the threshold
for entry in "$1"/*
do
  if [[ $entry =~ trial-.*\.json$ ]];
  then
    counter=$((counter+1))
    if [[ "$counter" -ge $2 ]];
    then
      python3 main.py $entry
      echo $counter "Trials completed"
    fi
  fi
done
