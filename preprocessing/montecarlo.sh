#!/bin/bash

directory='630outfiles' # directory where all SMILES files were stored

# executes if directory doesn't exist
if [ ! -d "$directory" ]; then
    echo "not found"
    exit 1
fi

# iterates through each file in 630_outfiles
for file in "$directory"/*; do
  if [ -f "$file" ]; then
    echo "$file"
    break
  fi
done

i=1
for file in "$directory"/*; do
    if [ -f "$file" ]; then
        echo "$file"
        name=${file##*/}
        name=${name%.*}
        echo ${name}
        echo "montecarlo${name}.xyz"
        obconformer 250 100 "$file" > "montecarlo${name}.xyz" # calling obconformer & placing file into corresponding montecarlo__.xyz file
    fi
    ((i+=1))
done
echo {$i}

