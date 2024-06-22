#!/bin/bash

directory='MonteCarlo1' # directory where previous files are stored

# executed if file does not exist
if [ ! -d "$directory" ]; then
    echo "not found"
    exit 1
fi

# executed if file exists
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
        echo "double${name}.xyz"
        obconformer 250 100 "$file" > "mc${name}.xyz" # stores newly conformed files in new xyz file
    fi
    ((i+=1))
done
echo {$i}


