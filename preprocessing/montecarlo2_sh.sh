#!/bin/bash

directory='MonteCarlo1'

if [ ! -d "$directory" ]; then
    echo "not found"
    exit 1
fi

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
        obconformer 250 100 "$file" > "mc${name}.xyz"
    fi
    ((i+=1))
done
echo {$i}


