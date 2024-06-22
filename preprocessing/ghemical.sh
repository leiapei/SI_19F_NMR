#!/bin/bash

directory='all_structures'

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
        echo "${name}.gpr"
        obabel -ixyz  "$file" -ogpr -O "${name}.gpr"
    fi
    ((i+=1))
done
echo {$i}


