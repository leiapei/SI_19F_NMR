#!/bin/bash
#export LD_LIBRARY_PATH=/opt/orca_5_0_0_linux_x86-64_shared_openmpi411/orca:$LD_LIBRARY_PATH
#source ~/.bashrc
echo "running"
directory=$1
echo "Directory: $directory"

if [ ! -d "$directory" ]; then
  echo "directory not found"
  exit 1
fi

:

for file in "$directory"/*; do
  if [ -f "$file" ]; then
          if [[ $file == *".inp" ]]; then
                start=$(date +%s)
                ((i++))
                base_name=$(basename "$file")
                root_name="${base_name%.*}"
                new_ending=".out"
                output_file="${root_name%.*}${new_ending}"
                echo "$file $directory/$output_file"
                export LD_LIBRARY_PATH=/opt/orca_5_0_0_linux_x86-64_shared_openmpi411:$LD_LIBRARY_PATH
                source ~/.bashrc
                #ls /opt/orca_5_0_0_linux_x86-64_shared_openmpi411/
                /opt/orca_5_0_0_linux_x86-64_shared_openmpi411/orca "$file" > "$directory/$output_file"
                python3 checkConvergence.py "$directory/$output_file"
                end=$(date +%s)
                echo "Elapsed Time: $(($end-$start)) seconds"
                sleep 10
        fi
  fi
done
