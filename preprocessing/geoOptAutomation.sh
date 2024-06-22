#!/bin/bash
#export LD_LIBRARY_PATH=/opt/orca_5_0_0_linux_x86-64_shared_openmpi411/orca:$LD_LIBRARY_PATH
#source ~/.bashrc

directory="leia_pei_2" # directory where all ORCA input files are stored (.inp), in this case leia_pei_2

# executed in case if directory does not exist
if [ ! -d "$directory" ]; then
  echo "directory not found"
  exit 1
fi

:'
OVERVIEW
1. iterates through each input (.inp) file in the provided directory
2. runs the job using ORCA on ASDRP computer cluster
3. records time elapsed for the current job
'
for file in "$directory"/*; do
  if [ -f "$file" ]; then # executes if file exists
          if [[ $file == *".inp" ]]; then
                start=$(date +%s) # records start time
                ((i++))
                base_name=$(basename "$file") 
                root_name="${base_name%.*}"
                new_ending=".out"
                output_file="${root_name%.*}${new_ending}" # creates file name for output file
                echo "$file $output_file" # displays original file name and the new output file now
                export LD_LIBRARY_PATH=/opt/orca_5_0_0_linux_x86-64_shared_openmpi411:$LD_LIBRARY_PATH # runs jobs using ORCA
                source ~/.bashrc # reloads bash to reflect changes
                /opt/orca_5_0_0_linux_x86-64_shared_openmpi411/orca "$file" > "$output_file" # stores output in output_file
                end=$(date +%s) # records end time
                echo "Elapsed Time: $(($end-$start)) seconds" # calculates & displays total time elapsed for operation
                sleep 10 # pauses for 10 seconds
        fi
  fi
done

