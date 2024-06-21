import os
import subprocess
import sys
import time
'''
notes - assuming geo opted coords in directory where is name + _inp, will ask name for input
this code will create a directory name_OrcaInp where the files are formatted xyz files from inp directory
then will run orca dft for shifts where results will be stored in a created directory name_OrcaOut 
'''
name = input("your outputted geo opt files should be stored in a directory name_inp, what is the name for yours (ex: ewa_inp, name = ewa)):")

xyz_directory = "/sli_mldd/" + name + "_inp"

#create directory for formatted files
orca_inp_directory = os.makedirs('/sli_mldd/' + name + '_OrcaInp')

def create_orca_inp(xyzfile):
    #remove unecessary top text from file
    with open(os.path.join(xyz_directory, xyzfile), 'r') as xyzfile:
        lines = xyzfile.readlines()[2:]
    content = ''.join(lines)

    orcaFile = os.path.splitext(xyzfile)[0] + ".inp"
    orca_inp = "! B3LYP 6-31G(d,p)++ NMR\n* xyz 0 1\n" + content + "\n\%EPRNMR\n     NUCLEI = ALL F {{SHIFT}}\nEND"
    with open(os.path.join(orca_inp_directory, orcaFile), 'w') as file: file.write(orca_inp)


xyz_files = [file for file in os.listdir(xyz_directory) if file.endswith(".xyz") and not file.endswith("trj.xyz")]

for xyzfile in xyz_files:
    create_orca_inp(xyzfile)

#FORMAT DONE, AUTOMATE SHIFTS------------------------------------------------------------------------------------------
#input files from orca_inp_directory (name_OrcaInp)

out_directory = os.makedirs("/sli_mldd/" + name + "_OrcaOut")

# Set LD_LIBRARY_PATH
os.environ['LD_LIBRARY_PATH'] = '/opt/orca_5_0_0_linux_x86-64_shared_openmpi411:' + os.environ.get('LD_LIBRARY_PATH', '')

def runOrca(file):
    subprocess.run(['bash', '-c', 'source ~/.bashrc'])
    start = time.time()
    orca_command = ["/opt/orca_5_0_0_linux_x86-64_shared_openmpi411/orca", file]
    with open(os.path.splitext(file)[0] + ".out", 'w') as output_file:
        subprocess.run(orca_command, stdout=output_file)

    output_path = os.path.join(out_directory, output_file)
    # Run checkConvergence.py
    subprocess.run("python3", "checkConvergence.py", output_path)

    end = time.time()
    # Calculate elapsed time
    elapsed_time = end - start
    print(f"Elapsed Time: {elapsed_time} seconds")

for file in os.listdir(orca_inp_directory):
    runOrca(file)
