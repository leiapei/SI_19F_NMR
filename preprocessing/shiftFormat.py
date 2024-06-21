import os
import subprocess
import sys
import time
'''
notes - assuming geo opted coords in directory where is name + _inp, will ask name for input
this code will create a directory name_OrcaInp where the files are formatted xyz files from inp directory
then will run orca dft for shifts where results will be stored in a created directory name_OrcaOut 
'''
name = sys.argv[1]

xyz_directory = name + "_inp"

#create directory for formatted files
orca_inp_directory = name + "_OrcaInp"
os.makedirs(name + '_OrcaInp')

def create_orca_inp(xyzfile):
    file_name = xyzfile
    #remove unecessary top text from file
    with open(os.path.join(xyz_directory, xyzfile), 'r') as xyzfile:
        lines = [line[2:] for line in xyzfile.readlines()[2:]]
    content = ''.join(lines)
    print(type(file_name))
    orcaFile = os.path.splitext(file_name)[0] + ".inp"
    orca_inp = "!B3LYP 6-31G(d,p) NMR\n* xyz 0 1\n" + content + "*\n%EPRNMR\n     NUCLEI = ALL F {SHIFT}\nEND"
    with open(os.path.join(orca_inp_directory, orcaFile), 'w') as file: file.write(orca_inp)


xyz_files = [file for file in os.listdir(xyz_directory) if file.endswith(".xyz") and not file.endswith("trj.xyz")]

for xyzfile in xyz_files:
    create_orca_inp(xyzfile)
