import os

xyz_directory = "/Users/tinapthai/Desktop/FNMR/xyzfilesOB"

orca_input_directory = "/Users/tinapthai/Desktop/FNMR/xyzfilesORCA"

os.makedirs(orca_input_directory, exist_ok=True)

#create ORCA input files for each XYZ file
def create_orca_input(xyz_filename):
    with open(os.path.join(xyz_directory, xyz_filename), 'r') as xyz_file:
        xyz_content = xyz_file.read()
        orca_input_content = "!B3LYP 6-31G(d,p) OPT\n* xyz 0 1\n" + xyz_content + "\n*"
    orca_input_filename = os.path.splitext(xyz_filename)[0] + ".inp"
    with open(os.path.join(orca_input_directory, orca_input_filename), 'w') as orca_input_file:orca_input_file.write(orca_input_content)


xyz_files = [file for file in os.listdir(xyz_directory) if file.endswith(".xyz")]

for xyz_file in xyz_files:
    create_orca_input(xyz_file)


