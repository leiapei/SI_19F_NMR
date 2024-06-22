# Import necessary modules 
import math
import os 
import pandas as pd
import warnings
from mendeleev import element 

# Suppress all warnings for a cleaner script output
warnings.filterwarnings("ignore")

# Function to read file content from file specified (file_path)
def extract_shifts(file_path):
    with open(file_path, "r") as file:
        file_content = file.read()
    # "CHEMICAL SHIELDING SUMMARY" is the orca output that indicates the isotropic shielding report will follow 
    # Extracts text from that point onwards
    chemical_shielding_index = file_content.find("CHEMICAL SHIELDING SUMMARY")
    text_after_chemical_shielding = file_content[chemical_shielding_index:]
    # Split the text into lines and strip leading/trailing whitespace
    # Skips first line (header)
    lines = text_after_chemical_shielding.split("\n")
    lines = [line.strip() for line in lines]
    lines = lines[1:]
    # Initialize an empty list (numbers) to store the extracted numbers
    numbers = []
    # Loop through the lines and extract the numbers below "Isotropic" to get isotropic shielding
    for line in lines:
        if line.strip().startswith("-") or line.strip().startswith("Nucleus"):
            continue  # Skip header lines and separator lines
        if (line.strip().startswith("Maximum")):
            break
        if line.strip():  # Check if the line is not empty
            columns = line.split()
            isotropic_value = columns[2]  # Extract the isotropic value (third column)
            numbers.append(float(isotropic_value))
    # Store the extracted numbers in numbers list
    return numbers

# Function to read XYZ files and extract atomic coordinates
def read_ghemical(file_path):
    # read atomic data from corresponding ".gpr" file from file_path parameter
    _, file_name = os.path.split(file_path)
    basename, _ = os.path.splitext(file_name)
    gpr_file = os.path.join("all_structures_ghemical", f'{basename}.gpr')
    # initializes lists to store charges and atoms, and creates flags (booleans) for data collection
    charges = []
    atoms = []
    a = False
    c = False
    # opens ".gpr" file, iterates through each line. Uses flags to control atom/charge collection
    with open(gpr_file, 'r') as file:
        for line in file:
            # Line starts mark when to begin collecting atoms/charges in the file (or whether to continue collecting)
            if line.startswith('!Atoms'):
                a = True
                c = False
            elif line.startswith('!Charges'):
                a = False
                c = True
            elif line.startswith('!'):
                a = False
                c = False
                # Skip other information sections
                continue
            # Process charge and atom data depending on flags
            else:
                if c:
                    # Extract charges for list charges
                    _, charge = map(float, line.split())
                    charges.append(charge)
                elif a:
                    # Extract atoms for list atoms
                    _, atomic_number = map(int, line.split())
                    atoms.append(atomic_number)
    # returns list charges and list atoms
    return charges, atoms

#reads atomic coordinates from an xyz file using specific marker
def read_xyz_file(file_path):
    coordinates_index = 2
    with open(file_path, "r") as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            if ("Coordinates from ORCA-job" in line):
                coordinates_index = i + 1
            i += 1
    # Initializese lists that store atomic coordinates, types, charges, atomic numbers, masses, and electronegativities
    # Extract atomic coordinates from xyz_file
    atomic_coordinates = []
    atom_types = []
    # Extract charges and atomic numbers from the ghemical files 
    charges, atomic_numbers = read_ghemical(file_path)
    masses = []
    ens = []
    # Extracting coordinates and atom types from each line and appends them to lists
    for line in lines[coordinates_index:]:
        if line.strip():  # Check if the line is not empty
            parts = line.split() #If the line is not empty, it must contain an xyz coordinate
            atom_type = parts[0]
            x, y, z = map(float, parts[1:])
            atomic_coordinates.append([x, y, z])
            atom_types.append(atom_type)
            # Gets element electronegativity and mass
            el = element(atom_type)
            ens.append(el.en_pauling)
            masses.append(el.mass_number)
    # Declares list for each different angstrom level
    f_neighbors_5 = []
    f_neighbors_4 = []
    f_neighbors_3 = []
    f_neighbors_2 = []
    index = 0
    # Loops through atom types, identifies fluorine atoms,
    # calculates their distance from neighbors, and stores properties
    for i in range(len(atom_types)):
        if (atom_types[i] is "F"):
            # Identify fluorine atoms
            x = atomic_coordinates[i][0]
            y = atomic_coordinates[i][1]
            z = atomic_coordinates[i][2]
            f_neighbors_5.append(list())
            f_neighbors_4.append(list())
            f_neighbors_3.append(list())
            f_neighbors_2.append(list())
            for j in range(len(atom_types)):
                if (j != i):
                    en = ens[j]
                    mass = masses[j]
                    # Euclidean distance formula from every other atom to the given fluorine 
                    dist_x = (atomic_coordinates[j][0] - x) ** 2
                    dist_y = (atomic_coordinates[j][1] - y) ** 2
                    dist_z = (atomic_coordinates[j][2] - z) ** 2
                    distance = math.sqrt(dist_x + dist_y + dist_z)
                    # Varying conditions for different angstrom level sets
                    if (distance <= 5):
                        # Features: Neighbor identity, distance from the fluorine, charge, atomic number, electronegativity, mass number
                        f_neighbors_5[index].append([atom_types[j], distance, charges[j], atomic_numbers[j], en, mass])
                    if (distance <= 4):
                        f_neighbors_4[index].append([atom_types[j], distance, charges[j], atomic_numbers[j], en, mass])
                    if (distance <= 3):
                        f_neighbors_3[index].append([atom_types[j], distance, charges[j], atomic_numbers[j], en, mass])
                    if (distance <= 2):
                        f_neighbors_2[index].append([atom_types[j], distance, charges[j], atomic_numbers[j], en, mass])
            index += 1
    # returns lists appended with atom features
    return f_neighbors_5, f_neighbors_4, f_neighbors_3, f_neighbors_2
# Finds maximum number of fluorine chemical shifts across file directories
def count_max_F(directory):
    maximum = -1
    # Loops through files to extract shifts and updates maximum count
    for filename in os.listdir(directory):
        basename, extension = os.path.splitext(filename)
        shift_path = os.path.join("all_shifts", f'{basename}.out')
        shifts = extract_shifts(shift_path)
        maximum = max(maximum, len(shifts))
    return maximum
# Processes all xyz files in a directory using data frames
def process_xyz_files(directory):
    # Process all XYZ files into a directory, intialize DataFrames for neighbors and chemical shifts
    data_2 = pd.DataFrame(columns=['neighbors', 'chemical_shift'])
    data_3 = pd.DataFrame(columns=['neighbors', 'chemical_shift'])
    data_4 = pd.DataFrame(columns=['neighbors', 'chemical_shift'])
    data_5 = pd.DataFrame(columns=['neighbors', 'chemical_shift'])
    print("Processing...")
    # Reads each file in directory, extracts shifts
    for filename in os.listdir(directory):
        basename, extension = os.path.splitext(filename)
        file_path = os.path.join(directory, filename)
        shift_path = os.path.join("all_shifts", f'{basename}.out')
        fn5, fn4, fn3, fn2 = read_xyz_file(file_path)
        shifts = extract_shifts(shift_path)
        # Add the fluorine neighbors and the chemical shifts to their respective datasets; one neighbor-set + shift pair per fluorine atom
        for i in range(len(fn2)):
            row = {
                'neighbors': fn2[i],  
                'chemical_shift': shifts[i]  
            }
            data_2 = data_2._append(row, ignore_index=True)
        for i in range(len(fn3)):
            row = {
                'neighbors': fn3[i], 
                'chemical_shift': shifts[i]  
            }
            data_3 = data_3._append(row, ignore_index=True)
        for i in range(len(fn4)):
            row = {
                'neighbors': fn4[i],  
                'chemical_shift': shifts[i]  
            }
            data_4 = data_4._append(row, ignore_index=True)
        for i in range(len(fn5)):
            row = {
                'neighbors': fn5[i],  
                'chemical_shift': shifts[i] 
            }
            data_5 = data_5._append(row, ignore_index=True)
    print("Processing complete")
    return data_5, data_4, data_3, data_2

# Processing csv file data
directory = "all_structures"
data_5, data_4, data_3, data_2 = process_xyz_files(directory)
data_5.to_csv("data_5.csv")
data_4.to_csv("data_4.csv")
data_3.to_csv("data_3.csv")
data_2.to_csv("data_2.csv")
