import pandas as pd

# Initialize the DataFrame
nmrdb_f = pd.DataFrame(columns=["SMILES", "Shifts", "Atom#", "Solvent"])

# Read the file content
with open("/Users/tinapthai/Code/nmrshiftdb2.nmredata.sd", 'r') as file:
    content = file.read()

# Split the content by "$$$$" to process each compound separately
compounds = content.split('$$$$\n')

print(f"# of compounds: " + str(len(compounds)) + "....processing pls wait")

fcount = 0

for compound in compounds:
    lines = compound.splitlines()
    if '<NMREDATA_1D_19F>' in compound:
        fcount+=1
        #extract SMILES
        smiles_start = compound.find('<NMREDATA_SMILES>') + len('<NMREDATA_SMILES>\n')
        smiles_end = compound.find('\n>',smiles_start)
        smiles = compound[smiles_start:smiles_end].strip()

        #extract solvent
        solvent_start = compound.find('<NMREDATA_SOLVENT>') + len('<NMREDATA_SMILES>\n')
        solvent_end = compound.find('\n>',solvent_start)
        solvent = compound[solvent_start:solvent_end].strip().replace('\\','')

        #extract fluorine shifts
        shifts = []
        s = []
        shifts_start = compound.find('<NMREDATA_1D_19F>') + len('<NMREDATA_1D_19F>\n')
        shifts_end = compound.find('\n>', shifts_start)
        shifts_section = compound[shifts_start:shifts_end].strip()
        
        shifts_lines = shifts_section.split('\n')
        for line in shifts_lines:
            if ',' in line:
                shift_value, label = line.split(', L=')
                shifts.append(shift_value.strip())
                label = label.replace('\\','')
                s.append(label.strip())


        #extract atom numbers
        atom_numbers = []
        assignment_start = compound.find('<NMREDATA_ASSIGNMENT>') + len('<NMREDATA_ASSIGNMENT>\n')
        assignment_end = compound.find('\n>', assignment_start)
        assignment_section = compound[assignment_start:assignment_end].strip()
        
        assignment_lines = assignment_section.split('\n')
        for line in assignment_lines:
            parts = line.split(',')
            if parts[0] in s:
                shift_nums = []
                if len(parts) > 2:
                    atoms = parts[2:]
                    for atom in atoms:
                        try:
                            atom = atom.replace('\\','').strip()
                            shift_nums.append(atom)
                        except:
                            continue
                else:
                    shift_nums.append(parts[1])
                atom_numbers.append(shift_nums)


        #append to df
        new_row = pd.DataFrame([{
            "SMILES": smiles,
            "Shifts": shifts,
            "Atom#": atom_numbers,
            "Solvent": solvent
        }])

        nmrdb_f = pd.concat([nmrdb_f, new_row], ignore_index=True)

print(fcount)
nmrdb_f.to_csv("/Users/tinapthai/Code/nmrshiftdb2_19fdata.csv", index=False)
