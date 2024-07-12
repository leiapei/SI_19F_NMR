# SI_19F_NMR

This is the repository containing all the code and data associated with the work "Machine Learning for the Accelerated Prediction of Density Functional Theory Calculated 19F Chemical Shifts Based on Local Atomic Environments". All rationale used when developing our code can be found in our Electronic Supporting Information [https://docs.google.com/document/d/1DrFKF79PyoaqLxkP4kbHjVXd9jnKd2svpInhRKLE5OU/edit](ESI) in addition to the methodology section of our manuscript.

## Data folder
This contains all the generated chemical shift values for fluorinated neighbors in environments at 2, 3, 4, and 5 angstroms surrounding the varying target Fluorine atoms for our 501 total compounds titled n_angstrom_shifts.csv respectively. This folder also contains all of our original chemical shift values (generated via ORCA) titled "all_shifts.zip, all of our original fluorinated structures (from the Aspiring Scholars Directed Research Program Cheminventory) titled all_structures.zip, in addition to all of our Ghemical formatted structure files titled "all_structures_ghemical.zip" that represent our 501 compounds in Ghemical format. 

## Graphs folder
This contains all of the code used to generate figures 3-5 in our manuscript.

## Model files folder
This contains all 13 models that were tested in the project, each saved in .pkl (pickle) format using Python's pickle module.

## Preprocessing folder
This contains all of our preprocessing code, which are explained in detail in our [https://docs.google.com/document/d/1DrFKF79PyoaqLxkP4kbHjVXd9jnKd2svpInhRKLE5OU/edit](Electronic Supporting Information). We used the OpenBabel library and the ORCA package during this step. 

## Preprocessing workflow in brief
(More details can be found in our [https://docs.google.com/document/d/1DrFKF79PyoaqLxkP4kbHjVXd9jnKd2svpInhRKLE5OU/edit](Electronic Supporting Information).)

First, 3D structure generation was performed using the gen3D function in OpenBabel. This converted our SMILES strings to initial 3D structures, represented by XYZ coordinates in a .xyz file. All of our XYZ files can be found under the data folder.

Two Monte Carlo conformer searches were then performed using the obconformer command in OpenBabel with parameters 100 and 250 respectively. The code for both runs are stored in montecarlo.sh and montecarlo2_sh.sh under the preprocessing folder. All resultant XYZ coordinates are stored as MonteCarloXYZ.zip under the data folder.

This was then followed by geometry optimization, performed using ORCA's geometry optimization function at the DFT level using the B3LYP functional and the 6-31G(d,p) basis set in the gas phase. The code used for formatting the XYZ files into input files for ORCA geometry optimization is stored in geoFormat.py. The bash script used to automate this process for a large number of files is stored in geoOptAutomation.sh.

Resonances were then extracted using DFT calculations, performed with ORCA. Our specific parameters and commands for this step can be found in our [https://docs.google.com/document/d/1DrFKF79PyoaqLxkP4kbHjVXd9jnKd2svpInhRKLE5OU/edit](Electronic Supporting Information). checkConvergence.py was used to read the resultant XYZ files from geometry optimization and check its convergence status.

## Model training & testing workflow in brief

From the geometry optimized XYZ coordinates, Ghemical files were created using Openbabel. This allowed us to label encode each neighboring atom with features such as atomic number, charge, electronegativity, etc.. These were extracted from neighboring atoms at varying distances (2, 3, 4, or 5 Angstroms) away from the fluorine atom of interest. This generated four datasets, each stored under the data folder as n_angstrom_shifts.csv (where n = 2, 3, 4, 5).

For details on our models and their performances, visit our [https://docs.google.com/document/d/1DrFKF79PyoaqLxkP4kbHjVXd9jnKd2svpInhRKLE5OU/edit](ESI).
