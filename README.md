# SI_19F_NMR

This is the repository containing all the code and data associated with the work "Machine Learning for the Accelerated Prediction of Density Functional Theory Calculated 19F Chemical Shifts Based on Local Atomic Environments". All rationale used when developing all of our software can be found in our Electronic Supporting Information [https://docs.google.com/document/d/1DrFKF79PyoaqLxkP4kbHjVXd9jnKd2svpInhRKLE5OU/edit](ESI) in addition to the methodology section of our manuscript.

## Data folder
This contains all the generated chemical shift values for fluorinated neighbors in environments at 2, 3, 4, and 5 angstroms surrounding the varying target Fluorine atoms for our 501 total compounds titled n_angstrom_shifts.csv respectively. This folder also contains all of our original chemical shift values (generated via ORCA) titled "all_shifts.zip, all of our original fluorinated structures (from the Aspiring Scholars Directed Research Program Cheminventory) titled all_structures.zip, in addition to all of our Ghemical formatted structure files titled "all_structures_ghemical.zip" that represent our 501 compounds in Ghemical format. 

## Graphs folder
This contains all of the code used to generate figures 3-5 in our publication.

## Model files folder
This contains all 13 models that were tested in the project, each saved in .pkl (pickle) format using Python's pickle module.

## Preprocessing folder
This contains all of our preprocessing code, which are explained in detail in our [https://docs.google.com/document/d/1DrFKF79PyoaqLxkP4kbHjVXd9jnKd2svpInhRKLE5OU/edit](ESI).
