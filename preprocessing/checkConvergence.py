import sys

file_path = sys.argv[1]
converged = False
complete = False
warning = False
print(f"file: {file_path}")
# Function to check if the string "geometry optimization" exists in the file
def check_geometry_optimization(file_path):
    try:
        with open(file_path, 'r') as file:
            #i = 0
            for line in file:
                #i+=1
                global converged
                global complete
                if (converged and complete):
                    break
                if ("optimization run done" in line.lower() or "chemical shielding summary" in line.lower()):
                    #print(line)
                    complete = True
                if ("scf converged after" in line.lower()):
                    #print(line)
                    converged = True
                if ("the optimization did not converge but reached the maximum number" in line.lower()):
                    warning = True
            #print(i)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Check if the string exists in the file
check_geometry_optimization(file_path)

# Display the result
if(converged and complete):
    print('success!')
elif (not(converged or complete) or warning):
    print(f" {file_path} failed")
elif (converged):
    print(f" {file_path} converged, not complete")
elif (complete):
    print(f" {file_path} complete, not converged")
