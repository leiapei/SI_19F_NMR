# Import necessary libraries for data manipulation, machine learning, and plotting
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import ast
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# Initialize list of model names
models = ['MLP', 'SVR', 'GBR', 'GPR', 'RF', 'DT', 'BR', 'ADB', 'ET', 'KNN', 'EN', 'LSO', 'RDG']

# Reads "data_3.csv" into a pandas DataFrame
data = pd.read_csv("data_3.csv")

# Takes a list of neighbor data and flattens it into seperate lists
# for distances, neighbor types, charges, atomic numbers, electronegativities, and masses
def flatten_neighbors_with_types(neighbors):
    flattened_distances = []
    flattened_neighbors = []
    flattened_charges = []
    flattened_nums = []
    flattened_en = []
    flattened_mass = []
    max_length = max(len(neighbor_list) for neighbor_list in neighbors)
    for neighbor_list in neighbors:
        distances = []
        neighbor_types = []
        charges = []
        nums = []
        ens = []
        masses = []
        # Extract all desired features in order of storage in neighbor_list
        for neighbor, distance, charge, num, en, mass in neighbor_list:
            distances.append(distance) 
            neighbor_types.append(neighbor)   
            charges.append(charge)
            nums.append(num)
            ens.append(en)
            masses.append(mass)
        # Pad the lists with zeros to match the length of the longest list; required for regression models
        padding_length = max_length - len(distances)
        distances.extend([0] * padding_length)
        neighbor_types.extend([0] * padding_length)
        charges.extend([0] * padding_length)
        nums.extend([0] * padding_length)
        masses.extend([0] * padding_length)
        ens.extend([0] * padding_length)
        # Append padded data to their "flattened" sets
        flattened_distances.append(distances)
        flattened_neighbors.append(neighbor_types)
        flattened_charges.append(charges)
        flattened_nums.append(nums)
        flattened_en.append(ens)
        flattened_mass.append(masses)
    return flattened_distances, flattened_neighbors, flattened_charges, flattened_nums, flattened_en, flattened_mass

# Calls "flatten_neighbors_with_types" on DataFrame's "neighbors" column
# Initializes variables for resulting flattened lists
distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))

# Encode atom types using one-hot encoding (into one-hot vectors)
encoder = OneHotEncoder(sparse=False)
# Flattened lists converted into numpy arrays
atom_types_encoded = encoder.fit_transform(np.array(neighbors))  # Encode atom types
distances_encoded = np.array(distances)
charges = np.array(charges)
nums = np.array(nums)
en = np.array(en)
mass = np.array(mass)

# Comvines all feature arrays into featur matrix (input data)
X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])

# Sets target variable "y" as chemical shifts column (output data)
y = data['chemical_shift']

# Splits data into 80-20 train-test ratio
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Plots model predictions by loading saved models from pickle files and re-evaluating on the test data
for model in models:
    with open(f"model_files/{model}_3.pkl", "rb") as pickle_file:
        predictor = pickle.load(pickle_file)
    if (model == "RDG"): # For some reason, loaded RDG predictions are different from the original model's predictions
        predictor.fit(X_train, y_train) # Even after training, predictions are slightly different 
    y_pred = predictor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    stderror = y_test.sub(y_pred).abs().sem()
    # Prints MAE and R-squared values
    print(model)
    print(mae)
    print(r2)
    # Displays scatter plot comparing chemical shifts "y_test" to predicted chemical shifts "y_pred"
    plt.scatter(y_test, y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red",linewidth=1, alpha=0.7)
    plt.xlabel("DFT Chemical Shifts")
    plt.xlabel(f"Predicted Chemical Shifts - {model}")
    plt.title(f"Actual vs Predicted Chemical Shifts by {model}")
    plt.savefig(f"model_predictions/{model}_predictions.png")
    plt.show()
