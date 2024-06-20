import pandas as pd
import time
import warnings
import pickle
from bayes_opt import BayesianOptimization
from hyperopt import fmin, tpe, hp, Trials

from skopt import BayesSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer, Categorical

#Import all the models
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

# Suppress all warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import ast
np.int = int
from sklearn.metrics import mean_absolute_error, r2_score

#Read in all the datasets for each angstrom level
data_5 = pd.read_csv("data_5.csv")
data_4 = pd.read_csv("data_4.csv")
data_3 = pd.read_csv("data_3.csv")
data_2 = pd.read_csv("data_2.csv")

# Function to "flatten" all input features for each fluorine atom's neighbors (distance, id, charge, atomic number, electronegativity, mass number)
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

#Additional preprocessing; declare parameters and results dataframes to store corresponding values
data_list = [data_5, data_4, data_3, data_2]
models = ['MLP', 'SVR', 'GBR', 'GPR', 'RF', 'DT', 'BR', 'ADB', 'ET', 'KNN', 'EN', 'LSO', 'RDG']
parameters = pd.DataFrame(columns=["Model", "5 angstroms", "4 angstroms", "3 angstroms", "2 angstroms"])
results = pd.DataFrame(columns=["Model", "Metric", "5 angstroms", "4 angstroms", "3 angstroms", "2 angstroms"])
stderr = pd.DataFrame(columns=["Model", "Metric", "5 angstroms", "4 angstroms", "3 angstroms", "2 angstroms"])
models_column = list()
metrics_column = list()
#For each model run, must store MAE, R^2, and Runtime
for model in models:
    models_column.append(model)
    models_column.append(model)
    models_column.append(model)
    metrics_column.append("MAE")
    metrics_column.append("R^2")
    metrics_column.append("Runtime")
results["Model"] = models_column
results["Metric"] = metrics_column
parameters["Model"] = models
stderr["Model"] = models

print(results.head())
print("Dataframe Done")

##ALL MODELS WILL UNDERGO BAYESIAN OPTIMIZATION, OR SOME VARIANT OF BAYESIAN OPTIMIZATION. THE GENERAL TUNING PROCESSES ARE IDENTICAL AND DOCUMENTED FOR THE FIRST FEW MODELS 

#Note: Selection of bayes_opt library vs BayesSearchCV was based on observed performance with both tuning methods (best performing selected). Hyperparameter bounds were either selected arbitrarily or adjusted based on observed performance. 

#Tuning Decision Trees 
# Loop through all 5 angstrom level sets, starting from 5
index = 5
for data in data_list:
    column = f'{index} angstroms'
    index -= 1
    #_Flatten neighbors with types into feature vectors
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    #_Encode atom types using one-hot encoding
    encoder = OneHotEncoder(sparse=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))  # Encode atom types
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)
    # Add features to input data
    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    #_Target output
    y = data['chemical_shift']
    # 80-20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # Define objective function inside or outside of training loop
    def train_decision_tree(max_depth, min_samples_split, min_samples_leaf):
        if (max_depth < 2):
            max_depth = None
        else:
            max_depth=int(max_depth)
        dt = DecisionTreeRegressor(max_depth=max_depth,
                                min_samples_split=int(min_samples_split),
                                min_samples_leaf=int(min_samples_leaf),
                                random_state=42)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return -mae  # Return negative MAE for maximization (Bayesian optimization maximizes the objective function)

    # Define hyperparameter bounds (most commonly tuned hyperparameters chosen; bounds were manually adjusted based on performance)
    pbounds = {'max_depth': (1, 20),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10)}

    # Initialize Bayesian optimization
    optimizer = BayesianOptimization(f=train_decision_tree, pbounds=pbounds, random_state=42)

    optimizer.maximize(init_points=10, n_iter=10)
    best_params = optimizer.max['params']
    
    # Train Decision Tree with best hyperparameters
    start_time = time.time()
    
    dt = DecisionTreeRegressor(max_depth=int(best_params['max_depth']),
                               min_samples_split=int(best_params['max_depth']),
                               min_samples_leaf=int(best_params['min_samples_leaf']),
                               random_state=42)
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    # Predict on testing data
    y_pred = dt.predict(X_test)
    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Calculate the standard error (of the mean) of the absolute errors 
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "DT"), column] = stderror
    parameters.loc[(parameters["Model"] == "DT"), column] = str(best_params)
    # Update results dataframe
    results.loc[(results["Model"] == "DT") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "DT") & (results["Metric"] == "MAE"), column] = mae  # Convert back to positive value
    results.loc[(results["Model"] == "DT") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/DT_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(dt, file)

print(results)
print("1) DT done")

# Purpose of uploading to csv early: tuning process generally takes a long time (especially for BayesSearchCV), so in case the code stops in the middle, the results are saved and reread using pandas  
results.to_csv("results_temp_2.csv", index=False) 
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)
stderr = pd.read_csv("stderr_temp_2.csv")
parameters = pd.read_csv("parameters_temp_2.csv")
results = pd.read_csv("results_temp_2.csv")

#Tuning SVR 
# Define search space for Bayesian Optimization
svr_param_space = {
    'C': Real(1e-6, 1e+6, prior='log-uniform'),
    'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
    'epsilon': Real(1e-6, 1e+1, prior='log-uniform')
}

# Inside SVR training loop
index = 5
for data in data_list:
    column = f'{index} angstroms'
    index -= 1
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    encoder = OneHotEncoder(sparse=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))  # Encode atom types
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)
    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    y = data['chemical_shift']
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    print("Fitting SVR...")
    svr = SVR(kernel='rbf')
    svr_bayes_search = BayesSearchCV(
        svr,
        svr_param_space,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_iter=50,
        random_state=42
    )
    svr_bayes_search.fit(X_train, y_train)
    # Get best SVR model and fit on the training data 
    start_time = time.time()
    best_svr = svr_bayes_search.best_estimator_
    best_svr.fit(X_train, y_train)
    # + Predict on testing data
    y_pred = best_svr.predict(X_test)
    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # Update results and parameters dataframes
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "SVR"), column] = stderror
    parameters.loc[(parameters["Model"] == "SVR"), column] = str(svr_bayes_search.best_params_) 
    results.loc[(results["Model"] == "SVR") & (results["Metric"] == "MAE"), column] = mae
    results.loc[(results["Model"] == "SVR") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "SVR") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/SVR_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(best_svr, file)
print("2) SVR done")
results.to_csv("results_temp_2.csv", index=False)
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)

stderr = pd.read_csv("stderr_temp_2.csv")
parameters = pd.read_csv("parameters_temp_2.csv")
results = pd.read_csv("results_temp_2.csv")

# Tuning Random Forest
rf_param_space = {
    'n_estimators': Integer(10, 300), 
    'max_depth': Integer(3, 20), 
    'min_samples_split': Integer(2, 20), 
    'min_samples_leaf': Integer(1, 20),
    'max_features': Categorical(['log2', 'sqrt', None]),  
    'bootstrap': Categorical([True, False])
}
rf_bayes_search = BayesSearchCV(
    RandomForestRegressor(),
    rf_param_space,
    scoring='neg_mean_absolute_error',
    cv=5,
    n_iter=50,
    random_state=42
)
index = 5
for data in data_list:
    column = f'{index} angstroms'
    index -= 1
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    encoder = OneHotEncoder(sparse=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))  # Encode atom types
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)
    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    y = data['chemical_shift']
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    print("Fitting RF...")
    rf_bayes_search.fit(X_train, y_train)
    start_time = time.time()
    best_rf = rf_bayes_search.best_estimator_
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)
    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "RF"), column] = stderror
    parameters.loc[(parameters["Model"] == "RF"), column] = str(rf_bayes_search.best_params_)
    results.loc[(results["Model"] == "RF") & (results["Metric"] == "MAE"), column] = mae
    results.loc[(results["Model"] == "RF") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "RF") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/RF_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(best_rf, file)
print("3) RF done")

results.to_csv("results_temp_2.csv", index=False)
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)
stderr = pd.read_csv("stderr_temp_2.csv")
parameters = pd.read_csv("parameters_temp_2.csv")
results = pd.read_csv("results_temp_2.csv")


# Tuning MLP
mlp_param_space = {
    'hidden_layer_sizes': Integer(32, 512),
    'learning_rate_init': Real(0.001, 0.05, prior='log-uniform'),
    'max_iter': Integer(3000, 5000)
}
index = 5
for data in data_list:
    column = f'{index} angstroms'
    index -= 1
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    encoder = OneHotEncoder(sparse=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))  # Encode atom types
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)
    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    y = data['chemical_shift']
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    mlp = MLPRegressor(early_stopping=True, validation_fraction=0.2)
    mlp_bayes_search = BayesSearchCV(
        mlp,
        mlp_param_space,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_iter=50, 
        random_state=42
    )
    mlp_bayes_search.fit(X_train, y_train)
    start_time = time.time()
    best_mlp = mlp_bayes_search.best_estimator_
    best_mlp.fit(X_train, y_train)
    y_pred = best_mlp.predict(X_test)
    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "MLP"), column] = stderror
    parameters.loc[(parameters["Model"] == "MLP"), column] = str(mlp_bayes_search.best_params_)
    results.loc[(results["Model"] == "MLP") & (results["Metric"] == "MAE"), column] = mae
    results.loc[(results["Model"] == "MLP") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "MLP") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/MLP_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(best_mlp, file)
print("4) MLP done ")
results.to_csv("results_temp_2.csv", index=False)
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)

stderr = pd.read_csv("stderr_temp_2.csv")
parameters = pd.read_csv("parameters_temp_2.csv")
results = pd.read_csv("results_temp_2.csv")


# Tuning Gradient Boosting Regression with Decision Trees (high-performing model!)
gbdt_param_space = {
    'learning_rate': (0.001, 1.0, 'log-uniform'),  # Learning rate for gradient boosting
    'n_estimators': (10, 300),                     # Number of boosting stages
    'max_depth': (3, 20),                          # Maximum depth of the individual regression estimators
    'min_samples_split': (2, 20),                  # Minimum number of samples required to split an internal node
    'min_samples_leaf': (1, 20),                   # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2', None],  # Number of features to consider when looking for the best split
    'subsample': (0.1, 1.0, 'uniform')             # Fraction of samples used for fitting the individual base learners
}
gbdt_bayes_search = BayesSearchCV(
    GradientBoostingRegressor(),
    gbdt_param_space,
    scoring='neg_mean_absolute_error',
    cv=5,
    n_iter=50, 
    random_state=42
)
index = 2
for data in data_list:
    data = data_2
    column = f'{index} angstroms'
    index -= 1
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    encoder = OneHotEncoder(sparse=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))  # Encode atom types
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)
    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    y = data['chemical_shift']
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    print("Fitting GBR...")
    gbdt_bayes_search.fit(X_train, y_train)
    best_gbdt = gbdt_bayes_search.best_estimator_
    #best_gbdt = GradientBoostingRegressor(**OrderedDict([('learning_rate', 0.083759380187059), ('max_depth', 20), ('max_features', 'sqrt'), ('min_samples_leaf', 1), ('min_samples_split', 19), ('n_estimators', 300), ('subsample', 0.8480243834493593)]))
    start_time = time.time()
    best_gbdt.fit(X_train, y_train)
    y_pred = best_gbdt.predict(X_test)
    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "GBR"), column] = stderror
    parameters.loc[(parameters["Model"] == "GBR"), column] = str(gbdt_bayes_search.best_params_)
    results.loc[(results["Model"] == "GBR") & (results["Metric"] == "MAE"), column] = mae
    results.loc[(results["Model"] == "GBR") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "GBR") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/GBR_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(best_gbdt, file)
    input("Continue?")
print("5) GBR done")
results.to_csv("results_temp_2.csv", index=False)
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)
stderr = pd.read_csv("stderr_temp_2.csv")
parameters = pd.read_csv("parameters_temp_2.csv")
results = pd.read_csv("results_temp_2.csv")

#Tuning KNN 
#Back to defining the objective function
def knn_objective(n_neighbors, weights, p, X_train, X_test, y_train, y_test):
    n_neighbors = int(n_neighbors)
    weights = 'uniform' if weights < 0.5 else 'distance'
    p = int(p)
    
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return -mae  
index = 5
for data in data_list:
    column = f'{index} angstroms'
    index -= 1
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    encoder = OneHotEncoder(sparse=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))  # Encode atom types
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)
    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    y = data['chemical_shift']
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    pbounds = {'n_neighbors': (1, 30), 'weights': (0, 1), 'p': (1, 2)}
    optimizer = BayesianOptimization(
        f=lambda n_neighbors, weights, p: knn_objective(n_neighbors, weights, p, X_train, X_test, y_train, y_test),
        pbounds=pbounds,
        random_state=42,
    )
    optimizer.maximize(init_points=5, n_iter=10)
    best_params = optimizer.max['params']
    n_neighbors = int(best_params['n_neighbors'])
    weights = 'uniform' if best_params['weights'] < 0.5 else 'distance'
    p = int(best_params['p'])
    print("Best Hyperparameters:", best_params)
    start_time = time.time()
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, p=p)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "KNN"), column] = stderror
    parameters.loc[(parameters["Model"] == "KNN"), column] = str(best_params)
    results.loc[(results["Model"] == "KNN") & (results["Metric"] == "MAE"), column] = mae
    results.loc[(results["Model"] == "KNN") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "KNN") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/KNN_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(knn, file)

print("6) KNN done")
results.to_csv("results_temp_2.csv", index=False)
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)
stderr = pd.read_csv("stderr_temp_2.csv")
parameters = pd.read_csv("parameters_temp_2.csv")
results = pd.read_csv("results_temp_2.csv")

# Tuning GPR 
def gpr_evaluate(alpha, X_train, X_test, y_train, y_test):
    kernel = DotProduct()
    kernel += WhiteKernel(alpha)
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

index = 5
for data in data_list:
    column = f'{index} angstroms'
    index -= 1
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    encoder = OneHotEncoder(sparse=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)

    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    y = data['chemical_shift']

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    pbounds = {'alpha': (1e-10, 1e-1)}
    optimizer = BayesianOptimization(
        f=lambda alpha: gpr_evaluate(alpha, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test),
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(init_points=5, n_iter=25)
    best_params = optimizer.max['params']
    alpha = best_params['alpha']
    print("Best Hyperparameters:", best_params)
    start_time = time.time()
    kernel = DotProduct() + WhiteKernel(alpha)
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "GPR"), column] = stderror
    parameters.loc[(parameters["Model"] == "GPR"), column] = str(best_params)
    results.loc[(results["Model"] == "GPR") & (results["Metric"] == "MAE"), column] = mae
    results.loc[(results["Model"] == "GPR") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "GPR") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/GPR_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(gpr, file)
print("7) GPR done")
results.to_csv("results_temp_2.csv", index=False)
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)
stderr = pd.read_csv("stderr_temp_2.csv")
parameters = pd.read_csv("parameters_temp_2.csv")
results = pd.read_csv("results_temp_2.csv")

# Tuning Bagging Regressor
def bagging_regressor_evaluate(n_estimators, max_samples, max_features, X_train, y_train, X_test, y_test):
    n_estimators = int(n_estimators)
    br = BaggingRegressor(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, n_jobs=-1)
    br.fit(X_train, y_train)
    y_pred = br.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2


index = 5
for data in data_list:
    column = f'{index} angstroms'
    index -= 1
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    encoder = OneHotEncoder(sparse=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)

    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    y = data['chemical_shift']

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    pbounds = {'n_estimators': (10, 100),
               'max_samples': (0.1, 1.0),
               'max_features': (0.1, 1.0)}
    optimizer = BayesianOptimization(
        f=lambda n_estimators, max_samples, max_features: bagging_regressor_evaluate(n_estimators, max_samples, max_features, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test),
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(init_points=10, n_iter=50)
    best_params = optimizer.max['params']
    n_estimators = int(best_params['n_estimators'])
    max_samples = best_params['max_samples']
    max_features = best_params['max_features']

    print("Best Hyperparameters:", best_params)
    start_time = time.time()
    br = BaggingRegressor(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, n_jobs=-1)
    br.fit(X_train, y_train)
    y_pred = br.predict(X_test)

    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "BR"), column] = stderror
    parameters.loc[(parameters["Model"] == "BR"), column] = str(best_params)
    results.loc[(results["Model"] == "BR") & (results["Metric"] == "MAE"), column] = mae
    results.loc[(results["Model"] == "BR") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "BR") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/BR_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(br, file)
print("8) BR done")
results.to_csv("results_temp_2.csv", index=False)
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)
stderr = pd.read_csv("stderr_temp_2.csv")
parameters = pd.read_csv("parameters_temp_2.csv")
results = pd.read_csv("results_temp_2.csv")

# Tuning Adaboost with BO-TPE instead of standard BO
def adaboost_evaluate(params, X_train, y_train, X_test, y_test):
    params['n_estimators'] = int(round(params['n_estimators']))
    adb = AdaBoostRegressor(**params)
    adb.fit(X_train, y_train)
    y_pred = adb.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return -r2  # Hyperopt minimizes the function

index = 5
for data in data_list:
    column = f'{index} angstroms'
    index -= 1
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    encoder = OneHotEncoder(sparse=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)

    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    y = data['chemical_shift']

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 500, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 1.0)
    }

    trials = Trials()
    best = fmin(fn=lambda params: adaboost_evaluate(params, X_train, y_train, X_test, y_test),
                space=space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials)

    best_params = {key: int(round(value)) if key == 'n_estimators' else value for key, value in best.items()}
    print("Best Hyperparameters:", best_params)
    adb = AdaBoostRegressor(**best_params)
    start_time = time.time()
    adb.fit(X_train, y_train)
    y_pred = adb.predict(X_test)

    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    parameters.loc[(parameters["Model"] == "ADB"), column] = str(best_params)
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "ADB"), column] = stderror
    results.loc[(results["Model"] == "ADB") & (results["Metric"] == "MAE"), column] = mae
    results.loc[(results["Model"] == "ADB") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "ADB") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/ADB_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(adb, file)
print("9) ADB done")
results.to_csv("results_temp_2.csv", index=False)
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)
stderr = pd.read_csv("stderr_temp_2.csv")
parameters = pd.read_csv("parameters_temp_2.csv")
results = pd.read_csv("results_temp_2.csv")

# Tuning Extra Trees
def extratrees_evaluate(max_depth, min_samples_split, min_samples_leaf, max_features, X_train, y_train, X_test, y_test):
    model = ExtraTreeRegressor(
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=min(max_features, 0.999), 
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return -mae


index = 5
for data in data_list:
    column = f'{index} angstroms'
    index -= 1
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    encoder = OneHotEncoder(sparse_output=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)

    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    y = data['chemical_shift']

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    pbounds = {
        'max_depth': (3, 20), 
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'max_features': (0.1, 0.999)
    }
    optimizer = BayesianOptimization(
        f=lambda max_depth, min_samples_split, min_samples_leaf, max_features: extratrees_evaluate(max_depth, min_samples_split, min_samples_leaf, max_features, X_train, y_train, X_test, y_test),
        pbounds=pbounds,
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=25)
    best_params = optimizer.max['params']
    best_max_depth = int(best_params['max_depth'])
    best_min_samples_split = int(best_params['min_samples_split'])
    best_min_samples_leaf = int(best_params['min_samples_leaf'])
    best_max_features = min(best_params['max_features'], 0.999)
    print("Best Hyperparameters:", best_params)
    start_time = time.time()
    model = ExtraTreeRegressor(
        max_depth=best_max_depth,
        min_samples_split=best_min_samples_split,
        min_samples_leaf=best_min_samples_leaf,
        max_features=best_max_features,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    parameters.loc[(parameters["Model"] == "ET"), column] = str(best_params)
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "ET"), column] = stderror
    results.loc[(results["Model"] == "ET") & (results["Metric"] == "MAE"), column] = mae
    results.loc[(results["Model"] == "ET") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "ET") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/ET_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(model, file)
print("10) ET done")
results.to_csv("results_temp_2.csv", index=False)
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)
stderr = pd.read_csv("stderr_temp_2.csv")
parameters = pd.read_csv("parameters_temp_2.csv")
results = pd.read_csv("results_temp_2.csv")


# Tuning Elastic Net 
def elasticnet_evaluate(alpha, X_train, y_train, X_test, y_test):
    en = ElasticNet(alpha=alpha)
    en.fit(X_train, y_train)
    y_pred = en.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2 

index = 5
for data in data_list:
    column = f'{index} angstroms'
    index -= 1
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    encoder = OneHotEncoder(sparse_output=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)

    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    y = data['chemical_shift']

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    pbounds = {
        'alpha': (1e-3, 1e3)
    }
    optimizer = BayesianOptimization(
        f=lambda alpha: elasticnet_evaluate(alpha, X_train, y_train, X_test, y_test),
        pbounds=pbounds,
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=25)
    best_params = optimizer.max['params']
    best_alpha = best_params['alpha']
    print("Best Hyperparameters:", best_params)
    start_time = time.time()
    en = ElasticNet(alpha=best_alpha)
    en.fit(X_train, y_train)
    y_pred = en.predict(X_test)

    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    parameters.loc[(parameters["Model"] == "EN"), column] = str(best_params)
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "EN"), column] = stderror
    results.loc[(results["Model"] == "EN") & (results["Metric"] == "MAE"), column] = mae
    results.loc[(results["Model"] == "EN") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "EN") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/EN_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(en, file)
print("11) EN done")
results.to_csv("results_temp_2.csv", index=False)
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)
stderr = pd.read_csv("stderr_temp_2.csv")
parameters = pd.read_csv("parameters_temp_2.csv")
results = pd.read_csv("results_temp_2.csv")

# Tuning Lasso Regression 
def lasso_evaluate(alpha, X_train, y_train, X_test, y_test):
    lso = Lasso(alpha=alpha)
    lso.fit(X_train, y_train)
    y_pred = lso.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2 

index = 5
for data in data_list:
    column = f'{index} angstroms'
    index -= 1
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    encoder = OneHotEncoder(sparse_output=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)

    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    y = data['chemical_shift']

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    pbounds = {
        'alpha': (1e-3, 1e3)
    }

    optimizer = BayesianOptimization(
        f=lambda alpha: lasso_evaluate(alpha, X_train, y_train, X_test, y_test),
        pbounds=pbounds,
        random_state=42
    )

    optimizer.maximize(init_points=5, n_iter=25)
    best_params = optimizer.max['params']
    best_alpha = best_params['alpha']
    print("Best Hyperparameters:", best_params)
    start_time = time.time()
    lso = Lasso(alpha=best_alpha)
    lso.fit(X_train, y_train)
    y_pred = lso.predict(X_test)

    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "LSO"), column] = stderror
    parameters.loc[(parameters["Model"] == "LSO"), column] = str(best_params)

    results.loc[(results["Model"] == "LSO") & (results["Metric"] == "MAE"), column] = mae
    results.loc[(results["Model"] == "LSO") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "LSO") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/LSO_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(lso, file)
print("12) LSO done")
results.to_csv("results_temp_2.csv", index=False)
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)
stderr = pd.read_csv("stderr_temp_2.csv")
parameters = pd.read_csv("parameters_temp_2.csv")
results = pd.read_csv("results_temp_2.csv")

# Tuning Ridge Regression 
def ridge_evaluate(alpha, fit_intercept, solver, X_train, y_train, X_test, y_test):
    # Convert categorical hyperparameters from float to appropriate type
    fit_intercept = bool(round(fit_intercept))
    solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'][int(solver)]

    rr = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver)
    rr.fit(X_train, y_train)
    y_pred = rr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return r2

index = 5
for data in data_list:
    column = f'{index} angstroms'
    index -= 1
    distances, neighbors, charges, nums, en, mass = flatten_neighbors_with_types(data['neighbors'].apply(ast.literal_eval))
    encoder = OneHotEncoder(sparse_output=False)
    atom_types_encoded = encoder.fit_transform(np.array(neighbors))
    distances_encoded = np.array(distances)
    charges = np.array(charges)
    nums = np.array(nums)
    en = np.array(en)
    mass = np.array(mass)

    X_encoded = np.hstack([distances_encoded, atom_types_encoded, charges, nums, en, mass])
    y = data['chemical_shift']

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    # Normalize the features for Ridge
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    pbounds = {
        'alpha': (1e-3, 1e3),
        'fit_intercept': (0, 1),  # 0 or 1 for boolean
        'solver': (0, 6)  # indices of the solver list
    }
    optimizer = BayesianOptimization(
        f=lambda alpha, fit_intercept, solver: ridge_evaluate(alpha, fit_intercept, solver, X_train, y_train, X_test, y_test),
        pbounds=pbounds,
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=25)
    best_params = optimizer.max['params']
    best_alpha = best_params['alpha']
    best_fit_intercept = bool(round(best_params['fit_intercept']))
    best_solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'][int(best_params['solver'])]
    print("Best Hyperparameters:", best_params)
    start_time = time.time()
    rr = Ridge(alpha=best_alpha, fit_intercept=best_fit_intercept, solver=best_solver)
    rr.fit(X_train, y_train)
    y_pred = rr.predict(X_test)

    end_time = time.time()
    runtime = end_time - start_time
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    parameters.loc[(parameters["Model"] == "RDG"), column] = str(best_params)
    stderror = y_test.sub(y_pred).abs().sem()
    stderr.loc[(stderr["Model"] == "RDG"), column] = stderror
    results.loc[(results["Model"] == "RDG") & (results["Metric"] == "MAE"), column] = mae
    results.loc[(results["Model"] == "RDG") & (results["Metric"] == "R^2"), column] = r2
    results.loc[(results["Model"] == "RDG") & (results["Metric"] == "Runtime"), column] = runtime
    pickle_file = f"model_files/RDG_{index+1}.pkl"
    with open(pickle_file, 'wb') as file:
        pickle.dump(rr, file)
print("13) RDG done")
results.to_csv("results_temp_2.csv", index=False)
print(parameters)
parameters.to_csv("parameters_temp_2.csv", index=False)
stderr.to_csv("stderr_temp_2.csv", index=False)
