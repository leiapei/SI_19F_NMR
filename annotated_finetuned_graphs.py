# Import necessary modules for plotting, data manipulations, and numerical operations
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

# PLOT DOUBLE BAR GRAPHS 
# Create models list for each model
models = ['MLP', 'SVR', 'GBR', 'GPR', 'RF', 'DT', 'BR', 'ADB', 'ET', 'KNN', 'EN', 'LSO', 'RDG']
# Initialize empty MAE, R-squared, and SEM value lists
mae_values = list()
r2_values = list()
# Angstrom level set to 3 angstroms
angstrom_level = "3 angstroms"
sem_values = list()

# Processes data from csv files to DataFrames
results_df = pd.read_csv("results_temp_2.csv")
stderrors = pd.read_csv("stderr_temp_2.csv")
# Loops through each model, extracts MAE, R-squared, and SEM values from DataFrames
# Values rounded to 2 decimal places and appended to respective lists
for model in models:
    error = round(results_df.loc[(results_df["Model"] == model) & (results_df["Metric"] == "MAE"), angstrom_level].values[0], 2)
    rsq = round(results_df.loc[(results_df["Model"] == model) & (results_df["Metric"] == "R^2"), angstrom_level].values[0], 2)
    mae_values.append(error)
    stderr = round(stderrors.loc[(stderrors["Model"] == model), angstrom_level].values[0], 2)
    r2_values.append(rsq)
    sem_values.append(stderr)
print(mae_values)
print(r2_values)

# Creates a figure and an axis for bar chart
fig, ax1 = plt.subplots(figsize=(15,8))
bar_width = 0.35

# Creates first set of bars bar chart
bars1 = ax1.bar(np.arange(len(models))  - bar_width/2, mae_values, bar_width, label='MAE', color='blue')

# Adds error bars (1x standard error) to MAE bars using SEM values
plt.errorbar(np.arange(len(models))  - bar_width/2, mae_values, yerr=sem_values, color="black", fmt='.',capsize=3, capthick=0.5)

# Creates a secondary y-axis for R-squared values
ax2 = ax1.twinx()

# Creates second set of bars for R-squared values
bars2 = ax2.bar(np.arange(len(models)) + bar_width/2, r2_values, bar_width, label='R$^2$ ', color='orange')

# Add title and labels on x-axis and y-axis
ax1.set_title(f'Mean Absolute Error and R$^2$ Comparison at {angstrom_level}', pad=20)
ax1.set_xlabel('Models')
ax1.set_ylabel('Mean Absolute Error')
ax2.set_ylabel('R$^2$')

# Sets each x-axis tick/label to correspond to a model name
ax1.set_xticks(np.arange(len(models)))
ax1.set_xticklabels(models)

# Adds text labels for bars to display MAE and R-squared values
# Sets proper arrangement to avoid  overlapping bars/labels
for i in range(len(models)):
    ax1.text(np.arange(len(models))[i] - bar_width/2 - 0.01, mae_values[i] + max(mae_values)*0.05, f"MAE:\n{mae_values[i]}", ha='center', fontsize=8)
    ax2.text(np.arange(len(models))[i] + bar_width/2, r2_values[i] + max(r2_values)*0.05, f"R$^2$:\n{r2_values[i]}", ha='center', fontsize=8)

# Creates plot grid, spacing, save png file, and displays plot
plt.grid(True)
plt.tight_layout()
plt.savefig("double_bars_sem.png")
plt.show()