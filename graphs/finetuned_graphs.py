import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

# PLOT DOUBLE BAR GRAPHS 
# Manual models list
models = ['MLP', 'SVR', 'GBR', 'GPR', 'RF', 'DT', 'BR', 'ADB', 'ET', 'KNN', 'EN', 'LSO', 'RDG']
mae_values = list()
r2_values = list()
angstrom_level = "3 angstroms"
sem_values = list()

#Process data _ put into the lists
results_df = pd.read_csv("results_temp_2.csv")
stderrors = pd.read_csv("stderr_temp_2.csv")
for model in models:
    error = round(results_df.loc[(results_df["Model"] == model) & (results_df["Metric"] == "MAE"), angstrom_level].values[0], 2)
    rsq = round(results_df.loc[(results_df["Model"] == model) & (results_df["Metric"] == "R^2"), angstrom_level].values[0], 2)
    mae_values.append(error)
    stderr = round(stderrors.loc[(stderrors["Model"] == model), angstrom_level].values[0], 2)
    r2_values.append(rsq)
    sem_values.append(stderr)
print(mae_values)
print(r2_values)

fig, ax1 = plt.subplots(figsize=(15,8))
bar_width = 0.35

# Create bar chart
bars1 = ax1.bar(np.arange(len(models))  - bar_width/2, mae_values, bar_width, label='MAE', color='blue')

# Error bars (1x standard error)
plt.errorbar(np.arange(len(models))  - bar_width/2, mae_values, yerr=sem_values, color="black", fmt='.',capsize=3, capthick=0.5)

# Add secondary y-axis for R^2
ax2 = ax1.twinx()

bars2 = ax2.bar(np.arange(len(models)) + bar_width/2, r2_values, bar_width, label='R$^2$ ', color='orange')

# Add title and labels
ax1.set_title(f'Mean Absolute Error and R$^2$ Comparison at {angstrom_level}', pad=20)
ax1.set_xlabel('Models')
ax1.set_ylabel('Mean Absolute Error')
ax2.set_ylabel('R$^2$')

ax1.set_xticks(np.arange(len(models)))
ax1.set_xticklabels(models)

# Need proper arrangement to avoid wonky overlapping bars/labels
for i in range(len(models)):
    ax1.text(np.arange(len(models))[i] - bar_width/2 - 0.01, mae_values[i] + max(mae_values)*0.05, f"MAE:\n{mae_values[i]}", ha='center', fontsize=8)
    ax2.text(np.arange(len(models))[i] + bar_width/2, r2_values[i] + max(r2_values)*0.05, f"R$^2$:\n{r2_values[i]}", ha='center', fontsize=8)

plt.grid(True)
plt.tight_layout()
plt.savefig("double_bars_sem.png")
plt.show()
