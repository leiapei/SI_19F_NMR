import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# PLOT 4 ANGSTROM COMPARISON HEATMAPS 
# Data from results_temp (final results)
data = pd.read_csv('results_temp_2.csv')

# Create separate dataframes for each metric
mae_df = data[data['Metric'] == 'MAE'].pivot(index='Model', columns='Metric', values=['5 angstroms', '4 angstroms', '3 angstroms', '2 angstroms'])
r2_df = data[data['Metric'] == 'R^2'].pivot(index='Model', columns='Metric', values=['5 angstroms', '4 angstroms', '3 angstroms', '2 angstroms'])
runtime_df = data[data['Metric'] == 'Runtime'].pivot(index='Model', columns='Metric', values=['5 angstroms', '4 angstroms', '3 angstroms', '2 angstroms'])

# Flatten the column MultiIndex
mae_df.columns = mae_df.columns.get_level_values(0)
r2_df.columns = r2_df.columns.get_level_values(0)
runtime_df.columns = runtime_df.columns.get_level_values(0)

# Min-Max scaling to normalize runtimes for each model (row-wise normalization; normalize within each model so it's a fairer comparison)
runtime_df_normalized = runtime_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# Plot the MAE heatmap
sns.heatmap(mae_df, annot=True, fmt=".2f", cmap="YlGnBu", ax=axs[0, 0])
axs[0, 0].set_title('MAE Heatmap')
axs[0, 0].set_xlabel('Angstrom Levels')
axs[0, 0].set_ylabel('Models')

# Plot the R^2 heatmap
sns.heatmap(r2_df, annot=True, fmt=".4f", cmap="YlGnBu", ax=axs[0, 1])
axs[0, 1].set_title('R^2 Heatmap')
axs[0, 1].set_xlabel('Angstrom Levels')
axs[0, 1].set_ylabel('Models')

# Plot the Runtime heatmap
sns.heatmap(runtime_df, annot=True, fmt=".3f", cmap="YlGnBu", ax=axs[1, 0])
axs[1, 0].set_title('Runtime Heatmap')
axs[1, 0].set_xlabel('Angstrom Levels')
axs[1, 0].set_ylabel('Models')

# Plot the normalized Runtime heatmap
sns.heatmap(runtime_df_normalized, annot=True, fmt=".4f", cmap="YlGnBu", ax=axs[1, 1])
axs[1, 1].set_title('Normalized Runtime Heatmap')
axs[1, 1].set_xlabel('Angstrom Levels')
axs[1, 1].set_ylabel('Models')

# Adjust layout to make room for titles and avoid overlap
plt.tight_layout()
plt.savefig("combined_heatmaps.png")
plt.show()