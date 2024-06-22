# Imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

# Set default figure to 11 in X 8 in
rcParams['figure.figsize'] = (11, 8)
# Creates lists for best and worst models
best_models = ["GBR","BR","RF","KNN"]
worst_models = ["GPR","MLP"]

# Creates a figure with subplots.
def graph_predictions(models, name, nrows, ncols):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10), tight_layout=True)
    images = list()
    # Iterates through all models and appends their corresponding images to images list
    for model in models:
        img = mpimg.imread(f'model_predictions/{model}_predictions.png')
        images.append(img)
    # Images are plotted looping through each cell in subplot grid
    index = 0
    for row in range(nrows):
        for col in range(ncols):
            img = images[index]
            if (nrows == 1):
                ax[col].imshow(img)
                ax[col].axis("off")
                ax[col].set_title(f"{chr(index + 97)})")
            else:
                ax[row, col].imshow(img)
                ax[row, col].axis("off")
                ax[row, col].set_title(f"{chr(index + 97)})")
            index += 1
    # Figure and saved aand displayed
    plt.savefig(f'{name}.png', bbox_inches='tight')
    plt.show()
# Calls graph_predictions on best and worst models individually
graph_predictions(best_models, "Best_Models", 2, 2)
graph_predictions(worst_models, "Worst_Models", 1, 2)
