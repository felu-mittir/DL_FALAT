import numpy as np
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Constants
DATA_PATH = "logs/results_"
NUM_RESULTS = 20
RAND_ACC = 0.5
DATA_CHUNK = 500
RECTANGLE_HEIGHT = 4.5


def load_data():
    """
    Load experimental results from numpy files and compute t-values.

    Returns:
        A list of computed t-values (absolute values).
    """
    t_values = []

    for data_index in range(NUM_RESULTS):
        try:
            acc = np.load(f"{DATA_PATH}{data_index}.npy")
            val, _ = ttest_1samp(acc, RAND_ACC)
            t_values.append(abs(val[0]))
        except FileNotFoundError:
            print(f"File '{DATA_PATH}{data_index}.npy' not found.")
            continue

    return t_values


def plot_values(t_values):
    """
    Generate a plot for the given t-values and save it as a PDF file.
    """
    fig, ax = plt.subplots(1, figsize=(7, 5))

    ax.plot(np.arange(1, len(t_values) + 1) * DATA_CHUNK,
            t_values, color='blue', linewidth=2, label="DL-FALAT")

    rect = Rectangle((DATA_CHUNK, 0),
                     width=NUM_RESULTS * DATA_CHUNK,
                     height=RECTANGLE_HEIGHT,
                     linewidth=1,
                     facecolor='pink',
                     label="Non-Leakage Zone")

    ax.add_patch(rect)
    ax.set_xlabel("#Ciphertexts", fontsize=25)
    ax.set_ylabel(r"$t$-Value (abs)", fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=20)
    max_t_value = max(t_values)
    ax.set_ylim(0, max_t_value + 5)
    ax.set_xlim(DATA_CHUNK, DATA_CHUNK * NUM_RESULTS)
    leg = plt.legend(prop={'size': 12})
    leg.get_frame().set_edgecolor('black')
    plt.savefig("plot.pdf", bbox_inches='tight', dpi=600)
    plt.close()



if __name__ == "__main__":
    t_values = load_data()
    plot_values(t_values)
