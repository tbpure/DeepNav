import numpy as np
from matplotlib import pyplot as plt


def plot_trajectory(np_array):
    plt.figure(figsize=(8, 8), dpi=120)
    plt.plot(np_array[:, 1], np_array[:, 0], label="GPS", linestyle='--', linewidth=1.8, alpha=1)

    plt.xlabel('Y', fontsize=12)
    plt.ylabel('X', fontsize=12)

    plt.title('2D Trajectory Comparison', fontsize=14)
    plt.legend(loc='best', fontsize=11)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


def plot_multidim_lines(np_array):
    plt.figure(figsize=(8, 8), dpi=120)
    n_dims = np_array.shape[1]

    for i in range(n_dims):
        plt.plot(np_array[:, i], linewidth=1.5)

    plt.title("title", fontsize=14)
    # plt.legend(loc='best', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_array_list(np_array_list:list[np.ndarray]):
    if len(np_array_list) == 0:
        raise ValueError("The array is empty")

    for i in range(np_array_list[0].shape[1]):
        plt.figure(figsize=(8, 8), dpi=120)
        for j in range(len(np_array_list)):
            plt.plot(np_array_list[j][:, i], linewidth=1.5)
        plt.title("title", fontsize=14)
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
