from decimal import Decimal
import matplotlib.pyplot as plt
import os
import geomstats.backend as gs
from scipy.integrate import simpson
from nsimplices import *
from common import *
from scipy.stats import wasserstein_distance
import scipy.stats as stats

gs.random.seed(2021)


base_path = "/home/wanxinli/dyn/dyn/"
data_path = os.path.join(base_path, "datasets")

dataset_name = 'breast_cancer'
figs_dir = os.path.join("/home/wanxinli/dyn/dyn/saved_figs", dataset_name)
print(f"Will save figs to {figs_dir}")


def load_breast_cancer_cells():
    """Load dataset of mutated retinal cells.

    The cells are grouped by mutation in the dataset :
    - the *control* cells are ARPE19,
    - the cells treated with Akt mutation,
    - and the ones treated with Mek mutation
    - and the cells treated with the Ras mutation.

    Additionally, in each of these classes, the cells are cultured on two surfaces :
    - the *GDA* cells (simple glass)
    - the *FN* ones (Fibronectin coated glass).
    
    Returns
    -------
    cells : array of n_cells planar discrete curves
        Each curve represents the boundary of a cell in counterclockwise order.
        Their barycenters are fixed at 0 (translation has been removed).
        Their lengths are not necessarily equal (scaling has not been removed).
    lines : list of n_cells strings
        List of the cell lines 

    """

    cell_path = os.path.join(data_path, "breast_cancer", "cells.txt")
    lines_path = os.path.join(data_path, "breast_cancer", "lines.txt")

    with open(cell_path) as cells_file:
        cells = cells_file.read().split("\n\n")
    for i, cell in enumerate(cells):
        cell = cell.split("\n")
        curve = []
        for point in cell:
            coords = [int(coord) for coord in point.split()]
            curve.append(coords)
        cells[i] = gs.cast(gs.array(curve), gs.float32)
    with open(lines_path) as lines_file:
        lines = lines_file.read().split("\n")
    
    # remove the last blank cell
    cells = cells[:-1]
    lines = lines[:-1]
    return cells, lines


cells, lines = load_breast_cancer_cells()
print(f"Total number of cells : {len(cells)}")



import pandas as pd

LINES = gs.unique(lines)
print(LINES)
METRICS = ['SRV', 'Linear']


cell_idx = 1
plt.plot(cells[cell_idx][:, 0], cells[cell_idx][:, 1], "blue")
plt.plot(cells[cell_idx][0, 0], cells[cell_idx][0, 1], "blue", marker="o");


ds = {}

n_cells_arr = gs.zeros(3)


for j, line in enumerate(LINES):
    to_keep = gs.array(
        [
            one_line == line
            for one_line in lines
        ]
    )
    ds[line] = [
        cell_i for cell_i, to_keep_i in zip(cells, to_keep) if to_keep_i
    ]
    nb = len(ds[line])
    print(f"{line}: {nb}")
    n_cells_arr[j] = nb


print({'MCF10A': n_cells_arr[0], 'MCF7': n_cells_arr[1], 'MDA_MB_231': n_cells_arr[2]})
n_cells_df = pd.DataFrame({'MCF10A': [n_cells_arr[0]], 'MCF7': [n_cells_arr[1]], 'MDA_MB_231': [n_cells_arr[2]]})


def apply_func_to_ds(input_ds, func):
    """Apply the input function func to the input dictionnary input_ds.

    This function goes through the dictionnary structure and applies
    func to every cell in input_ds[treatment][line].

    It stores the result in a dictionnary output_ds that is returned
    to the user.

    Parameters
    ----------
    input_ds : dict
        Input dictionnary, with keys treatment-line.
    func : callable
        Function to be applied to the values of the dictionnary, i.e.
        the cells.

    Returns
    -------
    output_ds : dict
        Output dictionnary, with the same keys as input_ds.
    """
    output_ds = {}
    for line in LINES:
        output_list = []
        for one_cell in input_ds[line]:
            output_list.append(func(one_cell))
        output_ds[line] = gs.array(output_list)
    return output_ds


def interpolate(curve, nb_points):
    """Interpolate a discrete curve with nb_points from a discrete curve.

    Returns
    -------
    interpolation : discrete curve with nb_points points
    """
    old_length = curve.shape[0]
    interpolation = gs.zeros((nb_points, 2))
    incr = old_length / nb_points
    pos = 0
    for i in range(nb_points):
        index = int(gs.floor(pos))
        interpolation[i] = curve[index] + (pos - index) * (
            curve[(index + 1) % old_length] - curve[index]
        )
        pos += incr
    return interpolation

k_sampling_points = 200

ds_interp = apply_func_to_ds(
    input_ds=ds, func=lambda x: interpolate(x, k_sampling_points)
)


import numpy as np

def preprocess(curve, tol=1e-10):
    """Preprocess curve to ensure that there are no consecutive duplicate points.

    Returns
    -------
    curve : discrete curve
    """

    dist = curve[1:] - curve[:-1]
    dist_norm = np.sqrt(np.sum(np.square(dist), axis=1))

    if np.any( dist_norm < tol ):
        for i in range(len(curve)-1):
            if np.sqrt(np.sum(np.square(curve[i+1] - curve[i]), axis=0)) < tol:
                curve[i+1] = (curve[i] + curve[i+2]) / 2

    return curve


data_folder = os.path.join(data_path, dataset_name, "aligned")

ds_proc = apply_func_to_ds(ds_interp, func=lambda x: preprocess(x))

BASE_CURVE = ds_proc["MCF10A"][0]


for line in LINES:
    cells = ds_proc[line]
    for i, cell in enumerate(cells):
        try:
            print("try exhaustive align with reparamterization")
            aligned_cell = exhaustive_align(cell, BASE_CURVE, k_sampling_points, dynamic=False, rotation_only=False)
            file_path = os.path.join(data_folder, f"{line}_{i}.txt")
            np.savetxt(file_path, aligned_cell)
        except Exception:
            print("exception")
            # file_path = os.path.join(data_folder, f"{line}_{i}_rotation_only.txt")
            # aligned_cell = exhaustive_align(cell, BASE_CURVE, k_sampling_points, rotation_only=True)
            # np.savetxt(file_path, aligned_cell)
            pass

