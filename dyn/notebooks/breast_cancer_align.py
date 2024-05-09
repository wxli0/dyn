""" 
The script can be run in the following way:
python3.11 breast_cancer_align.py --rescale --rotation --reparameterization (each option enables True for the boolean variable)
"""

import argparse
from decimal import Decimal
import matplotlib.pyplot as plt

import geomstats.backend as gs
import numpy as np
from nsimplices import *
from common import *
import scipy.stats as stats
from geomstats.learning.frechet_mean import FrechetMean
import geomstats.datasets.utils as data_utils
import pandas as pd



# Helper functions for alignment
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


def parse_args():
    """ 
    Parse arguments from command line
    """
    global rescale, rotation, reparameterization
    
    parser = argparse.ArgumentParser(description="Script to handle command line arguments.")
    parser.add_argument('--rescale', action='store_true', help="Set this flag to enable rescaling.")
    parser.add_argument('--rotation', action='store_true', help="Set this flag to enable rotation.")
    parser.add_argument('--reparameterization', action='store_true', help="Set this flag to enable reparameterization.")

    
    args = parser.parse_args()

    if args.rescale:
        rescale = True
    else:
        rescale = False

    if args.rotation:
        rotation = True
    else:
        rotation = False

    if args.reparameterization:
        reparameterization = True
    else:
        reparameterization = False
    
    print(f"rescale is: {rescale}, rotation is: {rotation}, reparameterization is: {reparameterization}")


def get_full_suffix(add_suffix = None):
    """ 
    Get the name of the data folder given rescale, rotation \
        and reparameterization, add_suffix variable
    """
    
    suffix = 'projection'

    if rescale:
        suffix += '_rescale'

    if rotation:
        suffix += '_rotation'

    if reparameterization:
        suffix += '_reparameterization'

    if add_suffix is not None:
        suffix += "_"+add_suffix
    
    return suffix


# Procedure for aligning the cells 

# (1) Set up global variables 
base_path = "/home/wanxinli/dyn/dyn/"
data_path = os.path.join(base_path, "datasets")

dataset_name = 'breast_cancer'
figs_dir = os.path.join("/home/wanxinli/dyn/dyn/saved_figs", dataset_name)
print(f"Will save figs to {figs_dir}")


# (2) Load data 
cells, lines = load_breast_cancer_cells()
print(f"Total number of cells : {len(cells)}")

LINES = gs.unique(lines)
print(LINES)
METRICS = ['SRV', 'Linear']


# (3) Prepare ds_proc
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

k_sampling_points = 2000

ds_interp = apply_func_to_ds(
    input_ds=ds, func=lambda x: interpolate(x, k_sampling_points)
)

ds_proc = apply_func_to_ds(ds_interp, func=lambda x: preprocess(x))


# (4) Parse command line arguments and start alignment for the first round

BASE_CURVE = generate_ellipse(k_sampling_points)
data_folder = os.path.join(data_path, dataset_name, "aligned")
suffix = 'projection'
parse_args()

suffix = get_full_suffix("first_round")

data_folder = os.path.join(data_folder, suffix)
print("data_folder for the first round is:", data_folder)

# If the first round has been done, we can comment the code below to jump to step (6)

# Comment starts 

aligned_cells = []
for line in LINES:
    cells = ds_proc[line]
    for i, cell in enumerate(cells):
        try:
            aligned_cell = align(cell, BASE_CURVE, rescale, rotation, reparameterization, k_sampling_points)
            file_path = os.path.join(data_folder, f"{line}_{i}.txt")
            np.savetxt(file_path, aligned_cell)
            aligned_cells.append(aligned_cell)
        except:
                print(f"first round: {line}, {i} cannot be aligned")

# First round alignment results


# (5) Calculate the mean shape and set it as the reference curve 
BASE_CURVE =  gs.mean(aligned_cells, axis=0)
reference_path = os.path.join(data_folder, f"reference.txt")
np.savetxt(reference_path, BASE_CURVE)

# Comment ends 


# (6) Set up variables and start alignment for the second round, \
# with the mean from the first round as the reference curve

reference_path = os.path.join(data_folder, f"reference.txt")

suffix = get_full_suffix()

data_folder = os.path.join(data_path, dataset_name, "aligned")
data_folder = os.path.join(data_folder, suffix)
print("data_folder for the second round is:", data_folder)

BASE_CURVE = np.loadtxt(reference_path)
print("BASE_CURVE shape is:", BASE_CURVE.shape)

for line in LINES:
    cells = ds_proc[line]
    for i, cell in enumerate(cells):
        try:
            aligned_cell = align(cell, BASE_CURVE, rescale, rotation, reparameterization, k_sampling_points)
            file_path = os.path.join(data_folder, f"{line}_{i}.txt")
            np.savetxt(file_path, aligned_cell)
        except:
                print(f"second round: {line}, {i} cannot be aligned")
