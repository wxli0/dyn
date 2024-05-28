""" 
The script can be run in the following way:
python3.11 osteosarcoma_align.py --rescale --rotation --reparameterization (each option enables True for the boolean variable)
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
    for treatment in TREATMENTS:
        output_ds[treatment] = {}
        for line in LINES:
            output_list = []
            for one_cell in input_ds[treatment][line]:
                output_list.append(func(one_cell))
            output_ds[treatment][line] = gs.array(output_list)
    return output_ds


def interpolate(curve, nb_points):
    """Interpolate a discrete curve with nb_points from a discrete curve.
    TODO: does not handle the last point nicely

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


def check_duplicate(cell):
    """ 
    Return true if there are duplicate points in the cell
    """
    for i in range(cell.shape[0]-1):
        cur_coord = cell[i]
        next_coord = cell[i+1]
        if np.linalg.norm(cur_coord-next_coord) == 0:
            return True
        
    # Checking the last point vs the first poit
    if np.linalg.norm(cell[-1]-cell[0]) == 0:
        return True
    
    return False


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

dataset_name = 'osteosarcoma'
figs_dir = os.path.join("/home/wanxinli/dyn/dyn/saved_figs", dataset_name)
print(f"Will save figs to {figs_dir}")


# (2) Load data 
cells, lines, treatments = data_utils.load_cells()
print(f"Total number of cells : {len(cells)}")

TREATMENTS = gs.unique(treatments)
print(TREATMENTS)
LINES = gs.unique(lines)
print(LINES)
METRICS = ['SRV', 'Linear']


# (3) Prepare ds_proc
ds = {}

n_cells_arr = gs.zeros((3, 2))

for i, treatment in enumerate(TREATMENTS):
    print(f"{treatment} :")
    ds[treatment] = {}
    for j, line in enumerate(LINES):
        to_keep = gs.array(
            [
                one_treatment == treatment and one_line == line
                for one_treatment, one_line in zip(treatments, lines)
            ]
        )
        ds[treatment][line] = [
            cell_i for cell_i, to_keep_i in zip(cells, to_keep) if to_keep_i
        ]
        nb = len(ds[treatment][line])
        print(f"\t {nb} {line}")
        n_cells_arr[i, j] = nb

n_cells_df = pd.DataFrame({"dlm8": n_cells_arr[:, 0], "dunn": n_cells_arr[:, 1]})
n_cells_df = n_cells_df.set_index(TREATMENTS)


k_sampling_points = 2000


cell_rand = cells[gs.random.randint(len(cells))]
cell_interpolation = interpolate(cell_rand, k_sampling_points)


ds_interp = apply_func_to_ds(
    input_ds=ds, func=lambda x: interpolate(x, k_sampling_points)
)

ds_proc = apply_func_to_ds(ds_interp, func=lambda x: preprocess(x))


# (4) Parse command line arguments and start alignment for the first round
BASE_CURVE = generate_ellipse(k_sampling_points)
data_folder = os.path.join(data_path, dataset_name, "aligned")
parse_args()

suffix = get_full_suffix("first_round")

data_folder = os.path.join(data_folder, suffix)
print("data_folder for the first round is:", data_folder)

# If the first round has been done, we can comment the code below to jump to step (6)

# # Comment starts 

# aligned_cells = []
# for treatment in TREATMENTS:
#     for line in LINES:
#         cells = ds_proc[treatment][line]
#         for i, cell in enumerate(cells):
#             try:
#                 file_path = os.path.join(data_folder, f"{treatment}_{line}_{i}.txt")
#                 aligned_cell = align(cell, BASE_CURVE, rescale, rotation, reparameterization, k_sampling_points)
#                 np.savetxt(file_path, aligned_cell)
#                 aligned_cells.append(aligned_cell)
#             except:
#                 print(f"first round: {treatment}, {line}, {i} cannot be aligned")

# # # First round alignment results:
# # control, dlm8, 51 cannot be aligned
# # control, dunn, 8 cannot be aligned
# # control, dunn, 38 cannot be aligned
# # control, dunn, 80 cannot be aligned
# # control, dunn, 196 cannot be aligned
# # control, dunn, 197 cannot be aligned
# # cytd, dlm8, 4 cannot be aligned
# # cytd, dlm8, 6 cannot be aligned
# # cytd, dlm8, 30 cannot be aligned
# # cytd, dlm8, 46 cannot be aligned
# # cytd, dlm8, 69 cannot be aligned
# # cytd, dlm8, 81 cannot be aligned
# # cytd, dunn, 27 cannot be aligned
# # jasp, dlm8, 9 cannot be aligned
# # jasp, dlm8, 10 cannot be aligned
# # jasp, dlm8, 20 cannot be aligned
# # jasp, dlm8, 26 cannot be aligned
# # jasp, dlm8, 28 cannot be aligned
# # jasp, dlm8, 41 cannot be aligned
# # jasp, dunn, 8 cannot be aligned
# # jasp, dunn, 12 cannot be aligned
# # jasp, dunn, 85 cannot be aligned
# # jasp, dunn, 90 cannot be aligned


# # (5) Calculate the mean shape and set it as the reference curve 
# BASE_CURVE =  gs.mean(aligned_cells, axis=0)
# reference_path = os.path.join(data_folder, f"reference.txt")
# np.savetxt(reference_path, BASE_CURVE)

# # Comment ends 

# (6) Set up variables and start alignment for the second round, \
# with the mean from the first round as the reference curve

reference_path = os.path.join(data_folder, f"reference.txt")

suffix = get_full_suffix()

data_folder = os.path.join(data_path, dataset_name, "aligned")
data_folder = os.path.join(data_folder, suffix)
print("data_folder for the second round is:", data_folder)

BASE_CURVE = np.loadtxt(reference_path)
print("BASE_CURVE shape is:", BASE_CURVE.shape)

for treatment in TREATMENTS:
    for line in LINES:
        cells = ds_proc[treatment][line]
        for i, cell in enumerate(cells):
            try:
                file_path = os.path.join(data_folder, f"{treatment}_{line}_{i}.txt")
                aligned_cell = align(cell, BASE_CURVE, rescale, rotation, reparameterization, k_sampling_points)
                np.savetxt(file_path, aligned_cell)
            except:
                print(f"second round: {treatment}, {line}, {i} cannot be aligned")


# Second round alignment results
# second round: control, dlm8, 51 cannot be aligned
# second round: control, dunn, 8 cannot be aligned
# second round: control, dunn, 38 cannot be aligned
# second round: control, dunn, 80 cannot be aligned
# second round: control, dunn, 196 cannot be aligned
# second round: control, dunn, 197 cannot be aligned
# second round: cytd, dlm8, 4 cannot be aligned
# second round: cytd, dlm8, 6 cannot be aligned
# second round: cytd, dlm8, 30 cannot be aligned
# second round: cytd, dlm8, 46 cannot be aligned
# second round: cytd, dlm8, 69 cannot be aligned
# second round: cytd, dlm8, 81 cannot be aligned
# second round: cytd, dunn, 27 cannot be aligned
# second round: jasp, dlm8, 9 cannot be aligned
# second round: jasp, dlm8, 10 cannot be aligned
# second round: jasp, dlm8, 20 cannot be aligned
# second round: jasp, dlm8, 26 cannot be aligned
# second round: jasp, dlm8, 28 cannot be aligned
# second round: jasp, dlm8, 41 cannot be aligned
# second round: jasp, dunn, 8 cannot be aligned
# second round: jasp, dunn, 12 cannot be aligned
# second round: jasp, dunn, 85 cannot be aligned
# second round: jasp, dunn, 90 cannot be aligned