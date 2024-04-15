from decimal import Decimal
import matplotlib.pyplot as plt

import geomstats.backend as gs
import numpy as np
from nsimplices import *
from common import *
import scipy.stats as stats

gs.random.seed(2021)


# In[3]:


base_path = "/home/wanxinli/dyn/dyn/"
data_path = os.path.join(base_path, "datasets")

dataset_name = 'osteosarcoma'
figs_dir = os.path.join("/home/wanxinli/dyn/dyn/saved_figs", dataset_name)
rescale = True
print(f"Will save figs to {figs_dir}")


import geomstats.datasets.utils as data_utils

cells, lines, treatments = data_utils.load_cells()
print(f"Total number of cells : {len(cells)}")


import pandas as pd

TREATMENTS = gs.unique(treatments)
print(TREATMENTS)
LINES = gs.unique(lines)
print(LINES)
METRICS = ['SRV', 'Linear']



ds = {}

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


ds_proc = apply_func_to_ds(ds_interp, func=lambda x: preprocess(x))


BASE_CURVE = generate_circle_points(k_sampling_points)



rescale = True
reparameterization = True
dynamic = True

treatment = 'control'
line = 'dlm8'
index = 42
unaligned_cell = ds_proc[treatment][line][index]
# unaligned_cell = np.array([[0,0], [0,1], [0,2], [1,1]])

def check_duplicates(points):
    # Check if the number of unique rows is less than the total number of rows
    unique_points = np.unique(points, axis=0)
    return unique_points.shape[0] != points.shape[0]

print("check for duplicates in base curve:", check_duplicates(BASE_CURVE))
print("check for duplicates in unaligned cell:", check_duplicates(unaligned_cell))


def align(point, base_point):
    total_space = DiscreteCurvesStartingAtOrigin(k_sampling_points=k_sampling_points)
    total_space.fiber_bundle = SRVReparametrizationBundle(total_space, aligner=DynamicProgrammingAligner)
    # total_space.fiber_bundle.aligner = DynamicProgrammingAligner(total_space)
    # other_aligner = DynamicProgrammingAligner()
    # sampling_points = gs.linspace(0.0, 1.0, k_sampling_points)

    base_point = total_space.projection(base_point)
    point = total_space.projection(point)
    # other_aligned = other_aligner.align(total_space, point, base_point)
    other_aligned = total_space.fiber_bundle.align(point, base_point)
    return other_aligned


aligned_cell = align(unaligned_cell, BASE_CURVE)

# aligned_cell = exhaustive_align(unaligned_cell, BASE_CURVE, k_sampling_points, rescale=rescale, reparameterization=reparameterization, dynamic=dynamic)


print(aligned_cell)


print(unaligned_cell)

