from decimal import Decimal
import matplotlib.pyplot as plt

import geomstats.backend as gs
import numpy as np
from nsimplices import *
from common import *
import scipy.stats as stats

gs.random.seed(2021)


# In[2]:


base_path = "/home/wanxinli/dyn/dyn/"
data_path = os.path.join(base_path, "datasets")

dataset_name = 'osteosarcoma'
figs_dir = os.path.join("/home/wanxinli/dyn/dyn/saved_figs", dataset_name)
print(f"Will save figs to {figs_dir}")


# # 2. Dataset Description

# We study a dataset of mouse *Osteosarcoma* imaged cells [(AXCFP2019)](#References). The dataset contains two different cancer cell lines : *DLM8* and *DUNN*, respectively representing a more agressive and a less agressive cancer.  Among these cells, some have also been treated with different single drugs that perturb the cellular cytoskeleton. Overall, we can label each cell according to their cell line (*DLM8* and *DUNN*), and also if it is a *control* cell (no treatment), or has been treated with one of the following drugs : *Jasp* (jasplakinolide) and *Cytd* (cytochalasin D).
# 
# Each cell comes from a raw image containing a set of cells, which was thresholded to generate binarized images.
# 
# <td>
#     <img src="figures/binarized_cells.png" width=400px />
# </td>
# 
# After binarizing the images, contouring was used to isolate each cell, and to extract their boundaries as a counter-clockwise ordered list of 2D coordinates, which corresponds to the representation of discrete curve in Geomstats. We load these discrete curves into the notebook.

# In[3]:


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


k_sampling_points = 2000


cell_rand = cells[gs.random.randint(len(cells))]
cell_interpolation = interpolate(cell_rand, k_sampling_points)


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

ds_proc = apply_func_to_ds(ds_interp, func=lambda x: preprocess(x))




def rotation_align(curve, base_curve, k_sampling_points):
    """Align curve to base_curve to minimize the L² distance.

    Returns
    -------
    aligned_curve : discrete curve
    """
    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    base_curve = gs.array(base_curve)

    # Rotation is done after projection, so the origin is removed
    total_space = DiscreteCurvesStartingAtOrigin(k_sampling_points=k_sampling_points-1)
    total_space.fiber_bundle = SRVRotationBundle(total_space)

    for shift in range(nb_sampling):
        reparametrized = [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        print(len(reparametrized), len(base_curve))
        aligned = total_space.fiber_bundle.align(
            point=gs.array(reparametrized), base_point=base_curve
        )
        distances[shift] = np.linalg.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)
    reparametrized_min = [
        curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
    ]
    aligned_curve = total_space.fiber_bundle.align(
        point=gs.array(reparametrized_min), base_point=base_curve
    )
    return aligned_curve



def align(point, base_point, rescale, rotation, reparameterization):
    """
    Align point and base_point via quotienting out translation, rescaling and reparameterization

    Right now we do not quotient out rotation since
    - Current geomstats does not support changing aligner for SRVRotationReparametrizationBundle
    - The base curve we chose is a unit circle, so quotienting out rotation won't affect the result too much
    """

    total_space = DiscreteCurvesStartingAtOrigin(k_sampling_points=k_sampling_points)
   
    
    # Quotient out translation
    point = total_space.projection(point)
    base_point = total_space.projection(base_point)

    # Quotient out rescaling
    if rescale:
        point = total_space.normalize(point)
        base_point = total_space.normalize(base_point)
    
    # Quotient out rotation
    if rotation:
        point = rotation_align(point, base_point, k_sampling_points)

    # Quotient out reparameterization
    if reparameterization:
        aligner = DynamicProgrammingAligner(total_space)
        total_space.fiber_bundle = SRVReparametrizationBundle(total_space, aligner=aligner)
        point = total_space.fiber_bundle.align(point, base_point)
    return point


BASE_CURVE = generate_ellipse(k_sampling_points)

data_folder = os.path.join(data_path, dataset_name, "aligned")

suffix = 'projection'
rescale = True
rotation = True
reparameterization = False
add_suffix = 'rotation_align'

if rescale:
    suffix += '_rescale'

if rotation:
    suffix += '_rotation'

if reparameterization:
    suffix += '_reparameterization'

if add_suffix is not None:
    suffix += "_"+add_suffix

data_folder = os.path.join(data_folder, suffix)


for treatment in TREATMENTS:
    for line in LINES:
        cells = ds_proc[treatment][line]
        for i, cell in enumerate(cells):
            # try:
            aligned_cell = align(cell, BASE_CURVE, rescale, rotation, reparameterization)
            file_path = os.path.join(data_folder, f"{treatment}_{line}_{i}.txt")
            np.savetxt(file_path, aligned_cell)



