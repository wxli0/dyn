#!/usr/bin/env python
# coding: utf-8

# # Shape Analysis of Cancer Cells

# Lead author: Nina Miolane.
# 
# This notebook studies *Osteosarcoma* (bone cancer) cells and the impact of drug treatment on their *morphological shapes*, by analyzing cell images obtained from fluorescence microscopy. 
# 
# This analysis relies on the *elastic metric between discrete curves* from Geomstats. We will study to which extent this metric can detect how the cell shape is associated with the response to treatment.
# 
# This notebook is adapted from Florent Michel's submission to the [ICLR 2021 Computational Geometry and Topology challenge](https://github.com/geomstats/challenge-iclr-2021).

# <center>
#     <img src="figures/cells_image.png" width=600px />
# </center>

# Figure 1: Representative images of the cell lines using fluorescence microscopy, studied in this notebook (Image credit : Ashok Prasad). The cells nuclei (blue), the actin cytoskeleton (green) and the lipid membrane (red) of each cell are stained and colored. We only focus on the cell shape in our analysis.

# # 1. Introduction and Motivation

# Biological cells adopt a variety of shapes, determined by multiple processes and biophysical forces under the control of the cell. These shapes can be studied with different quantitative measures that reflect the cellular morphology [(MGCKCKDDRTWSBCC2018)](#References). With the emergence of large-scale biological cell image data, morphological studies have many applications. For example, measures of irregularity and spreading of cells allow accurate classification and discrimination between cancer cell lines treated with different drugs [(AXCFP2019)](#References).
# 
# As metrics defined on the shape space of curves, the *elastic metrics* [(SKJJ2010)](#References) implemented in Geomstats are a potential tool for analyzing and comparing biological cell shapes. Their associated geodesics and geodesic distances provide a natural framework for optimally matching, deforming, and comparing cell shapes.

# In[4]:


from decimal import Decimal
import matplotlib.pyplot as plt

import geomstats.backend as gs
import numpy as np
from nsimplices import *
from common import *
import scipy.stats as stats

gs.random.seed(2021)


# In[5]:


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

# In[6]:


import geomstats.datasets.utils as data_utils

cells, lines, treatments = data_utils.load_cells()
print(f"Total number of cells : {len(cells)}")


# The cells are grouped by treatment class in the dataset : 
# - the *control* cells, 
# - the cells treated with *Cytd*,
# - and the ones treated with *Jasp*. 
# 
# Additionally, in each of these classes, there are two cell lines : 
# - the *DLM8* cells, and
# - the *DUNN* ones.

# This is shown by displaying the unique elements in the lists `treatments` and `lines`:

# In[7]:


import pandas as pd

TREATMENTS = gs.unique(treatments)
print(TREATMENTS)
LINES = gs.unique(lines)
print(LINES)
METRICS = ['SRV', 'Linear']


# In[8]:


cell_idx = 1
plt.plot(cells[cell_idx][:, 0], cells[cell_idx][:, 1], "blue")
plt.plot(cells[cell_idx][0, 0], cells[cell_idx][0, 1], "blue", marker="o");


# The size of each class is displayed below:

# In[9]:


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



# The above code also created a dictionnary `ds`, that contains the cell boundaries data sorted by `treatment` and cell `line`. To access all the cells corresponding to a given treatment and a given cell line, we use the syntax `ds[treatment][line]` as in the following code that computes the number of cells in the cytd-dlm8 class.

# In[10]:


len(ds["cytd"]["dlm8"])


# We have organized the cell data into the dictionnary `ds`. Before proceeding to the actual data analysis, we provide an auxiliary function `apply_func_to_ds`.

# In[11]:


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


# Now we can move on to the actual data analysis, starting with a preprocessing of the cell boundaries.

# # 3. Preprocessing 
# 
# ### Interpolation: Encoding Discrete Curves With Same Number of Points
# 
# As we need discrete curves with the same number of sampled points to compute pairwise distances, the following interpolation is applied to each curve, after setting the number of sampling points.
# 
# To set up the number of sampling points, you can edit the following line in the next cell:

# In[12]:


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


# To illustrate the result of this interpolation, we compare for a randomly chosen cell the original curve with the correponding interpolated one (to visualize another cell, you can simply re-run the code).

# In[13]:


cell_rand = cells[gs.random.randint(len(cells))]
cell_interpolation = interpolate(cell_rand, k_sampling_points)

fig = plt.figure(figsize=(15, 5))

fig.add_subplot(121)
plt.plot(cell_rand[:, 0], cell_rand[:, 1])
plt.axis("equal")
plt.title(f"Original curve ({len(cell_rand)} points)")
plt.axis("off")

fig.add_subplot(122)
plt.plot(cell_interpolation[:, 0], cell_interpolation[:, 1])
plt.axis("equal")
plt.title(f"Interpolated curve ({k_sampling_points} points)")
plt.axis("off")

# plt.savefig(os.path.join(figs_dir, "interpolation.svg"))
# plt.savefig(os.path.join(figs_dir, "interpolation.pdf"))


# As the interpolation is working as expected, we use the auxiliary function `apply_func_to_ds` to apply the function `func=interpolate` to the dataset `ds`, i.e. the dictionnary containing the cells boundaries.
# 
# We obtain a new dictionnary, `ds_interp`, with the interpolated cell boundaries.

# In[14]:


ds_interp = apply_func_to_ds(
    input_ds=ds, func=lambda x: interpolate(x, k_sampling_points)
)


# For each key treatment-control, we check that the number of sampling points is the one expected, i.e. `k_sampling_points`:

# In[15]:


print(ds_interp["control"]["dunn"].shape)


# The shape of an array of cells in `ds_interp[treatment][cell]` is therefore: `("number of cells in treatment-line", "number of sampling points", 2)`, where 2 refers to the fact that we are considering cell shapes in 2D. 

# ### Visualization of Interpolated Dataset of Curves

# We visualize the curves obtained, for a sample of control cells and treated cells (top row shows control, i.e. non-treated cells; bottom rows shows treated cells) across cell lines (left and blue for dlm8 and right and orange for dunn).

# In[16]:


n_cells_to_plot = 5

fig = plt.figure(figsize=(16, 6))
count = 1
for i, treatment in enumerate(TREATMENTS):
    for line in LINES:
        cell_data = ds_interp[treatment][line]
        for i_to_plot in range(n_cells_to_plot):
            cell = gs.random.choice(cell_data)
            fig.add_subplot(3, 2 * n_cells_to_plot, count)
            count += 1
            plt.plot(cell[:, 0], cell[:, 1], color="C" + str(i))
            plt.xlim(-170, 170)
            plt.ylim(-170, 170)
            plt.axis("equal")
            plt.axis("off")
            if i_to_plot == n_cells_to_plot // 2:
                plt.title(f"{treatment}   -   {line}", fontsize=20)
plt.savefig(os.path.join(figs_dir, "sample_cells.svg"))
plt.savefig(os.path.join(figs_dir, "sample_cells.pdf"))


# Visual inspection of these curves seems to indicate more protusions appearing in treated cells, compared with control ones. This is in agreement with the physiological impact of the drugs, which are known to perturb the internal cytoskeleton connected to the cell membrane. Using the elastic metric, our goal will be to see if we can quantitatively confirm these differences.

# ### Remove duplicate samples in curves
# 
# During interpolation it is likely that some of the discrete curves in the dataset are downsampled from higher number of discrete data points to lower number of data points. Hence, two sampled data points that are close enough may end up overlapping after interpolation and hence such data points have to be dealt with specifically. 

# In[17]:


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


# ### Alignment
# 
# Our goal is to study the cell boundaries in our dataset, as points in a shape space of closed curves quotiented by translation, scaling, and rotation, so these transformations do not affect our measure of distance between curves.
# 
# In practice, we apply functions that were initially designed to center (substract the barycenter), rescale (divide by the Frobenius norm) and then align (find the rotation minimizing the L² distance) two sets of landmarks and reparamterization.
# 
# Additionally, since we are working with closed curves, the starting point associated with the parametrization of the discrete curves is also arbitrary. Thus, we conduct an exhaustive search to find which parametrization produces the best alignment according to the above procedure (i.e. the distance to the base curve is the smallest). 
# 
# This exhaustive search is implemented in the function `exhaustive_align` in `common.py`.
# 
# Since the alignment procedure takes 10+ hours, we ran `osteosarocoma_align.py` and saved the results in `~/dyn/datasets/osteosarcoma/aligned`

# In[18]:


ds_proc = apply_func_to_ds(ds_interp, func=lambda x: preprocess(x))

BASE_CURVE = ds_proc["control"]["dunn"][0]


# Load aligned cells from txt files. These files were generated by calling `exhaustive_align` function in `osteosarcoma_align.py`.

# In[19]:


aligned_folder = os.path.join(data_path, dataset_name, "aligned")

ds_align = {}
for treatment in TREATMENTS:
    ds_align[treatment] = {}
    for line in LINES:
        ds_align[treatment][line] = []
        cell_num = len(ds_proc[treatment][line])
        for i in range(cell_num):
            file_path = os.path.join(aligned_folder, f"{treatment}_{line}_{i}.txt")
            if not os.path.exists(file_path):
                file_path = os.path.join(aligned_folder, f"{treatment}_{line}_{i}_rotation_only.txt")
            cell = np.loadtxt(file_path)
            ds_align[treatment][line].append(cell)
        


# Check we did not loss any cells after alignment

# In[20]:


for treatment in TREATMENTS:
    for line in LINES:
        print(f"{treatment} and {line}: {len(ds_align[treatment][line])}")


# Check one cell that the aligner does not work

# In[30]:


from geomstats.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOrigin,
    DynamicProgrammingAligner,
    ElasticMetric,
    FTransform,
    IterativeHorizontalGeodesicAligner,
    L2CurvesMetric,
    SRVReparametrizationBundle,
    SRVRotationBundle,
    SRVRotationReparametrizationBundle,
    SRVTransform,
)

def exhaustive_align(curve, ref_curve, k_sampling_points, dynamic=False, rotation_only=False):
    """ 
    Quotient out
        - translation (move curve to start at the origin) 
        - rescaling (normalize to have length one)
        - rotation (try different starting points, during alignment)
        - reparametrization (resampling in the discrete case, during alignment)
    
    :param bool dynamic: Use dynamic aligner or not 
    :param bool rotation_only: quotient out rotation only rather than rotation and reparameterization

    """
    print("enter exhaustive_align")
    
    curves_r2 = DiscreteCurvesStartingAtOrigin(
        ambient_dim=2, k_sampling_points=k_sampling_points, equip=False
    )

    if dynamic:
        curves_r2.fiber_bundle = SRVReparametrizationBundle(curves_r2)
        curves_r2.fiber_bundle.aligner = DynamicProgrammingAligner()

    # Quotient out translation
    curve = curves_r2.projection(curve)
    ref_curve = curves_r2.projection(ref_curve)

    # Quotient out rescaling
    curve = curves_r2.normalize(curve)
    ref_curve = curves_r2.normalize(ref_curve)

    # Quotient out rotation and reparamterization
    curves_r2.equip_with_metric(SRVMetric)
    if rotation_only:
        curves_r2.equip_with_group_action("rotations")
    else:
        curves_r2.equip_with_group_action("rotations and reparametrizations")
        
    curves_r2.equip_with_quotient_structure()
    aligned_curve = curves_r2.fiber_bundle.align(curve, ref_curve)
    return aligned_curve


# In[41]:


cell = ds_proc['control']['dlm8'][70]
exhaustive_align(cell, BASE_CURVE, k_sampling_points, dynamic=True)

# Use dynamic aligner 
# 42 `y` must contain only finite values
# 63 `y` must contain only finite values
# 70 raise LinAlgError("SVD did not converge")
# 74 `y` must contain only finite values.
# 75 completed
# 99 `y` must contain only finite values.

# Use horizontal aligner 
# 42 `y` must contain only finite values.
# 63 `y` must contain only finite values
# 70 raise LinAlgError("SVD did not converge")
# 74 `y` must contain only finite values.
# 75 completed
# 99 `y` must contain only finite values.
