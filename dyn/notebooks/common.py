import numpy as np 
from numba import jit, njit, prange
import scipy.stats as stats
from scipy.integrate import simpson
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


from geomstats.geometry.discrete_curves import (
    DiscreteCurvesStartingAtOrigin,
    DynamicProgrammingAligner,
    SRVReparametrizationBundle,
    SRVMetric
)


def del_arr_elements(arr, indices):
    """
    Delete elements in indices from array arr
    """

    # Sort the indices in reverse order to avoid index shifting during deletion
    indices.sort(reverse=True)

    # Iterate over each index in the list of indices
    for index in indices:
        del arr[index]
    return arr



@jit(nopython=False, forceobj=True)
def parallel_dist(cells, dist_fun, k_sampling_points):
    pairwise_dists = np.zeros((cells.shape[0], cells.shape[0]))
    for i in prange(cells.shape[0]):
        for j in prange(i + 1, cells.shape[0]):
            pairwise_dists[i, j] = dist_fun(cells[i], cells[j]) / k_sampling_points
    pairwise_dists += pairwise_dists.T
    return pairwise_dists

def remove_ds_one_layer(ds, delete_indices):
    count = 0
    for line, line_cells in ds.items():
        for i, _ in reversed(list(enumerate(line_cells))):
            if count in delete_indices:
                ds[line] = np.concatenate((ds[line][:i], ds[line][i+1:]), axis=0)
            count += 1
    return ds

def remove_ds_two_layer(ds, delete_indices):
    count = 0
    for treatment, treatment_values in ds.items():
        for line, line_cells in treatment_values.items():
            for i, _ in reversed(list(enumerate(line_cells))):
                if count in delete_indices:
                    ds[treatment][line] = np.concatenate((ds[treatment][line][:i], ds[treatment][line][i+1:]), axis=0)
                count += 1
    return ds


def remove_cells_one_layer(cells, cell_shapes, lines, ds_proc, ds_align, delete_indices):
    """ 
    Remove cells of control group from cells, cell_shapes, ds,
    the parameters returned from load_treated_osteosarcoma_cells
    Also update n_cells

    :param list[int] delete_indices: the indices to delete
    """
    delete_indices = sorted(delete_indices, reverse=True) # to prevent change in index when deleting elements
    
    # Delete elements
    cells = del_arr_elements(cells, delete_indices)
    cell_shapes = np.delete(np.array(cell_shapes), delete_indices, axis=0)
    lines = list(np.delete(np.array(lines), delete_indices, axis=0))
    ds_proc = remove_ds_one_layer(ds_proc, delete_indices)
    ds_align = remove_ds_one_layer(ds_align, delete_indices)

    return cells, cell_shapes, lines,  ds_proc, ds_align


def remove_cells_two_layer(cells, cell_shapes, lines, treatments, ds_proc, ds_align, delete_indices):
    """ 
    Remove cells of control group from cells, cell_shapes, ds,
    the parameters returned from load_treated_osteosarcoma_cells
    Also update n_cells

    :param list[int] delete_indices: the indices to delete
    """
    delete_indices = sorted(delete_indices, reverse=True) # to prevent change in index when deleting elements
    
    # Delete elements
    cells = del_arr_elements(cells, delete_indices)
    cell_shapes = np.delete(np.array(cell_shapes), delete_indices, axis=0)
    lines = list(np.delete(np.array(lines), delete_indices, axis=0))
    treatments = list(np.delete(np.array(treatments), delete_indices, axis=0))
    ds_proc = remove_ds_two_layer(ds_proc, delete_indices)
    ds_align = remove_ds_two_layer(ds_align, delete_indices)

    return cells, cell_shapes, lines, treatments, ds_proc, ds_align



def remove_cell_shapes(cell_shapes, ds_align, delete_indices, num_layer):
    """ 
    Remove cells of control group from cells, cell_shapes, lines, ds_align,
    the parameters returned from load_treated_osteosarcoma_cells
    Also update n_cells

    :param list[int] delete_indices: the indices to delete
    """
    delete_indices.sort(reverse=True) # to prevent change in index when deleting elements
    
    # Delete elements
    cell_shapes = np.delete(np.array(cell_shapes), delete_indices, axis=0)
    if num_layer == 1:
        ds_align = remove_ds_align_one_layer(ds_align, delete_indices)
    elif num_layer == 2:
        ds_align = remove_ds_align_two_layer(ds_align, delete_indices)
    return cell_shapes, ds_align


def overlap_ratio(distance1, distance2):
    """ 
    Calculate the ratio of overlap regions between the histograms of distance1 and distance

    :param list[float] distance1: list of positive distances 
    :param list[float] distance2: list of positive distances 
    :param function kde1: the kernel density estimation of distance1
    :param function kde2: the kernel density estimation of distance2
    """

    # Define a common set of points for evaluation (covering the range of both datasets)
    x_eval = np.linspace(min(np.min(distance1), np.min(distance2)), max(np.max(distance1), np.max(distance2)), 1000)

    # Create KDEs for the two datasets
    kde1 = stats.gaussian_kde(distance1)
    kde2 = stats.gaussian_kde(distance2)

    # Evaluate the KDEs on these points
    kde_values1 = kde1(x_eval)
    kde_values2 = kde2(x_eval)

    # Find the minimum of the two KDEs at each point to determine the overlap
    overlap_values = np.minimum(kde_values1, kde_values2)

    # Integrate the overlap using the composite Simpson's rule
    overlap_area = simpson(overlap_values, x=x_eval)

    # Calculate the total area under one of the KDEs as a reference (should be close to 1)
    total_area = simpson(kde_values1, x=x_eval)  # or use kde_values2, should be about the same

    # Calculate the ratio of the overlap
    overlap_ratio = (overlap_area / total_area) 

    return overlap_ratio


def knn_score(pos, labels):
    clf = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(clf, pos, labels, cv=5, scoring='accuracy')
    return scores.mean()


def exhaustive_align(curve, ref_curve, k_sampling_points, rescale=True, dynamic=False, rotation_only=False):
    """ 
    Quotient out
        - translation (move curve to start at the origin) 
        - rescaling (normalize to have length one)
        - rotation (try different starting points, during alignment)
        - reparametrization (resampling in the discrete case, during alignment)
    
    :param bool rescale: quotient out rescaling or not 
    :param bool dynamic: Use dynamic aligner or not 
    :param bool rotation_only: quotient out rotation only rather than rotation and reparameterization

    """
    
    curves_r2 = DiscreteCurvesStartingAtOrigin(
        ambient_dim=2, k_sampling_points=k_sampling_points, equip=False
    )

    if dynamic:
        print("Use dynamic programming aligner")
        curves_r2.fiber_bundle = SRVReparametrizationBundle(curves_r2)
        curves_r2.fiber_bundle.aligner = DynamicProgrammingAligner()

    # Quotient out translation
    print("Quotientint out translation")
    curve = curves_r2.projection(curve)
    ref_curve = curves_r2.projection(ref_curve)

    # Quotient out rescaling
    if rescale:
        print("Quotientint out rescaling")
        curve = curves_r2.normalize(curve)
        ref_curve = curves_r2.normalize(ref_curve)

    # Quotient out rotation and reparamterization
    curves_r2.equip_with_metric(SRVMetric)
    if rotation_only:
        print("Quotientint out rotation")
        curves_r2.equip_with_group_action("rotations")
    else:
        print("Quotienting out rotation and reparamterization")
        curves_r2.equip_with_group_action("rotations and reparametrizations")
        
    curves_r2.equip_with_quotient_structure()
    aligned_curve = curves_r2.fiber_bundle.align(curve, ref_curve)
    return aligned_curve