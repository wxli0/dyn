import numpy as np 
from numba import jit, njit, prange


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

def remove_ds_align_one_layer(ds_align, delete_indices):
    count = 0
    for line, line_cells in ds_align.items():
        for i, _ in reversed(list(enumerate(line_cells))):
            if count in delete_indices:
                ds_align[line] = np.concatenate((ds_align[line][:i], ds_align[line][i+1:]), axis=0)
            count += 1
    return ds_align

def remove_ds_align_two_layer(ds_align, delete_indices):
    count = 0
    for treatment, treatment_values in ds_align.items():
        for line, line_cells in treatment_values.items():
            for i, _ in reversed(list(enumerate(line_cells))):
                if count in delete_indices:
                    ds_align[treatment][line] = np.concatenate((ds_align[treatment][line][:i], ds_align[treatment][line][i+1:]), axis=0)
                count += 1
    return ds_align


def remove_cells(cells, cell_shapes, lines, ds_align, delete_indices, num_layer):
    """ 
    Remove cells of control group from cells, cell_shapes, lines, ds_align,
    the parameters returned from load_treated_osteosarcoma_cells
    Also update n_cells

    :param list[int] delete_indices: the indices to delete
    """
    delete_indices.sort(reverse=True) # to prevent change in index when deleting elements
    
    # Delete elements
    cells = del_arr_elements(cells, delete_indices)
    cell_shapes = np.delete(np.array(cell_shapes), delete_indices, axis=0)
    lines = list(np.delete(np.array(lines), delete_indices, axis=0))
    if num_layer == 1:
        ds_align = remove_ds_align_one_layer(ds_align, delete_indices)
    elif num_layer == 2:
        ds_align = remove_ds_align_two_layer(ds_align, delete_indices)
    return cells, cell_shapes, lines, ds_align