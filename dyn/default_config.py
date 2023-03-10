"""Default configuration for a run."""
import logging
from datetime import datetime

import geomstats.datasets.utils as data_utils
import numpy as np

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Dataset
dataset_name = ["cells"]  # ["circles"]
a_true = [0.5]
assert np.all([a > 0 and a < 1 for a in a_true]), "a_true must be in (0, 1)"
m_true = [1]

n_sampling_points = [30]
n_times = [100]
noise_std = [0.0]

# Learning
percent_train = 0.6
percent_val = 0.3

a_optimization = "sgd"  # "adam"
a_lr = 0.2  # default for adam: 0.001
a_init_diff = [0.2, -0.2]
max_iter = 30
tol = 0.001

# loading cells
cells, lines, treatments = data_utils.load_cells()
start_cell = cells[0]
end_cell = cells[1]
quotient = ["scaling", "rotation"]

# m's
m_grid = [1]  # , 2, 3]

# Run name in wandb
now = str(datetime.now().replace(second=0, microsecond=0).strftime("%m%d-%H:%M:%S"))
