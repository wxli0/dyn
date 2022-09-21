"""Default configuration for a run."""
import logging
from datetime import datetime

import geomstats.datasets.utils as data_utils

# Can be replaced by logging.DEBUG or logging.WARNING
logging.basicConfig(level=logging.INFO)

# Dataset
dataset_name = ["circles"]
a_true = [1.0]
m_true = [1]

n_sampling_points = [50]
n_times = [25]
noise_var = [0.0]

# Learning
a_optimization = "gradient"
a_lr = 0.4
a_init_diff = [-0.2, 0.2]

# loading cells
cells, lines, treatments = data_utils.load_cells()
start_cell = cells[0]
end_cell = cells[1]
quotient = ["scaling", "rotation"]

# m's
m_grid = [1]  # , 2, 3]

# Run name in wandb
now = str(datetime.now().replace(second=0, microsecond=0).strftime("%m%d-%H:%M:%S"))
run_name = f"{now}_{dataset_name}"
