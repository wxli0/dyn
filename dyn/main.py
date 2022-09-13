"""Main script."""

import logging
import os
import tempfile

import default_config
import matplotlib.pyplot as plt
import numpy as np
import wandb

import dyn.dyn.datasets.synthetic as synthetic
import dyn.dyn.features.optimize_am as optimize_am

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

logging.info(f"Starting run {default_config.run_name}")
wandb.init(
    project="metric_learning",
    dir=tempfile.gettempdir(),
    config={
        "run_name": default_config.run_name,
        "dataset_name": default_config.dataset_name,
        "a_true": default_config.a_true,
        "m_true": default_config.m_true,
        "noise_var": default_config.noise_var,
        "n_sampling_points": default_config.n_sampling_points,
        "n_times": default_config.n_times,
        "a_initialization": default_config.a_initialization,
        "m_grid": default_config.m_grid,
        "a_optimization": default_config.a_optimization,
        "a_lr": default_config.a_lr,
    },
)

config = wandb.config
wandb.run.name = config.run_name

logging.info(
    f"Load dataset {config.dataset_name} with "
    "a_true = {config.a_true} and m_true = {config.m_true}"
)
dataset_of_trajectories = None
if config.dataset_name == "synthetic_circle_to_ellipse":
    if config.m_true == 1 and config.a_true == 1:
        dataset_of_trajectories = synthetic.geodesics_circle_to_ellipse(
            n_geodesics=1, n_times=config.n_times, n_points=config.n_sampling_points
        )
if dataset_of_trajectories is None:
    raise NotImplementedError()

one_trajectory = dataset_of_trajectories[0]
print(f"The shape of one_trajectory is: {one_trajectory.shape}")

if config.a_initialization == "close_to_ground_truth":
    init_a = config.a_true  # - 0.2
elif config.a_initialization == "random":
    init_a = 0.5
else:
    raise NotImplementedError()

logging.info("Find best a and m corresponding to the trajectory.")
best_a, best_m, best_r2, best_r2_from_m, as_steps, r2s_steps = optimize_am.find_best_am(
    one_trajectory, init_a=init_a, m_grid=config.m_grid, a_lr=config.a_lr
)

logging.info("Save results in wandb and local directory.")
best_amr2_table = wandb.Table(
    columns=["best_a", "best_m", "best_r2"], data=[[best_a, best_m, best_r2]]
)

best_r2_from_m_table = wandb.Table(
    columns=[f"m = {m}" for m in list(config.m_grid)], data=[list(best_r2_from_m)]
)

wandb.log({"best_amr2": best_amr2_table, "r2_from_m_results": best_r2_from_m_table})

fig, axs = plt.subplots(1, 2, figsize=(10, 5))


for i_m, m in enumerate(config.m_grid):
    print(f"Log results for optimization on a for m = {m}")
    axs[0].plot(np.arange(0, len(r2s_steps[i_m])), r2s_steps[i_m], label=f"m = {m}")

    a_r2_steps = wandb.Table(
        columns=["a", "r2"],
        data=[[float(a), float(r)] for a, r in zip(as_steps[i_m], r2s_steps[i_m])],
    )
    table_key = f"a_r2_steps_m_{m}"
    wandb.log({table_key: a_r2_steps})

wandb.finish()
