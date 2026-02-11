import os
import time

import matplotlib
import numpy as np

matplotlib.use("QtAgg")  # must run before importing pyplot / Drake visualizers

from gcs_planar_pushing.experiments.utils import get_default_plan_config, get_default_solver_params
from gcs_planar_pushing.geometry.planar.planar_pose import PlanarPose
from gcs_planar_pushing.planning.planar.mpc import PlanarPushingMPC
from gcs_planar_pushing.planning.planar.planar_plan_config import PlanarPushingStartAndGoal

solver_params = get_default_solver_params()

slider_type = "arbitrary"
arbitrary_shape_pickle_path = "arbitrary_shape_pickles/small_t_pusher.pkl"

# Initial poses
slider_initial_pose = PlanarPose(0.49916350319441594, -0.039942707670509475, 3.0109461664097754)
pusher_initial_pose = PlanarPose(0.587, 0.15, 0.0)

# Target poses
slider_target_pose = PlanarPose(0.587, -0.0355, 0.0)
pusher_target_pose = PlanarPose(0.587, 0.15, 0)

config = get_default_plan_config(
    slider_type=slider_type, arbitrary_shape_pickle_path=arbitrary_shape_pickle_path, use_case="normal"
)
start_and_goal = PlanarPushingStartAndGoal(
    slider_initial_pose=slider_initial_pose,
    slider_target_pose=slider_target_pose,
    pusher_initial_pose=pusher_initial_pose,
    pusher_target_pose=pusher_target_pose,
)

t = 2.01

print("Constructing MPC Planner...")

# Load original plan from cache if it exists to speed up testing
CACHE_PATH = f"mpc_path_cache_t={t}.pkl"
if os.path.exists(CACHE_PATH):
    print(f"Loading cached path from {CACHE_PATH}")
    mpc = PlanarPushingMPC(
        config,
        start_and_goal,
        solver_params,
        plan=False,
    )
    mpc.load_original_path(CACHE_PATH)
else:
    print("Computing fresh path...")
    mpc = PlanarPushingMPC(
        config,
        start_and_goal,
        solver_params,
        plan=True,
    )
    mpc.original_path.save(CACHE_PATH)

print("Planning with MPC...")
current_slider_pose = mpc.original_traj.get_slider_planar_pose(t)
current_pusher_pose = mpc.original_traj.get_pusher_planar_pose(t)
current_pusher_velocity = mpc.original_traj.get_pusher_velocity(t)

# current_slider_pose = PlanarPose(0.4991755125699597, -0.03995089541666773, 3.0110513740188143)
# current_pusher_pose = PlanarPose(0.5861310676117617, 0.1495927046771189, 0.00010802307327750782)

# current_slider_pose = mpc.original_traj.get_slider_planar_pose(0.1)
# current_pusher_pose = mpc.original_traj.get_pusher_planar_pose(0.1)
# current_pusher_velocity = mpc.original_traj.get_pusher_velocity(0.1)

# current_slider_pose = PlanarPose(0.49916350319441594, -0.039942707670509475, 3.0109461664097754)
# current_pusher_pose = PlanarPose(0.587, 0.15, 0.0)
# current_pusher_velocity = np.array([2.3674e-07, -3.6202e-08])

print(f"current_slider_pose: {current_slider_pose}")
print(f"current_pusher_pose: {current_pusher_pose}")
print(f"current_pusher_velocity: {current_pusher_velocity}")

start = time.time()
path = mpc.plan(
    t=t,
    current_slider_pose=current_slider_pose,
    current_pusher_pose=current_pusher_pose,
    current_pusher_velocity=current_pusher_velocity,
    output_folder="trajectories_mpc",
    output_name=f"arbitrary_small_t_pusher_trajectory_t={t}",
    save_video=True,
    interpolate_video=True,
    overlay_traj=True,
    save_traj=True,
)
print(f"Time taken for MPC replan: {time.time() - start}")
print("Done!")
