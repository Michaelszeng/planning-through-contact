import os
import time

import matplotlib
import numpy as np

matplotlib.use("QtAgg")  # must run before importing pyplot / Drake visualizers

from gcs_planar_pushing.experiments.utils import get_default_plan_config, get_default_solver_params
from gcs_planar_pushing.geometry.planar.planar_pose import PlanarPose
from gcs_planar_pushing.planning.planar.mpc import PlanarPushingMPC
from gcs_planar_pushing.planning.planar.planar_plan_config import PlanarPushingStartAndGoal

solver_params = get_default_solver_params(debug=True)
# solver_params.save_solver_output = True
# solver_params.print_solver_output = True

slider_type = "arbitrary"
arbitrary_shape_pickle_path = "arbitrary_shape_pickles/small_t_pusher.pkl"

# Initial poses
slider_initial_pose = PlanarPose(0.649973205950237, -0.016820301165871604, -2.1917248908936977)
# slider_initial_pose = PlanarPose(0.49916350319441594, -0.039942707670509475, 3.0109461664097754)
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

print("Constructing MPC Planner...")

# t = 5
# is_in_contact = True  # Whether the system is in contact with the slider at time t
# current_segment_index = 2  # The segment index that the system is currently in at/before time t
t = 0
is_in_contact = False  # Whether the system is in contact with the slider at time t
current_segment_index = 0  # The segment index that the system is currently in at/before time t

# Load original plan from cache if it exists to speed up testing
CACHE_PATH = f"mpc_path_cache_slider_initial_pose={slider_initial_pose.x},{slider_initial_pose.y},{slider_initial_pose.theta}_t={t}.pkl"
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

mpc.current_segment_index = current_segment_index

print("Planning with MPC...")
current_slider_pose = mpc.original_traj.get_slider_planar_pose(t)
current_pusher_pose = mpc.original_traj.get_pusher_planar_pose(t)
current_pusher_velocity = mpc.original_traj.get_pusher_velocity(t)

# Try some perturbation
pos = current_pusher_pose.pos()
theta = current_pusher_pose.theta
current_pusher_pose = PlanarPose(pos[0, 0] + 0.001, pos[1, 0], theta)
current_pusher_velocity = current_pusher_velocity * 1.05

print(f"current_slider_pose: {current_slider_pose}")
print(f"current_pusher_pose: {current_pusher_pose}")
print(f"current_pusher_velocity: {current_pusher_velocity}")

start = time.time()
path = mpc.plan(
    t=t,
    current_slider_pose=current_slider_pose,
    current_pusher_pose=current_pusher_pose,
    current_pusher_velocity=current_pusher_velocity,
    is_in_contact=is_in_contact,
    enforce_monotonic_progress=True,
    output_folder="trajectories_mpc",
    output_name=f"arbitrary_small_t_pusher_trajectory_slider_initial_pose={slider_initial_pose.x},{slider_initial_pose.y},{slider_initial_pose.theta}_t={t}",
    save_video=True,
    interpolate_video=True,
    overlay_traj=True,
    save_traj=True,
)
print(f"Time taken for MPC replan: {time.time() - start}")
print("Done!")
