import os
import time
import traceback

import matplotlib
import numpy as np

matplotlib.use("Agg")  # must run before importing pyplot / Drake visualizers

from gcs_planar_pushing.experiments.utils import get_default_plan_config, get_default_solver_params
from gcs_planar_pushing.geometry.planar.planar_pose import PlanarPose
from gcs_planar_pushing.planning.planar.mpc import PlanarPushingMPC
from gcs_planar_pushing.planning.planar.planar_plan_config import PlanarPushingStartAndGoal

solver_params = get_default_solver_params()
# solver_params = get_default_solver_params(debug=True)
# solver_params.save_solver_output = True
# solver_params.print_solver_output = True

slider_type = "arbitrary"
arbitrary_shape_pickle_path = "arbitrary_shape_pickles/small_t_pusher.pkl"

# Initial poses
slider_initial_poses = {
    "SEED=0": PlanarPose(0.6417846749285816, -0.10456398587083891, -2.8749678804892476),  # SEED = 0
    "SEED=1": PlanarPose(0.49916350319441594, -0.039942707670509475, 3.0109461664097754),  # SEED = 1
    "SEED=2": PlanarPose(0.649973205950237, -0.016820301165871604, -2.1917248908936977),  # SEED = 2
    "SEED=3": PlanarPose(0.5592512081656711, -0.009460428568557777, 1.4896221348603618),  # SEED = 3
}

pusher_initial_pose = PlanarPose(0.587, 0.15, 0.0)

# Target poses
slider_target_pose = PlanarPose(0.587, -0.0355, 0.0)
pusher_target_pose = PlanarPose(0.587, 0.15, 0)

config = get_default_plan_config(
    slider_type=slider_type, arbitrary_shape_pickle_path=arbitrary_shape_pickle_path, use_case="drake_iiwa"
)

for name, slider_initial_pose in slider_initial_poses.items():
    try:
        print("=" * 100 + "\nConstructing MPC Planner...")
        print(f"Trial name: {name}\nSlider initial pose: {slider_initial_pose}")

        start_and_goal = PlanarPushingStartAndGoal(
            slider_initial_pose=slider_initial_pose,
            slider_target_pose=slider_target_pose,
            pusher_initial_pose=pusher_initial_pose,
            pusher_target_pose=pusher_target_pose,
        )

        # t = 5
        # is_in_contact = True  # Whether the system is in contact with the slider at time t
        # current_segment_index = 2  # The segment index that the system is currently in at/before time t
        t = 0
        is_in_contact = False  # Whether the system is in contact with the slider at time t
        current_segment_index = 1  # The segment index that the system is currently in at/before time t
        # NOTE: this should never be 0, which corresponds to source

        # Load original plan from cache if it exists to speed up testing
        CACHE_PATH = f"mpc_path_cache_{name}_t={t}.pkl"
        if os.path.exists(CACHE_PATH):
            print(f"Loading cached path from {CACHE_PATH}")
            mpc = PlanarPushingMPC(
                config,
                start_and_goal,
                solver_params,
                planner_freq=10.0,
                plan=False,
                output_folder="trajectories_mpc",
                output_name=f"arbitrary_small_t_pusher_trajectory_ORIGINAL_{name}",
                save_video=True,
                interpolate_video=True,
            )
            mpc.load_original_path(CACHE_PATH)
        else:
            print("Computing fresh path...")
            mpc = PlanarPushingMPC(
                config,
                start_and_goal,
                solver_params,
                planner_freq=10.0,
                plan=True,
                output_folder="trajectories_mpc",
                output_name=f"arbitrary_small_t_pusher_trajectory_ORIGINAL_{name}",
                save_video=True,
                interpolate_video=True,
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
        current_pusher_pose = PlanarPose(pos[0, 0] + 0.0005, pos[1, 0], theta)
        current_pusher_velocity = current_pusher_velocity * 1.01

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
            output_name=f"arbitrary_small_t_pusher_trajectory_{name}_t={t}",
            save_video=True,
            interpolate_video=True,
            rounded=True,
            overlay_traj=True,
        )
        print(f"Time taken for MPC replan: {time.time() - start}")
        print("Done!")
    except Exception as e:
        print(f"Error for trial {name}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("Moving to next trial...")
