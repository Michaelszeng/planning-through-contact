import time

from planning_through_contact.experiments.utils import get_default_plan_config, get_default_solver_params
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.planning.planar.planar_plan_config import PlanarPushingStartAndGoal
from planning_through_contact.planning.planar.utils import create_plan

solver_params = get_default_solver_params()

slider_type = "arbitrary"
arbitrary_shape_pickle_path = "arbitrary_shape_pickles/small_t_pusher.pkl"

# Initial poses
slider_initial_poses = [
    # PlanarPose(0.49916350319441594, -0.039942707670509475, 3.0109461664097754),
    PlanarPose(0.5592512081656711, -0.009460428568557777, 1.4896221348603618),
    PlanarPose(0.5579781495994708, -0.023871593865226498, 2.546246573254197),
    PlanarPose(0.5703330552747963, -0.09006898880198506, -2.9434357613583937),
    PlanarPose(0.5602175924492352, 0.0009429270364799958, -0.5868155045414416),
    PlanarPose(0.6335503271011114, 0.018630363655909615, -1.5048908010790143),
]

pusher_initial_pose = PlanarPose(0.587, 0.15, 0.0)

# Target poses
slider_target_pose = PlanarPose(0.587, -0.0355, 0.0)
pusher_target_pose = PlanarPose(0.587, 0.15, 0)

config = get_default_plan_config(
    slider_type=slider_type, arbitrary_shape_pickle_path=arbitrary_shape_pickle_path, use_case="drake_iiwa"
)


for slider_initial_pose in slider_initial_poses:
    start_and_goal = PlanarPushingStartAndGoal(
        slider_initial_pose=slider_initial_pose,
        slider_target_pose=slider_target_pose,
        pusher_initial_pose=pusher_initial_pose,
        pusher_target_pose=pusher_target_pose,
    )

    print(f"Starting planning for slider type: {slider_type}")

    # create_plan handles planning, folder creation, trajectory saving, and visualization
    start = time.time()
    result = create_plan(
        start_and_target=start_and_goal,
        config=config,
        solver_params=solver_params,
        output_folder="trajectories",
        output_name=f"arbitrary_small_t_pusher_trajectory_x={slider_initial_pose.x:.3f}_y={slider_initial_pose.y:.3f}_theta={slider_initial_pose.theta:.3f}",
        save_video=True,
        interpolate_video=True,
        do_rounding=True,
        save_traj=True,
        debug=True,
    )
    print(f"Time taken for create_plan: {time.time() - start}")

# # Run create_plan with the mode-sequence fixed
# from planning_through_contact.planning.planar.mpc import PlanarPushingMPC

# print("Constructing MPC Planner...")
# mpc = PlanarPushingMPC(config, start_and_goal, solver_params)
# t = 0
# active_vertices = mpc._get_remaining_mode_sequence(start_and_goal, t=t)

# print("Planning with MPC...")
# result = create_plan(
#     start_and_target=start_and_goal,
#     config=config,
#     solver_params=solver_params,
#     active_vertices=active_vertices,
#     output_folder="trajectories",
#     output_name=f"{slider_type}_trajectory_test",
#     save_video=True,
#     interpolate_video=True,
#     do_rounding=True,
#     save_traj=True,
#     debug=True,
# )
