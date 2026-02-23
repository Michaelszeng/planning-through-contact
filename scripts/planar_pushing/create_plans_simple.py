from gcs_planar_pushing.experiments.utils import get_default_plan_config, get_default_solver_params
from gcs_planar_pushing.geometry.planar.planar_pose import PlanarPose
from gcs_planar_pushing.planning.planar.planar_plan_config import PlanarPushingStartAndGoal
from gcs_planar_pushing.planning.planar.utils import create_plan

solver_params = get_default_solver_params()

slider_initial_pose = PlanarPose(0.14, 0.05, -0.8)
slider_type = "arbitrary"
arbitrary_shape_pickle_path = "arbitrary_shape_pickles/small_t_pusher.pkl"

# Target poses
slider_target_pose = PlanarPose(0.0, 0.0, 0.0)
pusher_target_pose = PlanarPose(-0.3, 0, 0)

config = get_default_plan_config(
    slider_type=slider_type, arbitrary_shape_pickle_path=arbitrary_shape_pickle_path, use_case="drake_iiwa"
)
start_and_goal = PlanarPushingStartAndGoal(
    slider_initial_pose=slider_initial_pose,
    slider_target_pose=slider_target_pose,
    pusher_initial_pose=PlanarPose(-0.3, 0, 0),
    pusher_target_pose=pusher_target_pose,
)

print(f"Starting planning for slider type: {slider_type}")

# create_plan handles planning, folder creation, trajectory saving, and visualization
result = create_plan(
    start_and_target=start_and_goal,
    config=config,
    solver_params=solver_params,
    output_folder="trajectories",
    output_name="arbitrary_small_t_pusher_trajectory",
    save_video=True,
    interpolate_video=True,
    do_rounding=True,
    save_traj=True,
    debug=True,
)


# # Run create_plan with the mode-sequence fixed
# from gcs_planar_pushing.planning.planar.mpc import PlanarPushingMPC

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
