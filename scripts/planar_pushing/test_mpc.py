from planning_through_contact.experiments.utils import get_default_plan_config, get_default_solver_params
from planning_through_contact.geometry.planar.planar_pose import PlanarPose
from planning_through_contact.planning.planar.mpc import PlanarPushingMPC
from planning_through_contact.planning.planar.planar_plan_config import PlanarPushingStartAndGoal

solver_params = get_default_solver_params()

slider_initial_pose = PlanarPose(0.14, 0.05, -0.8)
slider_type = "sugar_box"

# Target poses
slider_target_pose = PlanarPose(0.0, 0.0, 0.0)
pusher_target_pose = PlanarPose(-0.3, 0, 0)

config = get_default_plan_config(slider_type=slider_type, use_case="normal")
start_and_goal = PlanarPushingStartAndGoal(
    slider_initial_pose=slider_initial_pose,
    slider_target_pose=slider_target_pose,
    pusher_initial_pose=PlanarPose(-0.3, 0, 0),
    pusher_target_pose=pusher_target_pose,
)

print("Constructing MPC Planner...")
mpc = PlanarPushingMPC(config, start_and_goal, solver_params)

print("Planning with MPC...")
t = 3
current_slider_pose = mpc.original_traj.get_slider_planar_pose(t)
current_pusher_pose = mpc.original_traj.get_pusher_planar_pose(t)
path = mpc.plan(
    t=t,
    current_slider_pose=current_slider_pose,
    current_pusher_pose=current_pusher_pose,
    output_folder="trajectories_mpc",
    output_name=f"{slider_type}_trajectory_t={t}",
    save_video=True,
    interpolate_video=True,
    overlay_traj=True,
    save_traj=True,
)
print("Done!")
