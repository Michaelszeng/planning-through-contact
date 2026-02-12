import os
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pydrake.geometry.optimization as opt

from gcs_planar_pushing.geometry.collision_geometry.collision_geometry import (
    ContactLocation,
    PolytopeContactLocation,
)
from gcs_planar_pushing.geometry.planar.abstract_mode import AbstractContactMode
from gcs_planar_pushing.geometry.planar.face_contact import FaceContactMode
from gcs_planar_pushing.geometry.planar.non_collision import (
    NonCollisionMode,
    NonCollisionVariables,
)
from gcs_planar_pushing.geometry.planar.non_collision_subgraph import (
    VertexModePair,
    gcs_add_edge_with_continuity,
)
from gcs_planar_pushing.geometry.planar.planar_pose import PlanarPose
from gcs_planar_pushing.geometry.planar.planar_pushing_path import PlanarPushingPath
from gcs_planar_pushing.geometry.planar.planar_pushing_trajectory import (
    NonCollisionTrajSegment,
)
from gcs_planar_pushing.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarSolverParams,
)
from gcs_planar_pushing.planning.planar.planar_pushing_planner import PlanarPushingPlanner
from gcs_planar_pushing.visualize.colors import COLORS
from gcs_planar_pushing.visualize.planar_pushing import (
    make_traj_figure,
    visualize_planar_pushing_trajectory,
)

GcsVertex = opt.GraphOfConvexSets.Vertex


class PlanarPushingMPC:
    def __init__(
        self,
        config: PlanarPlanConfig,
        start_and_goal: PlanarPushingStartAndGoal,
        solver_params: PlanarSolverParams,
        plan: bool = True,
    ):
        """
        Formulate initial GCS Problem and do full solve
        """
        self.config = config
        self.solver_params = solver_params
        self.planner = PlanarPushingPlanner(config)
        self.planner.config.start_and_goal = start_and_goal
        self.planner.formulate_problem()

        if plan:
            path = self.planner.plan_path(solver_params, store_result=False)
            assert path is not None
            self.original_path = path
            self.original_traj = path.to_traj()
            print(f"Original Path Mode Sequence: {[pair.vertex.name() for pair in self.original_path.pairs]}")
        else:
            self.original_path = None
            self.original_traj = None

        # Store "short" mode vertex, which is a copy of the original vertex in the GCS but with a shorter time to
        # account for the passage of time during MPC iterations.
        self.previous_short_mode_vertex = None  # Optional[GcsVertex]
        self.current_segment_index = 0

    def load_original_path(self, filename: str) -> None:
        """
        Loads the original path from a file.
        """
        all_pairs = self.planner._get_all_vertex_mode_pairs()
        self.original_path = PlanarPushingPath.load(filename, self.planner.gcs, all_pairs)
        self.original_traj = self.original_path.to_traj()
        self.current_segment_index = 0
        print(f"Original Path Mode Sequence: {[pair.vertex.name() for pair in self.original_path.pairs]}")

    def _is_state_in_mode(
        self,
        slider_pose: PlanarPose,
        pusher_pose: PlanarPose,
        mode: AbstractContactMode,
        tol: float = 1e-10,
    ) -> bool:
        """
        Checks if the given state is valid for the given mode.
        """
        p_WP = pusher_pose.pos()
        R_WB = slider_pose.two_d_rot_matrix()
        p_WB = slider_pose.pos()
        p_BP = R_WB.T.dot(p_WP - p_WB)

        slider_geometry = self.planner.config.dynamics_config.slider.geometry
        pusher_radius = self.planner.config.dynamics_config.pusher_radius

        if isinstance(mode, FaceContactMode):
            # Check distance to contact plane
            plane = slider_geometry.get_hyperplane_from_location(mode.contact_location)
            dist = plane.dist_to(p_BP)
            # We allow being slightly "inside" or "outside" the specific radius distance
            if abs(dist - pusher_radius) > tol:
                return False

            # Check projection onto face (lam in [0, 1])
            lam = slider_geometry.get_lam_from_p_BP_by_projection(p_BP, mode.contact_location)
            if lam < -tol or lam > 1.0 + tol:
                return False

            return True

        elif isinstance(mode, NonCollisionMode):
            # Firstly, check that pusher is in the collision-free region
            # Region is defined by intersection of half-spaces: a^T x >= b
            # dist_to returns a^T x - b.
            # We require dist >= 0 (point inside region).
            # We fail if dist < -tol (point strictly outside by more than tol).
            region_planes = slider_geometry.get_planes_for_collision_free_region(mode.contact_location.idx)
            for plane in region_planes:
                if plane.dist_to(p_BP) < -tol:
                    return False

            # Then, check that pusher is not in contact with the slider
            # We require dist to contact plane >= pusher_radius
            # We fail if dist < pusher_radius - tol
            contact_planes = slider_geometry.get_contact_planes(mode.contact_location.idx)
            for plane in contact_planes:
                if plane.dist_to(p_BP) < pusher_radius - tol:
                    return False

            return True

        return False

    def _get_segment_and_region_from_state(
        self,
        slider_pose: PlanarPose,
        pusher_pose: PlanarPose,
        t: float,
        enforce_monotonic_progress: bool = True,
    ) -> Tuple[int, PolytopeContactLocation]:
        """
        Finds the index of the mode in the original path that contains the current state,
        and returns the collision-free region for that mode.
        If multiple modes contain the state, uses time t to break ties.
        If no mode contains the state, fall back to the original planned mode at time t.

        If enforce_monotonic_progress is True, the function will only return a mode that is
        either the system's current mode or the one mode after the current mode.
        """
        # Determine candidate modes
        if enforce_monotonic_progress:  # Candidates only include current and next mode
            candidates = [self.current_segment_index]
            if self.current_segment_index + 1 < len(self.original_path.pairs):
                candidates.append(self.current_segment_index + 1)
        else:
            candidates = range(len(self.original_path.pairs))  # All modes are candidates

        # Find all candidate modes that the state *could* be in
        valid_indices = []
        for i in candidates:
            if self._is_state_in_mode(slider_pose, pusher_pose, self.original_path.pairs[i].mode):
                valid_indices.append(i)

        # If no mode is valid, fallback
        if len(valid_indices) == 0:
            print("WARNING: No mode is valid for the current state. Falling back to the original planned mode.")
            if enforce_monotonic_progress:
                valid_indices = list(candidates)
            segment_idx = self.original_traj._get_curr_segment_idx(t)  # type: ignore
            valid_indices = [int(segment_idx)]

        # If only one mode is valid, use it
        if len(valid_indices) == 1:
            segment_idx = valid_indices[0]

        # If multiple modes are valid, pick the one that is closest in time
        else:
            print("WARNING: Multiple modes are valid for the current state. Picking the one that is closest in time.")

            def _time_dist(idx: int) -> float:
                seg_start_time = self.original_traj.start_time if idx == 0 else self.original_traj.end_times[idx - 1]
                seg_end_time = self.original_traj.end_times[idx]

                if t < seg_start_time:
                    return seg_start_time - t
                elif t > seg_end_time:
                    return t - seg_end_time
                else:
                    return 0.0  # t is inside the segment

            segment_idx = min(valid_indices, key=_time_dist)

        self.current_segment_index = segment_idx

        # Extract the region corresponding to the mode index found above
        mode = self.original_path.pairs[segment_idx].mode
        if isinstance(mode, FaceContactMode):
            # Map face index to region index
            slider_geometry = self.planner.config.dynamics_config.slider.geometry
            region_idx = slider_geometry.get_collision_free_region_for_loc_idx(mode.contact_location.idx)
            collision_free_region = PolytopeContactLocation(ContactLocation.FACE, region_idx)
        else:
            collision_free_region = mode.contact_location

        return segment_idx, collision_free_region

    def _get_closest_time_and_segment(
        self,
        current_slider_pose: PlanarPose,
        current_pusher_pose: Optional[PlanarPose],
    ) -> Tuple[float, int]:
        """
        Finds the time t and segment index on the original trajectory that minimizes the distance to the current state.
        """
        traj = self.original_traj

        target_slider = current_slider_pose
        target_pusher = current_pusher_pose

        # Helper to compute distance
        def _dist_at_t(t_: float) -> float:
            slider_pose = traj.get_slider_planar_pose(t_)
            pusher_pose = traj.get_pusher_planar_pose(t_)

            # Position error
            pos_err = np.linalg.norm(slider_pose.pos() - target_slider.pos())

            # Angle error (handle wrapping)
            th_err = np.abs(slider_pose.theta - target_slider.theta)
            th_err = min(th_err, 2 * np.pi - th_err)

            # Pusher error
            pusher_err = 0.0
            if target_pusher is not None:
                pusher_err = np.linalg.norm(pusher_pose.pos() - target_pusher.pos())

            return pos_err + 0.5 * th_err + pusher_err

        # Coarse search over the trajectory
        NUM_STEPS = 100
        ts = np.linspace(traj.start_time, traj.end_time, NUM_STEPS)
        dists = [_dist_at_t(t) for t in ts]
        best_idx = np.argmin(dists)
        best_t = ts[best_idx]

        # Identify the segment index using the trajectory logic
        if hasattr(traj, "_get_curr_segment_idx"):
            segment_idx = traj._get_curr_segment_idx(best_t)  # type: ignore
        else:
            if best_t >= traj.end_time:
                segment_idx = len(traj.traj_segments) - 1
            else:
                segment_idx = np.where(best_t < traj.end_times)[0][0]

        return best_t, int(segment_idx)

    def _get_remaining_mode_sequence(
        self,
        segment_idx: int,
    ) -> List[str]:
        """
        Retrieves the remaining mode sequence (list of active vertices) from a PlanarPushingPath
        starting from the given segment index.
        """
        mode_sequence = [pair.vertex.name() for pair in self.original_path.pairs[segment_idx:]]

        # We must always start with the source mode for the planner to work
        if len(mode_sequence) == 0 or mode_sequence[0] != "source":
            mode_sequence.insert(0, "source")

        return mode_sequence

    def _update_initial_poses(
        self,
        current_slider_pose: PlanarPose,
        current_pusher_pose: PlanarPose,
        mode_sequence: List[str],
        current_pusher_velocity: Optional[npt.NDArray[np.float64]],
        collision_free_region: Optional[PolytopeContactLocation] = None,
    ) -> None:
        # Update config
        assert self.planner.config.start_and_goal is not None
        self.planner.config.start_and_goal.slider_initial_pose = current_slider_pose
        self.planner.config.start_and_goal.pusher_initial_pose = current_pusher_pose

        # Clean up old graph elements to prevent bloat
        # 1. Remove old source vertex
        old_source_pair = self.planner.source
        if old_source_pair is not None:
            old_vertex = old_source_pair.vertex
            old_name = old_vertex.name()

            # Remove from GCS
            self.planner.gcs.RemoveVertex(old_vertex)

            # Remove stale edges from planner.edges dictionary
            # Keys are (u_name, v_name)
            keys_to_remove = [k for k in self.planner.edges.keys() if k[0] == old_name]
            for k in keys_to_remove:
                del self.planner.edges[k]

        # 2. Remove old "short" mode vertex from previous iteration
        if self.previous_short_mode_vertex is not None:
            old_short_name = self.previous_short_mode_vertex.name()
            self.planner.gcs.RemoveVertex(self.previous_short_mode_vertex)
            # Remove stale edges connected to short mode from dictionary
            keys_to_remove = [k for k in self.planner.edges.keys() if k[0] == old_short_name or k[1] == old_short_name]
            for k in keys_to_remove:
                del self.planner.edges[k]
            # Remove from planner extra vertex mode pairs from previous MPC iteration
            if old_short_name in self.planner.extra_vertex_mode_pairs:
                del self.planner.extra_vertex_mode_pairs[old_short_name]
            self.previous_short_mode_vertex = None

        # Update planner source
        # This creates a new source vertex and connects it to the existing GCS instance
        # Then, call _set_initial_poses to set this new source vertex in the GCS
        self.planner._set_initial_poses(
            current_pusher_pose, current_slider_pose, collision_free_region=collision_free_region
        )

        if mode_sequence is not None and len(mode_sequence) >= 2:
            first_mode_name = mode_sequence[1]

            all_pairs = self.planner._get_all_vertex_mode_pairs()

            # Check if we need to replace the first mode with a custom timed one
            if self.config.time_first_mode is not None:
                if first_mode_name in all_pairs:
                    original_pair = all_pairs[first_mode_name]
                    original_mode = original_pair.mode

                    # Create new mode with custom time
                    new_mode = None
                    if isinstance(original_mode, FaceContactMode):
                        new_mode = FaceContactMode(
                            f"{first_mode_name}_SHORT",
                            self.config.num_knot_points_contact,
                            self.config.time_first_mode,
                            original_mode.contact_location,
                            self.config,
                        )
                    elif isinstance(original_mode, NonCollisionMode):
                        new_mode = NonCollisionMode(
                            f"{first_mode_name}_SHORT",
                            self.config.num_knot_points_non_collision,
                            self.config.time_first_mode,
                            original_mode.contact_location,
                            self.config,
                        )

                    if new_mode is not None:
                        # Enforce initial velocity constraint if applicable
                        if isinstance(new_mode, NonCollisionMode) and current_pusher_velocity is not None:
                            # Constrain initial velocity
                            # v_world = v_pusher_velocity  (pusher velocity in world frame)
                            # v_body = R_WB.T @ v_world  (pusher velocity in slider body frame)
                            # (slider is static during non-collision mode)
                            R_WB = current_slider_pose.two_d_rot_matrix()
                            v_body = R_WB.T @ current_pusher_velocity

                            # Velocity constraint for Bezier curve:
                            # For a Bezier curve of degree n parameterized over [0, T]:
                            #   dB/dt |_{t=0} = n/T * (P1 - P0)
                            # So to achieve initial velocity v_body:
                            #   v_body = order / time_in_mode * (c1 - c0)
                            #   c1 - c0 = v_body * time_in_mode / order
                            decision_vars = new_mode.variables
                            assert isinstance(decision_vars, NonCollisionVariables)
                            c0_x = decision_vars.p_BP_xs[0]  # 1st control point x
                            c0_y = decision_vars.p_BP_ys[0]  # 1st control point y
                            c1_x = decision_vars.p_BP_xs[1]  # 2nd control point x
                            c1_y = decision_vars.p_BP_ys[1]  # 2nd control point y

                            order = decision_vars.num_knot_points - 1
                            scale = decision_vars.time_in_mode / order
                            new_mode.prog.AddLinearEqualityConstraint(c1_x - c0_x == v_body[0] * scale)
                            new_mode.prog.AddLinearEqualityConstraint(c1_y - c0_y == v_body[1] * scale)
                            # vel_tol = 1e-1  # tolerance
                            # c_tol = vel_tol * scale
                            # new_mode.prog.AddLinearConstraint(c1_x - c0_x <= v_body[0] * scale + c_tol)
                            # new_mode.prog.AddLinearConstraint(c1_x - c0_x >= v_body[0] * scale - c_tol)
                            # new_mode.prog.AddLinearConstraint(c1_y - c0_y <= v_body[1] * scale + c_tol)
                            # new_mode.prog.AddLinearConstraint(c1_y - c0_y >= v_body[1] * scale - c_tol)

                        # Add to graph
                        new_vertex = self.planner.gcs.AddVertex(new_mode.get_convex_set(), new_mode.name)
                        new_mode.add_cost_to_vertex(new_vertex)
                        if isinstance(new_mode, NonCollisionMode):
                            new_mode.add_constraints_to_vertex(new_vertex)

                        new_pair = VertexModePair(new_vertex, new_mode)
                        self.previous_short_mode_vertex = new_vertex
                        # Register new short mode vertex in planner
                        self.planner.extra_vertex_mode_pairs[new_mode.name] = new_pair

                        # Connect Source -> New Mode
                        assert self.planner.source is not None
                        source_name = self.planner.source.vertex.name()
                        edge_key_source = (source_name, new_mode.name)
                        self.planner.edges[edge_key_source] = gcs_add_edge_with_continuity(
                            self.planner.gcs,
                            self.planner.source,
                            new_pair,
                            only_continuity_on_slider=False,
                            continuity_on_pusher_velocities=False,  # Source has no velocity
                        )

                        # Connect New Mode -> Next Mode
                        if len(mode_sequence) >= 3:
                            next_mode_name = mode_sequence[2]
                            if next_mode_name in all_pairs:
                                next_pair = all_pairs[next_mode_name]
                                edge_key_next = (new_mode.name, next_mode_name)

                                # Only enforce velocity continuity if both modes are NonCollisionMode
                                enforce_velocity = (
                                    self.config.non_collision_config.continuity_on_pusher_velocity
                                    and isinstance(new_mode, NonCollisionMode)
                                    and isinstance(next_pair.mode, NonCollisionMode)
                                    and next_pair.mode.num_knot_points > 1
                                )

                                self.planner.edges[edge_key_next] = gcs_add_edge_with_continuity(
                                    self.planner.gcs,
                                    new_pair,
                                    next_pair,
                                    only_continuity_on_slider=False,
                                    continuity_on_pusher_velocities=enforce_velocity,
                                )

                        # Update mode sequence in place so planner uses the new mode
                        mode_sequence[1] = new_mode.name
                        return

    def _update_mode_timing(self, t: float, segment_idx: int) -> None:
        """self.config.time_first_mode equal to the remaining time in the current mode"""
        traj = self.original_traj

        # If t > end_time, we are done / last segment
        if t >= traj.end_time:
            self.config.time_first_mode = 0.0
            return

        end_time = traj.end_times[segment_idx]
        remaining = end_time - t

        # Ensure a small positive minimum to avoid numerical issues
        self.config.time_first_mode = max(remaining, 1e-1)

        # If we are very close to the end of the mode, we might want to just skip to the next mode
        # This is a heuristic to avoid planning for very short durations
        if remaining < 1e-1:
            print(f"WARNING: Remaining time {remaining:.4f} < 0.1s. Increasing to 0.1s to avoid numerical issues.")

    def plan(
        self,
        t: float,
        current_slider_pose: PlanarPose,
        current_pusher_pose: PlanarPose,
        current_pusher_velocity: Optional[npt.NDArray[np.float64]],
        output_folder: str = "",
        output_name: str = "",
        save_video: bool = False,
        save_traj: bool = False,
        interpolate_video: bool = False,
        overlay_traj: bool = False,
        animation_lims: Optional[Tuple[float, float, float, float]] = None,
        hardware: bool = False,
        enforce_monotonic_progress: bool = False,
    ) -> Optional[PlanarPushingPath]:
        """
        Using current time, plan a new path from the current state and pusher velocity.
        1) Determine remaining mode sequence from the original path
        2) Set new initial state, and update source vertex (as well as edges from the source vertex) in the GCS instance
        3) Determine the remaining time for the current mode
        4) Update self.planner.config so the first mode takes the remaining time
        5) Solve the GCS convex restriction
        6) Optionally save trajectory and video outputs
        """

        # Determine best_t and segment_idx using state if available
        best_t = t
        segment_idx = None
        # We first find best_t based on the state to get a good time estimate
        # best_t, _ = self._get_closest_time_and_segment(current_slider_pose, current_pusher_pose)

        # Find segment_idx that the system is currently in by checking feasibility of state in each mode
        segment_idx, collision_free_region = self._get_segment_and_region_from_state(
            current_slider_pose,
            current_pusher_pose,
            t=best_t,  # time used to break ties or as fallbackonly
            enforce_monotonic_progress=enforce_monotonic_progress,
        )

        mode_sequence = self._get_remaining_mode_sequence(segment_idx=segment_idx)
        print(f"    Mode sequence: {mode_sequence}")

        self._update_mode_timing(t, segment_idx)
        print(f"    Remaining time in current mode: {self.config.time_first_mode}")

        self._update_initial_poses(
            current_slider_pose,
            current_pusher_pose,
            mode_sequence,
            current_pusher_velocity,
            collision_free_region=collision_free_region,
        )

        path = self.planner.plan_path(self.solver_params, active_vertices=mode_sequence, store_result=False)

        # Save outputs if requested
        if path is not None and (save_video or save_traj) and output_folder:
            os.makedirs(output_folder, exist_ok=True)

            traj = path.to_traj(rounded=True)

            if save_traj:
                trajectory_folder = f"{output_folder}/{output_name}/trajectory"
                os.makedirs(trajectory_folder, exist_ok=True)

                if traj is not None:
                    traj.save(f"{trajectory_folder}/traj.pkl")

                slider_color = COLORS["aquamarine4"].diffuse()

                if traj is not None:
                    make_traj_figure(
                        traj,
                        filename=f"{trajectory_folder}/traj",
                        slider_color=slider_color,
                        split_on_mode_type=True,
                        show_workspace=hardware,
                    )
                print(f"Saved trajectory to {trajectory_folder}")

            if save_video:
                if traj is not None:
                    # Prepare overlay trajectories if requested
                    overlay_trajs_arg = None
                    if overlay_traj:
                        # Original trajectory: black for both slider and pusher
                        original_slider_color = COLORS["black"]
                        original_pusher_color = COLORS["black"]
                        # New trajectory: use object colors (slider=aquamarine, pusher=firebrick)
                        new_slider_color = COLORS["aquamarine4"]
                        new_pusher_color = COLORS["firebrick3"]
                        overlay_trajs_arg = [
                            (self.original_traj, original_slider_color, original_pusher_color),
                            (traj, new_slider_color, new_pusher_color),
                        ]

                    visualize_planar_pushing_trajectory(
                        traj,
                        save=True,
                        filename=f"{output_folder}/{output_name}",
                        visualize_knot_points=not interpolate_video,
                        lims=animation_lims,
                        overlay_trajs=overlay_trajs_arg,
                    )
                    print(f"Saved video to {output_folder}")

        return path
