import copy
import os
from typing import List, Optional, Tuple, Union

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
    PlanarPushingTrajectory,
    StationaryPlanarPushingTrajectory,
)
from gcs_planar_pushing.planning.planar.planar_plan_config import (
    PlanarPlanConfig,
    PlanarPushingStartAndGoal,
    PlanarSolverParams,
)
from gcs_planar_pushing.planning.planar.planar_pushing_planner import PlanarPushingPlanner
from gcs_planar_pushing.planning.planar.utils import create_plan
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
        planner_freq: float = 10.0,
        double_plan: bool = False,  # Replan from scratch once in the non-collision mode after the last contact
        full_replan: bool = False,  # Replan from scratch after every contact mode
        plan: bool = True,
        output_folder: str = "",
        output_name: str = "",
        save_video: bool = False,
        interpolate_video: bool = True,
    ):
        """
        Formulate initial GCS Problem and do full solve
        """
        self.config = copy.deepcopy(
            config
        )  # Defensive copy to avoid mutating the original config which might be reused by caller
        self.config_hard_copy = copy.deepcopy(config)
        self.solver_params = solver_params
        self.double_plan = double_plan
        self.full_replan = full_replan
        self.output_folder = output_folder
        self.output_name = output_name
        self.save_video = save_video
        self.interpolate_video = interpolate_video

        self.planner = PlanarPushingPlanner(self.config)
        self.planner.config.start_and_goal = start_and_goal
        self.planner.formulate_problem()

        if plan:
            print("*" * 50 + " CREATING INITIAL PATH " + "*" * 50)
            path = create_plan(
                start_and_target=start_and_goal,
                config=config,
                solver_params=solver_params,
                planner=self.planner,
                output_folder=output_folder,
                output_name=output_name,
                save_video=save_video,
                interpolate_video=interpolate_video,
                do_rounding=True,
                save_traj=True,
                save_relaxed=True,
            )
            print(
                f"    🔍 path.relaxed_cost: {path.relaxed_cost}; path.rounded_cost: {path.rounded_cost} "
                f"(path.rounded_result.is_success(): {path.rounded_result.is_success()})"
            )
            print("*" * 123)
            assert path is not None
            self.original_path = path
            self.original_traj = path.to_traj()
            print(f"Original Path Mode Sequence: {[pair.vertex.name() for pair in self.original_path.pairs]}")
            self.traj = self.original_traj
        else:
            # The caller should also call load_original_path() if `plan` is False.
            self.original_path = None
            self.original_traj = None
            self.traj = None

        self.planner_freq = planner_freq

        # Both times below are in **episode** (simulation-clock) time.
        #
        # Episode time when `original_traj` was created.  `original_traj` is parameterised from t=0,
        # so to index into it we compute  t_episode - original_traj_start_time.
        self.original_traj_start_time = 0.0
        # Episode time when the latest `self.traj` was created (updated every MPC cycle).
        # `self.traj` is also parameterised from t=0, so to index into it we compute
        # t_episode - traj_start_time.
        self.traj_start_time = 0.0

        # Store "short" mode vertex, which is a copy of the original vertex in the GCS but with a shorter time to
        # account for the passage of time during MPC iterations.
        self.previous_short_mode_vertex = None  # Optional[GcsVertex]
        self.current_segment_index = 1  # Start at second mode to skip source node
        self.current_mode_start_time = 0.0  # t_local when the current mode actually began
        self._was_in_contact = False

        # Indicate whether we are currently executing the double plan (as opposed to the original plan).
        self._executing_double_plan = False

        # Tracks which segment_idx last triggered a full_replan so we don't re-trigger in the same mode.
        # Reset to None whenever the original path is replaced by a full replan.
        self._full_replan_last_triggered_segment_idx: Optional[int] = None

    def load_original_path(self, filename: str) -> None:
        """Loads the original path from a file."""
        all_pairs = self.planner._get_all_vertex_mode_pairs()
        path = PlanarPushingPath.load(filename, self.planner.gcs, all_pairs)
        self._reset_original_path(path, t=0.0, rounded=False)
        print(f"Original Path Mode Sequence: {[pair.vertex.name() for pair in self.original_path.pairs]}")

    def _reset_original_path(
        self,
        path: PlanarPushingPath,
        t: float,
        rounded: bool,
        precomputed_traj: Optional[PlanarPushingTrajectory] = None,
    ) -> None:
        """
        Replace the tracked original path with a freshly replanned one and reset all
        path-tracking state.  Callers are responsible for setting any replan-specific
        flags (e.g. ``_executing_double_plan``, ``_full_replan_last_triggered_segment_idx``)
        after this call.

        ``precomputed_traj`` may be supplied when the caller has already called
        ``path.to_traj()`` before GCS vertices were removed (e.g. in the full_replan
        two-phase flow where Phase 2 deletes the Phase 1 source vertex before this
        function is reached).  When provided, it is used directly instead of calling
        ``path.to_traj(rounded)`` here.
        """
        self.original_path = path
        self.original_traj = precomputed_traj if precomputed_traj is not None else path.to_traj(rounded=rounded)
        self.traj = self.original_traj
        self.original_traj_start_time = t
        self.traj_start_time = t
        self.current_segment_index = 1
        self.current_mode_start_time = 0.0
        self._was_in_contact = False

    def _is_state_in_mode(
        self,
        slider_pose: PlanarPose,
        pusher_pose: PlanarPose,
        mode: AbstractContactMode,
        coll_tol: float = 1e-3,
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
            if abs(dist - pusher_radius) > coll_tol:
                return False

            # Check projection onto face (lam in [0, 1])
            lam = slider_geometry.get_lam_from_p_BP_by_projection(p_BP, mode.contact_location)
            if lam < -coll_tol or lam > 1.0 + coll_tol:
                return False

            return True

        elif isinstance(mode, NonCollisionMode):
            # Use the same inflation as the mode's convex set so region assignment is consistent
            # between planning time and runtime.
            non_coll_tol = self.planner.config.non_collision_mode_region_inflation

            # Firstly, check that pusher is in the collision-free region
            # Region is defined by intersection of half-spaces: a^T x >= b
            # dist_to returns a^T x - b.
            # We require dist >= 0 (point inside region).
            # We fail if dist < -tol (point strictly outside by more than tol).
            region_planes = slider_geometry.get_planes_for_collision_free_region(mode.contact_location.idx)
            for plane in region_planes:
                if plane.dist_to(p_BP) < -non_coll_tol:
                    return False

            # Then, check that pusher is not in contact with the slider
            # We require dist to contact plane >= pusher_radius
            # We fail if dist < pusher_radius - tol
            contact_planes = slider_geometry.get_contact_planes(mode.contact_location.idx)
            for plane in contact_planes:
                if plane.dist_to(p_BP) < pusher_radius - non_coll_tol:
                    return False

            return True

        return False

    def _get_segment_and_region_from_state(
        self,
        slider_pose: PlanarPose,
        pusher_pose: PlanarPose,
        t: float,
        is_in_contact: bool,
        enforce_monotonic_progress: bool = True,
    ) -> Tuple[int, PolytopeContactLocation]:
        """
        Finds the index of the mode in the original path that contains the current state,
        and returns the collision-free region for that mode.
        """
        # Determine candidate modes
        if enforce_monotonic_progress:  # Candidates only include current and next mode
            candidates = [self.current_segment_index]
            if self.current_segment_index + 1 < len(self.original_path.pairs):
                candidates.append(self.current_segment_index + 1)
        else:
            candidates = list(range(len(self.original_path.pairs)))  # All modes are candidates

        # If there is contact
        if is_in_contact:
            current_mode = self.original_path.pairs[self.current_segment_index].mode
            # If we are currently in a non-collision mode:
            if isinstance(current_mode, NonCollisionMode):
                next_idx = self.current_segment_index + 1
                # Either, we are transitioning into a contact mode:
                if next_idx in candidates and isinstance(self.original_path.pairs[next_idx].mode, FaceContactMode):
                    segment_idx = next_idx
                # Or, we have just forced a transitioned out of a contact mode and are now in a non-collision mode, but
                # there is still residual contact. We stay in the current non-collision mode.
                else:
                    segment_idx = self.current_segment_index
            # If we are currently in a contact mode:
            elif isinstance(current_mode, FaceContactMode):
                remaining_time = self._get_remaining_time_in_current_mode(t)
                next_idx = self.current_segment_index + 1

                # If there is no time left in the contact mode, force a transition to the next mode
                if (
                    remaining_time <= 1e-3
                    and next_idx in candidates
                    and isinstance(self.original_path.pairs[next_idx].mode, NonCollisionMode)
                ):
                    print(
                        f"ℹ️ Forcing transition to next mode bc remaining t in contact mode too small: {remaining_time}"
                    )
                    segment_idx = next_idx
                # Otherwise, continue in the contact mode.
                else:
                    valid_indices = [
                        i for i in candidates if isinstance(self.original_path.pairs[i].mode, FaceContactMode)
                    ]
                    assert len(valid_indices) == 1, (
                        f"Expected one candidate mode when in contact, but got {len(valid_indices)}"
                    )
                    segment_idx = valid_indices[0]
        # If there is no contact, there could be one or two (if there are two consecutive non-collision modes) candidate
        # modes. We need to do an actual check based on the system state.
        else:
            candidates = [i for i in candidates if isinstance(self.original_path.pairs[i].mode, NonCollisionMode)]

            # Find all candidate modes that the state *could* be in
            valid_indices = [
                i
                for i in candidates
                if self._is_state_in_mode(slider_pose, pusher_pose, self.original_path.pairs[i].mode)
            ]
            print(f"candidates: {candidates} | valid_indices: {valid_indices}")

            if len(valid_indices) == 0:
                # raise ValueError(f"No mode is valid for the current state somehow. Candidates: {candidates}.")
                if len(candidates) == 1:
                    print(
                        f"⚠️ WARNING: No mode is valid for the current state somehow. Candidates: {candidates}."
                        "Picking the only candidate mode."
                    )
                    segment_idx = candidates[0]
                else:
                    print(
                        f"⚠️ WARNING: No mode is valid for the current state somehow. Candidates: {candidates}."
                        "Picking the second candidate mode."
                    )
                    # In the case that this happens at the transition between two non-collision modes, we pick the 2nd
                    segment_idx = candidates[1]

            # If only one mode is valid, use it
            elif len(valid_indices) == 1:
                segment_idx = valid_indices[0]
            # If multiple modes are valid, pick the one that is closest in time
            else:
                print("⚠️ WARNING: Multiple modes are valid for the current state.")

                # Shift planned segment boundaries by accumulated schedule drift so that
                # early/late mode transitions are reflected in the tiebreaker.
                planned_start_of_current = self.original_traj.start_times[self.current_segment_index]
                schedule_drift = self.current_mode_start_time - planned_start_of_current

                def _time_dist(idx: int) -> float:
                    seg_start_time = self.original_traj.start_times[idx] + schedule_drift
                    seg_end_time = self.original_traj.end_times[idx] + schedule_drift

                    if t < seg_start_time:
                        return seg_start_time - t
                    elif t > seg_end_time:
                        return t - seg_end_time
                    else:
                        return 0.0  # t is inside the segment

                segment_idx = min(valid_indices, key=_time_dist)

        # Record actual start time of the new mode
        if segment_idx != self.current_segment_index:
            self.current_mode_start_time = t
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

        # Ensure the last mode is "target" (if it was the target in the original path)
        # The original path's last element is always the target.
        if len(mode_sequence) > 0:
            mode_sequence[-1] = "target"

        return mode_sequence

    def _update_initial_poses(
        self,
        current_slider_pose: PlanarPose,
        current_pusher_pose: PlanarPose,
        mode_sequence: List[str],
        current_pusher_velocity: Optional[npt.NDArray[np.float64]],
        collision_free_region: Optional[PolytopeContactLocation] = None,
        soft_source_node_pose_constraint: bool = False,
        enforce_velocity_constraint: bool = True,
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
            current_pusher_pose,
            current_slider_pose,
            collision_free_region=collision_free_region,
            soft_source_node_pose_constraint=soft_source_node_pose_constraint,
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
                        # Enforce initial velocity constraint only if previous re-plan cycle was NOT in contact; coming
                        # out of contact, velocity points into the slider --> pushes 2nd control point into collision.
                        if (
                            enforce_velocity_constraint
                            and current_pusher_velocity is not None
                            and not self._was_in_contact
                            and isinstance(new_mode, NonCollisionMode)
                        ):
                            R_WB = current_slider_pose.two_d_rot_matrix()
                            v_WP_B = R_WB.T @ current_pusher_velocity

                            decision_vars = new_mode.variables
                            assert isinstance(decision_vars, NonCollisionVariables)
                            c0_x = decision_vars.p_BP_xs[0]
                            c0_y = decision_vars.p_BP_ys[0]
                            c1_x = decision_vars.p_BP_xs[1]
                            c1_y = decision_vars.p_BP_ys[1]

                            order = decision_vars.num_knot_points - 1
                            scale = decision_vars.time_in_mode / order
                            new_mode.prog.AddLinearEqualityConstraint(c1_x - c0_x == v_WP_B[0] * scale)
                            new_mode.prog.AddLinearEqualityConstraint(c1_y - c0_y == v_WP_B[1] * scale)
                        # We do not enforce velocity constraints for face contact modes because face contact modes are
                        # parameterized by piecewise first-order-hold curves which are internally discontinuous anyway.

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

    def _rebuild_target_node(
        self,
        target_slider_pose: PlanarPose,
        last_mode_name: Optional[str] = None,
        soft_slider_target_constraint: bool = True,
    ) -> None:
        """
        This function serves two purposes:
        1. Rebuild the target vertex so its body-frame pusher position gives the
        correct world-frame pusher position for the given slider pose.
        2. Rebuild the target vertex so it has soft constraints on slider target pose.
        """
        goal = self.planner.config.start_and_goal

        # Remove the old target vertex and its edges
        old_target = self.planner.target
        old_name = old_target.vertex.name()
        self.planner.gcs.RemoveVertex(old_target.vertex)
        for k in [k for k in self.planner.edges if k[1] == old_name]:
            del self.planner.edges[k]

        self.planner._set_target_poses(
            goal.pusher_target_pose, target_slider_pose, soft_slider_target_constraint=soft_slider_target_constraint
        )

        # Connect the last mode in the active path to the new target
        if last_mode_name is not None:
            all_pairs = self.planner._get_all_vertex_mode_pairs()
            last_pair = all_pairs[last_mode_name]
            target_name = self.planner.target.vertex.name()
            self.planner.edges[(last_mode_name, target_name)] = gcs_add_edge_with_continuity(
                self.planner.gcs,
                last_pair,
                self.planner.target,
                only_continuity_on_slider=False,
                continuity_on_pusher_velocities=False,
            )

        self.original_path.pairs[-1] = self.planner.target

    def _get_remaining_time_in_current_mode(self, t: float) -> float:
        """
        Gets the remaining time in the current mode.
        `t` is in the time parameterization of `original_traj` (starts at 0).
        Time is measured relative to when the mode actually started, not the
        globally planned start time, so early mode transitions don't bleed
        extra time into the next mode.
        """
        idx = self.current_segment_index
        planned_duration = self.original_traj.end_times[idx] - self.original_traj.start_times[idx]
        elapsed = t - self.current_mode_start_time
        return planned_duration - elapsed

    def _get_elapsed_time_in_current_mode(self, t: float) -> float:
        """
        Gets the elapsed time since the start of the current mode.
        `t` is in the time parameterization of `original_traj` (starts at 0).
        Time is measured relative to when the mode actually started.
        """
        return t - self.current_mode_start_time

    def _update_mode_timing(self, t: float, segment_idx: int) -> None:
        """
        Updates the timing of the first mode in the mode sequence.
        self.config.time_first_mode is set to the remaining time in the current mode.
        """
        remaining = self._get_remaining_time_in_current_mode(t)
        self.config.time_first_mode = max(remaining, 1e-1)
        if remaining < 1e-1:
            print(f"⚠️ WARNING: Remaining time {remaining:.4f} < 0.1s. Increasing to 0.1s to avoid numerical issues.")

    def plan(
        self,
        t: float,
        current_slider_pose: PlanarPose,
        current_pusher_pose: PlanarPose,
        current_pusher_velocity: Optional[npt.NDArray[np.float64]],
        is_in_contact: bool = False,
        output_folder: str = "",
        output_name: str = "",
        save_video: bool = False,
        save_traj: bool = False,
        save_unrounded_video: bool = False,
        interpolate_video: bool = True,
        overlay_traj: bool = True,
        animation_lims: Optional[Tuple[float, float, float, float]] = None,
        hardware: bool = False,
        rounded: bool = True,
        success: bool = False,  # Whether simulator detects success
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
        assert not (self.double_plan and self.full_replan), (
            "double_plan and full_replan cannot both be enabled simultaneously."
        )

        # Force rounded solutions after a double_plan (which always replans with rounding).
        if self._executing_double_plan:
            rounded = True

        # `t` stays as episode time throughout; `t_local` is original_traj-relative (starts at 0).
        t_local = t - self.original_traj_start_time

        # Find segment_idx that the system is currently in by checking feasibility of state in each mode
        segment_idx, collision_free_region = self._get_segment_and_region_from_state(
            current_slider_pose,
            current_pusher_pose,
            t=t_local,  # time used to break ties or as fallback only
            is_in_contact=is_in_contact,
        )

        # If the plan has completed (no time left in the final mode), return a stationary traj at current state.
        if (
            segment_idx == len(self.original_path.pairs) - 1
            and self._get_remaining_time_in_current_mode(t_local) <= 1e-3
        ):
            return StationaryPlanarPushingTrajectory(self.config, current_slider_pose, current_pusher_pose, 1), 0.0

        # Retrieve previous, current, and next modes relative to the active segment
        prev_mode = self.original_path.pairs[segment_idx - 1].mode if segment_idx > 0 else None
        current_mode = self.original_path.pairs[segment_idx].mode
        next_mode = (
            self.original_path.pairs[segment_idx + 1].mode if segment_idx + 1 < len(self.original_path.pairs) else None
        )

        ################################################################################################################
        ### Double-plan trigger: replan from scratch once the pusher is 0.3s into the first non-collision mode that
        ### follows the last contact mode. We intentionally allow the pusher to leave contact before initiating the
        ### double plan so the pusher gains some velocity and is incentivized to actually move and do a human-looking
        ### correction (rather than doing some unrealistic-looking friction hacking).
        DOUBLE_PLAN_DELAY = 0.1  # seconds into the post-contact non-collision mode
        if not success and self.double_plan and isinstance(current_mode, NonCollisionMode):
            # Check whether every remaining mode (after the current one) is non-collision, meaning
            # we have already left the last contact mode.
            all_remaining_non_collision = not any(
                isinstance(self.original_path.pairs[i].mode, FaceContactMode)
                for i in range(segment_idx, len(self.original_path.pairs))
            )
            if all_remaining_non_collision:
                elapsed_in_mode = self._get_elapsed_time_in_current_mode(t_local)
                if elapsed_in_mode >= DOUBLE_PLAN_DELAY - 1e-3:
                    print(
                        f"ℹ️ Double plan triggered: {elapsed_in_mode:.3f}s into "
                        "post-contact non-collision mode. Replanning from scratch..."
                    )
                    self.double_plan = False

                    # Tighten soft target tolerances so the replanned trajectory doesn't "give up early".
                    cfg = self.planner.config
                    cfg.soft_slider_target_eps_pos = cfg.double_plan_soft_slider_target_eps_pos
                    cfg.soft_slider_target_eps_ang = cfg.double_plan_soft_slider_target_eps_ang

                    self.previous_short_mode_vertex = None
                    self.planner.config.start_and_goal.slider_initial_pose = current_slider_pose
                    self.planner.config.start_and_goal.pusher_initial_pose = current_pusher_pose

                    # Optionally override time / costs for the double-plan.
                    if cfg.double_plan_time_in_contact is not None:
                        cfg.time_in_contact = cfg.double_plan_time_in_contact
                    if cfg.double_plan_contact_cost is not None:
                        cfg.contact_config.cost = cfg.double_plan_contact_cost
                    if cfg.double_plan_non_collision_cost is not None:
                        cfg.non_collision_cost = cfg.double_plan_non_collision_cost

                    self.planner.formulate_problem()
                    self._update_initial_poses(
                        current_slider_pose,
                        current_pusher_pose,
                        mode_sequence=[],
                        current_pusher_velocity=current_pusher_velocity,
                        collision_free_region=collision_free_region,
                        soft_source_node_pose_constraint=False,
                    )
                    # FORCE DOUBLE PLAN TO ROUND SOLUTION
                    path = self.planner.plan_path(self.solver_params, store_result=False, rounded=True)

                    if path is not None:
                        self._reset_original_path(path, t, rounded=True)
                        self._executing_double_plan = True
                        print(
                            "ℹ️ Double plan successful. New mode sequence: "
                            f"{[p.vertex.name() for p in self.original_path.pairs]}"
                        )
                        return self.traj, path.rounded_cost if rounded else path.relaxed_cost
                    else:
                        print("❌ Double plan failed. Continuing with original plan.")
        ################################################################################################################

        ################################################################################################################
        ### Full-replan trigger: replan from scratch 0.1s into every non-collision mode that immediately follows a
        ### contact mode. This generalises the double-plan strategy to all contact modes, not just the last one.
        FULL_REPLAN_DELAY = 0.1  # seconds into the post-contact non-collision mode
        if not success and self.full_replan and isinstance(current_mode, NonCollisionMode):
            if isinstance(prev_mode, FaceContactMode) and segment_idx != self._full_replan_last_triggered_segment_idx:
                elapsed_in_mode = self._get_elapsed_time_in_current_mode(t_local)
                if elapsed_in_mode >= FULL_REPLAN_DELAY - 1e-3:
                    # Use rounding only when the slider is already close to the target pose,
                    # since that indicates this is likely the final corrective replan.
                    cfg = self.config
                    pos_err = np.linalg.norm(current_slider_pose.pos() - cfg.start_and_goal.slider_target_pose.pos())
                    ang_err = abs(current_slider_pose.theta - cfg.start_and_goal.slider_target_pose.theta)
                    # Use hard-copied config rather than the possibly mutated self.config when checking if replan
                    # is necessary
                    replan_rounded = (
                        pos_err <= 2.0 * self.config_hard_copy.soft_slider_target_eps_pos
                        and ang_err <= 2.0 * self.config_hard_copy.soft_slider_target_eps_ang
                    )
                    print(
                        f"ℹ️ Full replan triggered: {elapsed_in_mode:.3f}s into post-contact non-collision mode "
                        f"(rounded={replan_rounded}, pos_err={pos_err:.4f}, ang_err={ang_err:.4f}). "
                        "Replanning from scratch..."
                    )
                    # Mark this segment so we don't retrigger on the next MPC tick.
                    self._full_replan_last_triggered_segment_idx = segment_idx

                    # When this is the final corrective replan, apply the same tightened tolerances
                    # and cost/timing overrides used by the double-plan.
                    if replan_rounded:
                        cfg.soft_slider_target_eps_pos = cfg.double_plan_soft_slider_target_eps_pos
                        cfg.soft_slider_target_eps_ang = cfg.double_plan_soft_slider_target_eps_ang
                        if cfg.double_plan_time_in_contact is not None:
                            cfg.time_in_contact = cfg.double_plan_time_in_contact
                        if cfg.double_plan_contact_cost is not None:
                            cfg.contact_config.cost = cfg.double_plan_contact_cost
                        if cfg.double_plan_non_collision_cost is not None:
                            cfg.non_collision_cost = cfg.double_plan_non_collision_cost

                    self.previous_short_mode_vertex = None
                    self.planner.config.start_and_goal.slider_initial_pose = current_slider_pose
                    self.planner.config.start_and_goal.pusher_initial_pose = current_pusher_pose

                    self.planner.formulate_problem()

                    # Phase 1: free solve (no mode sequence constraint) to discover the optimal mode sequence.
                    self._update_initial_poses(
                        current_slider_pose,
                        current_pusher_pose,
                        mode_sequence=[],
                        current_pusher_velocity=current_pusher_velocity,
                        collision_free_region=collision_free_region,
                        soft_source_node_pose_constraint=False,
                    )
                    path_p1 = self.planner.plan_path(self.solver_params, store_result=False, rounded=False)
                    path = path_p1  # default; may be overridden by Phase 2 for cost reporting

                    # Pre-compute Phase 1's trajectory NOW, while its source vertex is still alive.
                    # Phase 2's _update_initial_poses will delete the Phase 1 source vertex before
                    # _reset_original_path is reached, making path_p1.to_traj() crash there.
                    path_p1_traj = path_p1.to_traj(rounded=False) if path_p1 is not None else None

                    if path_p1 is not None and current_pusher_velocity is not None:
                        # Phase 2: re-solve along the discovered mode sequence with pusher velocity
                        # continuity enforced on the first mode. We are FULL_REPLAN_DELAY seconds into
                        # the non-collision mode so the velocity no longer points into the slider,
                        # making the constraint safe (unlike at the instant of contact release).
                        mode_sequence_p2 = [pair.vertex.name() for pair in path_p1.pairs]
                        self.config.time_first_mode = path_p1.pairs[1].mode.time_in_mode
                        self._was_in_contact = False  # Bypass contact-release guard; velocity is safe to enforce here
                        self._update_initial_poses(
                            current_slider_pose,
                            current_pusher_pose,
                            mode_sequence=mode_sequence_p2,
                            current_pusher_velocity=current_pusher_velocity,
                            collision_free_region=collision_free_region,
                            soft_source_node_pose_constraint=False,
                            enforce_velocity_constraint=True,
                        )
                        path_p2 = self.planner.plan_path(
                            self.solver_params,
                            active_vertices=mode_sequence_p2,
                            store_result=False,
                            rounded=replan_rounded,
                        )
                        if path_p2 is not None:
                            path = path_p2
                        else:
                            print("⚠️ Full replan phase 2 (velocity constraint) failed; using phase-1 solution.")
                            self.config.time_first_mode = None

                    if path is not None:
                        # Always store original_path from Phase 1 so it contains canonical (non-SHORT) mode
                        # names. The SHORT vertex created in Phase 2 is a temporary MPC construct — it is
                        # deleted at the start of the next _update_initial_poses call, so storing it in
                        # original_path would cause a KeyError when the next regular MPC iteration looks up
                        # the edge ('source', '..._SHORT'). Pass the pre-computed trajectory so that
                        # _reset_original_path does not call path_p1.to_traj() here, where the Phase 1
                        # source vertex has already been deleted by Phase 2's _update_initial_poses.
                        # If Phase 2 succeeded, override original_traj/traj with the velocity-continuous
                        # result; the timing structure is identical since both phases use the same mode durations.
                        self._reset_original_path(path_p1, t, rounded=False, precomputed_traj=path_p1_traj)
                        if path is not path_p1:
                            self.original_traj = path.to_traj(rounded=replan_rounded)
                            self.traj = self.original_traj
                        # Clear the trigger guard so the new path's post-contact modes can fire.
                        self._full_replan_last_triggered_segment_idx = None
                        print(
                            "ℹ️ Full replan successful. New mode sequence: "
                            f"{[p.vertex.name() for p in self.original_path.pairs[1:]]}"
                        )
                        return self.traj, path.rounded_cost if replan_rounded else path.relaxed_cost
                    else:
                        print("❌ Full replan failed. Continuing with original plan.")
        ################################################################################################################

        ################################################################################################################
        ### THIS IS A HACKY (NON-MARKOVIAN) FIX:
        ### If we are in the last 0.3s of the final contact mode, run the latter part of this mode open-loop to prevent
        ### the optimization from creating crazy results.
        if is_in_contact:
            if isinstance(current_mode, FaceContactMode):
                remaining_time = self._get_remaining_time_in_current_mode(t_local)
                is_last_contact_mode = not any(
                    isinstance(self.original_path.pairs[i].mode, FaceContactMode)
                    for i in range(segment_idx + 1, len(self.original_path.pairs))
                )
                if is_last_contact_mode and remaining_time <= 0.3 + 1e-3:
                    print(
                        f"ℹ️ Remaining time in last contact mode {remaining_time:.4f} <= 0.3s. "
                        "Returning slice of previous traj."
                    )
                    self._was_in_contact = True
                    return self.traj.get_slice(t - self.traj_start_time), 0.0
        ################################################################################################################

        mode_sequence = self._get_remaining_mode_sequence(segment_idx=segment_idx)
        print(f"    ⚙️ Mode sequence: {mode_sequence}")

        self._update_mode_timing(t_local, segment_idx)
        print(f"    ⏰ Current mode remaining t: {self.config.time_first_mode}")

        ################################################################################################################
        ### THIS IS A HACKY (NON-MARKOVIAN) FIX:
        ### If no remaining contact modes, rebuild target so world-frame pusher position is correct
        has_remaining_contact = any(
            isinstance(self.original_path.pairs[i].mode, FaceContactMode)
            for i in range(segment_idx, len(self.original_path.pairs))
        )
        last_mode_name = mode_sequence[-2]
        if not has_remaining_contact:
            self._rebuild_target_node(current_slider_pose, last_mode_name, soft_slider_target_constraint=True)
        else:
            # Rebuild target node with soft constraints for future re-plans
            self._rebuild_target_node(
                self.config.start_and_goal.slider_target_pose, last_mode_name, soft_slider_target_constraint=True
            )
        ################################################################################################################

        self._update_initial_poses(
            current_slider_pose,
            current_pusher_pose,
            mode_sequence,
            current_pusher_velocity,
            collision_free_region=collision_free_region,
            soft_source_node_pose_constraint=True,
            enforce_velocity_constraint=True,
        )

        path = self.planner.plan_path(
            self.solver_params, active_vertices=mode_sequence, store_result=False, rounded=rounded
        )
        if path is None:
            return None, None
        print(
            f"    🔍 path.relaxed_cost: {path.relaxed_cost}; path.rounded_cost: {path.rounded_cost} "
            f"(path.rounded_result.is_success(): {path.rounded_result.is_success()})"
        )
        self.traj = path.to_traj(rounded=rounded)
        self.traj_start_time = t

        rounded_traj = path.to_traj(rounded=True)
        unrounded_traj = path.to_traj(rounded=False)
        self._save_outputs(
            rounded_traj=rounded_traj,
            unrounded_traj=unrounded_traj,
            output_folder=output_folder,
            output_name=output_name,
            save_video=save_video,
            save_traj=save_traj,
            save_unrounded_video=save_unrounded_video,
            interpolate_video=interpolate_video,
            overlay_traj=overlay_traj,
            animation_lims=animation_lims,
            hardware=hardware,
        )

        self._was_in_contact = is_in_contact
        return self.traj, path.rounded_cost if rounded else path.relaxed_cost

    def _save_outputs(
        self,
        rounded_traj: PlanarPushingTrajectory,
        unrounded_traj: PlanarPushingTrajectory,
        output_folder: str,
        output_name: str,
        save_video: bool,
        save_traj: bool,
        save_unrounded_video: bool,
        interpolate_video: bool,
        overlay_traj: bool,
        animation_lims: Optional[Tuple[float, float, float, float]],
        hardware: bool,
    ) -> None:
        if (
            (rounded_traj is None and save_video)
            or (unrounded_traj is None and save_unrounded_video)
            or not (save_video or save_traj or save_unrounded_video)
            or not output_folder
        ):
            return

        os.makedirs(output_folder, exist_ok=True)

        if save_traj:
            trajectory_folder = f"{output_folder}/{output_name}/trajectory"
            os.makedirs(trajectory_folder, exist_ok=True)

            if rounded_traj is not None:
                rounded_traj.save(f"{trajectory_folder}/traj.pkl")

            slider_color = COLORS["aquamarine4"].diffuse()

            if rounded_traj is not None:
                make_traj_figure(
                    rounded_traj,
                    filename=f"{trajectory_folder}/traj",
                    slider_color=slider_color,
                    split_on_mode_type=True,
                    show_workspace=hardware,
                )
            print(f"Saved trajectory to {trajectory_folder}")

        if save_video or save_unrounded_video:
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
                    (rounded_traj, new_slider_color, new_pusher_color),
                ]

            if save_video:
                if rounded_traj is not None:
                    os.makedirs(f"{output_folder}", exist_ok=True)

                    visualize_planar_pushing_trajectory(
                        rounded_traj,
                        save=True,
                        filename=f"{output_folder}/{output_name}",
                        visualize_knot_points=not interpolate_video,
                        lims=animation_lims,
                        overlay_trajs=overlay_trajs_arg,
                        fast_save=True,
                    )
                    print(f"Saved video to {output_folder}/{output_name}")

            if save_unrounded_video:
                if unrounded_traj is not None:
                    os.makedirs(f"{output_folder}", exist_ok=True)

                    # Create overlay using the unrounded trajectory so it matches the video
                    unrounded_overlay_trajs_arg = None
                    if overlay_traj:
                        unrounded_overlay_trajs_arg = [
                            (self.original_traj, COLORS["black"], COLORS["black"]),
                            (unrounded_traj, COLORS["aquamarine4"], COLORS["firebrick3"]),
                        ]

                    visualize_planar_pushing_trajectory(
                        unrounded_traj,
                        save=True,
                        filename=f"{output_folder}/{output_name}_unrounded",
                        visualize_knot_points=not interpolate_video,
                        lims=animation_lims,
                        overlay_trajs=unrounded_overlay_trajs_arg,
                        fast_save=True,
                    )
                    print(f"Saved unrounded video to {output_folder}/{output_name}_unrounded")
