"""
This environment is built on HighwayEnv with one main road and one merging lane.
Dong Chen: chendon9@msu.edu
Date: 01/05/2021
"""
import numpy as np
from gym.envs.registration import register
from typing import Tuple

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.envs.common.risk_assessment import FDPFRiskField
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    n_a = 5
    n_s = 25

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy"],
                "flatten": True,
                "normalize": True,
                "absolute": False
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True
            },
            "controlled_vehicles": 1,
            "lanes_count": 3,
            "ramp_lanes_count": 2,
            "screen_width": 1000,
            "screen_height": 240,
            "centering_position": [0.3, 0.5],
            "scaling": 3,
            "simulation_frequency": 15,
            "duration": 20,
            "policy_frequency": 5,
            "reward_speed_range": [10, 30],
            "COLLISION_REWARD": 200,
            "HIGH_SPEED_REWARD": 2,
            "HEADWAY_COST": 2,
            "HEADWAY_TIME": 1.2,
            "MERGING_LANE_COST": 6,
            "LANE_CHANGE_COST": 0.2,
            "MERGE_SUCCESS_REWARD": 3,
            "traffic_density": 1,
            # Merge-end protection: force safer merge behavior before ramp-end obstacle.
            "merge_guard_enabled": True,
            "merge_guard_distance": 70.0,
            "merge_guard_hard_distance": 35.0,
            # FDPF-based risk perception for all controlled vehicles.
            "risk_assessment_enabled": True,
            "risk_upsilon_coeff_s": 0.3,
            "risk_upsilon_coeff_l": 0.7,
            "risk_major_scale": 2.0,
            "risk_semi_scale": 0.15,
            "risk_decay_coeff": 0.5,
            "risk_lane_margin": 0.8,
            "risk_look_back": 30.0,
            "risk_look_ahead": 40.0,
            "risk_lateral_scope_lanes": 2.0,
        })
        return config

    def _reward(self, action: list) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])

        # 末端合流车道惩罚
        merging_lane_cost = 0
        lane_idx = getattr(vehicle, "lane_index", None)
        if lane_idx is not None and len(lane_idx) >= 3 and lane_idx[0] == "b" and lane_idx[1] == "c":
            try:
                lane_obj = self.road.network.get_lane(lane_idx)
                if getattr(lane_obj, "forbidden", False):
                    merging_lane_cost = -np.exp(
                        -(vehicle.position[0] - sum(self.ends[:3])) ** 2 / (10 * self.ends[2])
                    )
            except Exception:
                merging_lane_cost = 0

    # 车头时距惩罚
        headway_distance = self._compute_headway_distance(vehicle)
        headway_cost = np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed)
        ) if vehicle.speed > 0 else 0

    # 新增：变道惩罚
        lane_change_penalty = 0.0
        if action in [0, 2]:   # 0=LANE_LEFT, 2=LANE_RIGHT
            lane_change_penalty = -self.config["LANE_CHANGE_COST"]

        reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
                 + self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1) \
                 + self.config["MERGING_LANE_COST"] * merging_lane_cost \
                 + self.config["HEADWAY_COST"] * (headway_cost if headway_cost < 0 else 0) \
                 + lane_change_penalty

        return reward

    def _regional_reward(self):
        for vehicle in self.controlled_vehicles:
            neighbor_vehicle = []
            added = set()

            def _append_if_valid(v):
                if v is None or not isinstance(v, MDPVehicle):
                    return
                vid = id(v)
                if vid in added:
                    return
                added.add(vid)
                neighbor_vehicle.append(v)

            # current lane neighbors
            v_front, v_rear = self.road.surrounding_vehicles(vehicle, vehicle.lane_index)
            _append_if_valid(v_front)
            _append_if_valid(v_rear)
            _append_if_valid(vehicle)

            # side-lane neighbors
            for side_lane in self.road.network.side_lanes(vehicle.lane_index):
                s_front, s_rear = self.road.surrounding_vehicles(vehicle, side_lane)
                _append_if_valid(s_front)
                _append_if_valid(s_rear)

            if neighbor_vehicle:
                regional_reward = sum(v.local_reward for v in neighbor_vehicle)
                vehicle.regional_reward = regional_reward / len(neighbor_vehicle)
            else:
                vehicle.regional_reward = 0.0

    def _is_vehicle_on_forbidden_lane(self, vehicle: Vehicle) -> bool:
        lane_idx = getattr(vehicle, "lane_index", None)
        if lane_idx is None:
            return False
        try:
            lane_obj = self.road.network.get_lane(lane_idx)
            return bool(getattr(lane_obj, "forbidden", False))
        except Exception:
            return False

    def _has_exited_scene(self, vehicle: Vehicle) -> bool:
        return bool(vehicle.position[0] >= sum(self.ends))

    def _is_scenario_success(self) -> bool:
        if not self.controlled_vehicles:
            return False
        if any(vehicle.crashed for vehicle in self.controlled_vehicles):
            return False

        ramp_started = getattr(self, "controlled_vehicles_on_ramp", [])
        if ramp_started and any(ramp_started):
            for i, vehicle in enumerate(self.controlled_vehicles):
                if i < len(ramp_started) and ramp_started[i]:
                    if (not self._has_exited_scene(vehicle)) or self._is_vehicle_on_forbidden_lane(vehicle):
                        return False
            return True

        return all(self._has_exited_scene(vehicle) for vehicle in self.controlled_vehicles)

    def _coerce_joint_action(self, action):
        if isinstance(action, np.ndarray):
            if action.ndim == 0:
                return [int(action.item())], False
            return [int(a) for a in action.reshape(-1).tolist()], True
        if isinstance(action, (list, tuple)):
            return [int(a) for a in action], True
        return [int(action)], False

    def _format_joint_action_like_input(self, joint_actions, input_was_sequence):
        if input_was_sequence or len(self.controlled_vehicles) > 1:
            return tuple(int(a) for a in joint_actions)
        return int(joint_actions[0]) if joint_actions else 1

    def _remaining_merge_distance(self, vehicle: Vehicle) -> float:
        if not self._is_vehicle_on_forbidden_lane(vehicle):
            return np.inf
        try:
            x = float(vehicle.position[0])
            merge_end_x = float(sum(self.ends[:3]))
        except Exception:
            return np.inf
        return max(0.0, merge_end_x - x)

    def _apply_merge_guard(self, action):
        if not bool(self.config.get("merge_guard_enabled", True)):
            return action, {}
        if not self.controlled_vehicles:
            return action, {}

        joint_actions, input_was_sequence = self._coerce_joint_action(action)
        n_agents = len(self.controlled_vehicles)
        if len(joint_actions) < n_agents:
            joint_actions.extend([1] * (n_agents - len(joint_actions)))
        elif len(joint_actions) > n_agents:
            joint_actions = joint_actions[:n_agents]

        merge_guard_distance = float(self.config.get("merge_guard_distance", 70.0))
        merge_guard_hard_distance = float(self.config.get("merge_guard_hard_distance", 35.0))

        overrides = 0
        for idx, vehicle in enumerate(self.controlled_vehicles):
            if getattr(vehicle, "crashed", False):
                continue
            if not self._is_vehicle_on_forbidden_lane(vehicle):
                continue

            remaining_merge = self._remaining_merge_distance(vehicle)
            if not np.isfinite(remaining_merge) or remaining_merge > merge_guard_distance:
                continue

            try:
                available = self._get_available_actions(vehicle, self)
                available_actions = {int(a) for a in available}
            except Exception:
                available_actions = set()

            old_action = int(joint_actions[idx])
            new_action = old_action
            if 0 in available_actions:  # LANE_LEFT
                new_action = 0
            elif remaining_merge <= merge_guard_hard_distance and 4 in available_actions:  # SLOWER
                new_action = 4
            elif remaining_merge <= merge_guard_hard_distance and 1 in available_actions:  # IDLE
                new_action = 1

            if new_action != old_action:
                joint_actions[idx] = new_action
                overrides += 1

        guarded_action = self._format_joint_action_like_input(joint_actions, input_was_sequence)
        guard_info = {
            "merge_guard_overrides": int(overrides),
            "merge_guard_triggered": bool(overrides > 0),
        }
        return guarded_action, guard_info

    def _build_risk_field(self) -> None:
        self.risk_field = FDPFRiskField(
            upsilon_coeff_s=float(self.config.get("risk_upsilon_coeff_s", 0.3)),
            upsilon_coeff_l=float(self.config.get("risk_upsilon_coeff_l", 0.7)),
            major_scale=float(self.config.get("risk_major_scale", 2.0)),
            semi_scale=float(self.config.get("risk_semi_scale", 0.15)),
            decay_coeff=float(self.config.get("risk_decay_coeff", 0.5)),
            margin_distance=float(self.config.get("risk_lane_margin", 0.8)),
            look_back=float(self.config.get("risk_look_back", 30.0)),
            look_ahead=float(self.config.get("risk_look_ahead", 40.0)),
            lateral_scope_lanes=float(self.config.get("risk_lateral_scope_lanes", 2.0)),
        )

    def _evaluate_controlled_vehicle_risk(self) -> dict:
        n_agents = len(self.controlled_vehicles)
        if n_agents == 0:
            zeros = np.zeros((0,), dtype=np.float32)
            return {
                "total": zeros,
                "raw": zeros,
                "longitudinal": zeros,
                "lateral": zeros,
                "boundary": zeros,
                "direction": zeros,
                "neighbors": zeros.astype(np.int32),
            }

        if not bool(self.config.get("risk_assessment_enabled", True)):
            zeros = np.zeros((n_agents,), dtype=np.float32)
            return {
                "total": zeros,
                "raw": zeros.copy(),
                "longitudinal": zeros.copy(),
                "lateral": zeros.copy(),
                "boundary": zeros.copy(),
                "direction": zeros.copy(),
                "neighbors": np.zeros((n_agents,), dtype=np.int32),
            }

        if not hasattr(self, "risk_field") or self.risk_field is None:
            self._build_risk_field()

        profiles = [self.risk_field.evaluate_vehicle(vehicle, self.road) for vehicle in self.controlled_vehicles]
        return {
            "total": np.array([p["intensity_with_bound"] for p in profiles], dtype=np.float32),
            "raw": np.array([p["intensity_raw"] for p in profiles], dtype=np.float32),
            "longitudinal": np.array([p["intensity_s"] for p in profiles], dtype=np.float32),
            "lateral": np.array([p["intensity_l"] for p in profiles], dtype=np.float32),
            "boundary": np.array([p["intensity_bound"] for p in profiles], dtype=np.float32),
            "direction": np.array([p["direction"] for p in profiles], dtype=np.float32),
            "neighbors": np.array([int(p["neighbor_count"]) for p in profiles], dtype=np.int32),
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = []
        guarded_action, guard_info = self._apply_merge_guard(action)
        risk_before = self._evaluate_controlled_vehicle_risk()
        obs, reward, done, info = super().step(guarded_action)
        risk_after = self._evaluate_controlled_vehicle_risk()
        scenario_crashed = any(vehicle.crashed for vehicle in self.controlled_vehicles)
        scenario_success = self._is_scenario_success()

        if done:
            if scenario_crashed:
                done_reason = "scenario_crashed"
            elif scenario_success:
                done_reason = "success"
            else:
                done_reason = "timeout"
        else:
            done_reason = "running"

        info["scenario_crashed"] = scenario_crashed
        info["scenario_success"] = scenario_success
        info["done_reason"] = done_reason
        info.update(guard_info)
        info["agent_risk_pre"] = risk_before["total"]
        info["agent_risk_post"] = risk_after["total"]
        info["agent_risk_delta"] = risk_after["total"] - risk_before["total"]
        info["agent_risk_raw"] = risk_after["raw"]
        info["agent_risk_longitudinal"] = risk_after["longitudinal"]
        info["agent_risk_lateral"] = risk_after["lateral"]
        info["agent_risk_boundary"] = risk_after["boundary"]
        info["agent_risk_direction"] = risk_after["direction"]
        info["agent_risk_neighbors"] = risk_after["neighbors"]
        info["fleet_risk_mean"] = float(np.mean(risk_after["total"])) if len(risk_after["total"]) else 0.0
        info["fleet_risk_max"] = float(np.max(risk_after["total"])) if len(risk_after["total"]) else 0.0
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
        info["agents_info"] = agent_info

        for i, vehicle in enumerate(self.controlled_vehicles):
            actual_action = guarded_action[i] if isinstance(guarded_action, (list, tuple, np.ndarray)) else guarded_action
            vehicle.local_reward = self._agent_reward(actual_action, vehicle)
        # local reward
        info["agents_rewards"] = tuple(vehicle.local_reward for vehicle in self.controlled_vehicles)
        # regional reward
        self._regional_reward()
        info["regional_rewards"] = tuple(vehicle.regional_reward for vehicle in self.controlled_vehicles)

        obs = np.asarray(obs).reshape((len(obs), -1))
        return obs, reward, done, info

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _reset(self, num_CAV=0) -> None:
        self._make_road()
        self.controlled_vehicles_on_ramp = []

        # 对 MAPPO 来说，controlled_vehicles 数量必须固定，
        # 否则 replay memory 里的 state/action 形状会不一致。
        fixed_cav = int(self.config.get("controlled_vehicles", 0))

        # 固定 CAV 数量
        if fixed_cav > 0:
            num_CAV = fixed_cav
        else:
            # 只有在没配置 controlled_vehicles 时，才走旧的随机逻辑
            if self.config["traffic_density"] == 1:
                num_CAV = np.random.choice(np.arange(1, 4), 1)[0]
            elif self.config["traffic_density"] == 2:
                num_CAV = np.random.choice(np.arange(2, 5), 1)[0]
            elif self.config["traffic_density"] == 3:
                num_CAV = np.random.choice(np.arange(4, 7), 1)[0]
            else:
                num_CAV = 4

        # HDV 数量仍然按 traffic_density 随机
        if self.config["traffic_density"] == 1:
            num_HDV = np.random.choice(np.arange(3, 5), 1)[0]
        elif self.config["traffic_density"] == 2:
            num_HDV = np.random.choice(np.arange(4, 6), 1)[0]
        elif self.config["traffic_density"] == 3:
            num_HDV = np.random.choice(np.arange(6, 9), 1)[0]
        else:
            num_HDV = 3

        self._make_vehicles(num_CAV, num_HDV)
        self.action_is_safe = True
        self._build_risk_field()
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
         :return: the road
        """
        net = RoadNetwork()
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE

        main_lanes = max(1, int(self.config.get("lanes_count", 3)))
        ramp_lanes = max(1, int(self.config.get("ramp_lanes_count", 2)))
        lane_width = 4.0
        ramp_lane_width = lane_width

    # =========================
    # 1) 主路：三段全水平直线
    # =========================
        for lane_id in range(main_lanes):
            y = lane_id * lane_width
            left_line = c if lane_id == 0 else s
            right_line = c if lane_id == (main_lanes - 1) else s

        # 在 merge 区域，主路最右边界改成虚线，方便视觉上表示可并入
            merge_right_line = s if lane_id == (main_lanes - 1) else right_line

            net.add_lane(
                "a", "b",
                StraightLane(
                    [0, y],
                    [sum(self.ends[:2]), y],
                    width=lane_width,
                    line_types=[left_line, right_line]
                )
            )
            net.add_lane(
                "b", "c",
                StraightLane(
                    [sum(self.ends[:2]), y],
                    [sum(self.ends[:3]), y],
                    width=lane_width,
                    line_types=[left_line, merge_right_line]
                )
            )
            net.add_lane(
                "c", "d",
                StraightLane(
                    [sum(self.ends[:3]), y],
                    [sum(self.ends), y],
                     width=lane_width,
                     line_types=[left_line, right_line]
                )
            )

    # ==========================================
    # 2) 匝道：前段水平 + 中段斜线 + 后段严格水平
    #    这样可以保证：
    #    - 两条匝道边界保持平行
    #    - 最下面那条线是水平的
    # ==========================================
        ramp_end_base_y = main_lanes * lane_width
        ramp_start_base_y = ramp_end_base_y + 2 * ramp_lane_width
        ramp_end_lanes = []

        for ramp_id in range(ramp_lanes):
            y_end = ramp_end_base_y + ramp_id * ramp_lane_width
            y_start = ramp_start_base_y + ramp_id * ramp_lane_width
 
            ramp_left_line = c if ramp_id == 0 else s
        # 共享边界只画一次，避免中间线条看起来过粗
            ramp_right_line = c if ramp_id == (ramp_lanes - 1) else n
            ramp_merge_left_line = n if ramp_id == 0 else s

        # 匝道前半段：严格水平
            ljk = StraightLane(
                [0, y_start],
                [self.ends[0], y_start],
                width=ramp_lane_width,
                line_types=[ramp_left_line, ramp_right_line],
                forbidden=True,
            )

        # 匝道汇入段：斜线段
            lkb = StraightLane(
                [self.ends[0], y_start],
                [sum(self.ends[:2]), y_end],
                width=ramp_lane_width,
                line_types=[ramp_left_line, ramp_right_line],
                forbidden=True,
            )

        # 匝道末段：严格水平
            merge_start = [sum(self.ends[:2]), y_end]
            merge_end = [sum(self.ends[:3]), y_end]
            lbc = StraightLane(
                merge_start,
                merge_end,
                width=ramp_lane_width,
                line_types=[ramp_merge_left_line, ramp_right_line],
                forbidden=True,
            )

            net.add_lane("j", "k", ljk)
            net.add_lane("k", "b", lkb)
            net.add_lane("b", "c", lbc)
            ramp_end_lanes.append(lbc)

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"]
        )

        for ramp_lane in ramp_end_lanes:
            road.objects.append(Obstacle(road, ramp_lane.position(self.ends[2], 0)))

        self.road = road

    def _make_vehicles(self, num_CAV=4, num_HDV=3) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as the ego vehicles.
        :return: the ego-vehicle
        """
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []
        self.controlled_vehicles_on_ramp = []
        main_lanes = max(1, int(self.config.get("lanes_count", 3)))
        ramp_lanes = max(1, int(self.config.get("ramp_lanes_count", 2)))

        spawn_points_s = [10, 50, 90, 130, 170, 210]
        spawn_points_m = [5, 45, 85, 125, 165, 205]

        # Spawn points for CAVs
        spawn_point_s_c = np.random.choice(spawn_points_s, num_CAV // 2, replace=False)
        spawn_point_m_c = np.random.choice(spawn_points_m, num_CAV - num_CAV // 2, replace=False)
        spawn_point_s_c = list(spawn_point_s_c)
        spawn_point_m_c = list(spawn_point_m_c)

        for a in spawn_point_s_c:
            spawn_points_s.remove(a)
        for b in spawn_point_m_c:
            spawn_points_m.remove(b)

        # Spawn points for HDVs
        spawn_point_s_h = np.random.choice(spawn_points_s, num_HDV // 2, replace=False)
        spawn_point_m_h = np.random.choice(spawn_points_m, num_HDV - num_HDV // 2, replace=False)
        spawn_point_s_h = list(spawn_point_s_h)
        spawn_point_m_h = list(spawn_point_m_h)

        initial_speed = np.random.rand(num_CAV + num_HDV) * 2 + 25
        loc_noise = np.random.rand(num_CAV + num_HDV) * 3 - 1.5
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)

        # Spawn CAVs on the main road.
        for _ in range(num_CAV // 2):
            lane_id = int(np.random.choice(np.arange(main_lanes), 1)[0])
            ego_vehicle = self.action_type.vehicle_class(
                road,
                road.network.get_lane(("a", "b", lane_id)).position(
                    spawn_point_s_c.pop(0) + loc_noise.pop(0), 0
                ),
                speed=initial_speed.pop(0),
            )
            self.controlled_vehicles.append(ego_vehicle)
            self.controlled_vehicles_on_ramp.append(False)
            road.vehicles.append(ego_vehicle)

        # Spawn remaining CAVs on the ramp.
        for _ in range(num_CAV - num_CAV // 2):
            ramp_lane_id = int(np.random.choice(np.arange(ramp_lanes), 1)[0])
            ego_vehicle = self.action_type.vehicle_class(
                road,
                road.network.get_lane(("j", "k", ramp_lane_id)).position(
                    spawn_point_m_c.pop(0) + loc_noise.pop(0), 0
                ),
                speed=initial_speed.pop(0),
            )
            self.controlled_vehicles.append(ego_vehicle)
            self.controlled_vehicles_on_ramp.append(True)
            road.vehicles.append(ego_vehicle)

        # Spawn HDVs on the main road.
        for _ in range(num_HDV // 2):
            lane_id = int(np.random.choice(np.arange(main_lanes), 1)[0])
            road.vehicles.append(
                other_vehicles_type(
                    road,
                    road.network.get_lane(("a", "b", lane_id)).position(
                        spawn_point_s_h.pop(0) + loc_noise.pop(0), 0
                    ),
                    speed=initial_speed.pop(0),
                )
            )

        # Spawn remaining HDVs on the ramp.
        for _ in range(num_HDV - num_HDV // 2):
            ramp_lane_id = int(np.random.choice(np.arange(ramp_lanes), 1)[0])
            road.vehicles.append(
                other_vehicles_type(
                    road,
                    road.network.get_lane(("j", "k", ramp_lane_id)).position(
                        spawn_point_m_h.pop(0) + loc_noise.pop(0), 0
                    ),
                    speed=initial_speed.pop(0),
                )
            )

    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds


class MergeEnvMARL(MergeEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }},
            "controlled_vehicles": 4
        })
        return config


register(
    id='merge-v1',
    entry_point='highway_env.envs:MergeEnv',
)

register(
    id='merge-multi-agent-v0',
    entry_point='highway_env.envs:MergeEnvMARL',
)
