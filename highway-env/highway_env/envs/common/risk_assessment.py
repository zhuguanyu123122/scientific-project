import math
from typing import Dict, List

import numpy as np

from highway_env import utils


class FDPFRiskField:
    """
    Frenet-inspired Dynamic Potential Field (FDPF) risk estimator.

    The implementation follows the core idea from Safe Human-in-the-Loop RL:
    1) project surrounding traffic into ego lane Frenet coordinates;
    2) aggregate a velocity-aware exponential field;
    3) inject lane-boundary risk.
    """

    def __init__(
        self,
        upsilon_coeff_s: float = 0.3,
        upsilon_coeff_l: float = 0.7,
        major_scale: float = 2.0,
        semi_scale: float = 0.15,
        decay_coeff: float = 0.5,
        margin_distance: float = 0.8,
        look_back: float = 30.0,
        look_ahead: float = 40.0,
        lateral_scope_lanes: float = 2.0,
    ) -> None:
        self.upsilon_coeff_s = float(upsilon_coeff_s)
        self.upsilon_coeff_l = float(upsilon_coeff_l)
        self.major_scale = float(major_scale)
        self.semi_scale = float(semi_scale)
        self.decay_coeff = float(decay_coeff)
        self.margin_distance = float(margin_distance)
        self.look_back = float(look_back)
        self.look_ahead = float(look_ahead)
        self.lateral_scope_lanes = float(lateral_scope_lanes)

    def _empty_result(self) -> Dict[str, float]:
        return {
            "intensity_raw": 0.0,
            "intensity_with_bound": 0.0,
            "direction": 0.0,
            "intensity_s": 0.0,
            "intensity_l": 0.0,
            "intensity_bound": 0.0,
            "neighbor_count": 0.0,
        }

    def evaluate_vehicle(self, ego_vehicle, road) -> Dict[str, float]:
        if ego_vehicle is None or road is None:
            return self._empty_result()
        lane_index = getattr(ego_vehicle, "lane_index", None)
        if lane_index is None:
            return self._empty_result()

        try:
            lane = road.network.get_lane(lane_index)
            lane_width = float(lane.width_at(0.0))
            ego_s, ego_l = lane.local_coordinates(ego_vehicle.position)
            lane_heading = lane.heading_at(ego_s)
        except Exception:
            return self._empty_result()

        ego_heading = float(getattr(ego_vehicle, "heading", 0.0))
        ego_delta_yaw = float(utils.wrap_to_pi(ego_heading - lane_heading))
        ego_v = float(getattr(ego_vehicle, "speed", 0.0))
        ego_v_s = ego_v * math.cos(ego_delta_yaw)
        ego_v_l = ego_v * math.sin(ego_delta_yaw)

        intensity_s = 0.0
        intensity_l = 0.0
        neighbor_count = 0

        entities: List[object] = list(getattr(road, "vehicles", []))
        entities.extend(list(getattr(road, "objects", [])))

        for entity in entities:
            if entity is ego_vehicle:
                continue
            if not hasattr(entity, "position"):
                continue
            try:
                s_i, l_i = lane.local_coordinates(entity.position)
            except Exception:
                continue

            relative_s = s_i - ego_s
            relative_l = l_i - ego_l
            if relative_s < -self.look_back or relative_s > self.look_ahead:
                continue
            if abs(relative_l) > self.lateral_scope_lanes * lane_width:
                continue

            obs_heading = float(getattr(entity, "heading", 0.0))
            obs_delta_yaw = float(utils.wrap_to_pi(obs_heading - lane_heading))
            obs_speed = float(getattr(entity, "speed", 0.0))
            obstacle_v_s = obs_speed * math.cos(obs_delta_yaw)
            obstacle_v_l = obs_speed * math.sin(obs_delta_yaw)

            relative_velocity_s = max(ego_v_s - obstacle_v_s, 0.0)
            relative_velocity_l = max(ego_v_l - obstacle_v_l, 0.0)
            coeff_s = self.upsilon_coeff_s / max(
                self.major_scale * self.upsilon_coeff_s + relative_velocity_s, 1e-6
            )
            coeff_l = self.upsilon_coeff_l / max(
                self.semi_scale * self.upsilon_coeff_l + relative_velocity_l, 1e-6
            )

            euclidean_distance = math.sqrt(
                (relative_s ** 2) * coeff_s + (relative_l ** 2) * coeff_l
            )
            intensity = math.exp(-euclidean_distance * self.decay_coeff)
            relative_direction = math.atan2(relative_l, relative_s)
            intensity_s += intensity * math.cos(relative_direction)
            intensity_l += intensity * math.sin(relative_direction)
            neighbor_count += 1

        left_boundary = lane_width * 0.5 - self.margin_distance
        right_boundary = -lane_width * 0.5 + self.margin_distance
        left_off_road = max(0.0, ego_l - left_boundary)
        right_off_road = min(0.0, ego_l - right_boundary)
        intensity_bound = left_off_road ** 2 - right_off_road ** 2

        intensity_l_with_bound = intensity_l + intensity_bound
        intensity_raw = math.sqrt(intensity_s ** 2 + intensity_l ** 2)
        intensity_with_bound = math.sqrt(intensity_s ** 2 + intensity_l_with_bound ** 2)
        frenet_direction = math.atan2(intensity_l_with_bound, intensity_s)

        return {
            "intensity_raw": float(intensity_raw),
            "intensity_with_bound": float(intensity_with_bound),
            "direction": float(frenet_direction),
            "intensity_s": float(intensity_s),
            "intensity_l": float(intensity_l),
            "intensity_bound": float(intensity_bound),
            "neighbor_count": float(neighbor_count),
        }
