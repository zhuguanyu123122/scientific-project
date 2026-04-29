import importlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from shapely.affinity import rotate as shp_rotate
    from shapely.affinity import translate as shp_translate
    from shapely.geometry import box as shp_box

    SHAPELY_AVAILABLE = True
except Exception:
    SHAPELY_AVAILABLE = False

try:
    # Optional dependency. API differs by version, so integration is best-effort.
    importlib.import_module("py_rss_anticipate")
    PY_RSS_AVAILABLE = True
except Exception:
    PY_RSS_AVAILABLE = False


# SanDRA semantic action set (top-level actions seen by the LLM).
# Each semantic action maps to one primitive highway-env action id:
# 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
SANDRA_SEMANTIC_ACTIONS: Dict[int, Dict[str, object]] = {
    0: {
        "name": "EMERGENCY_DECELERATE",
        "description": "Strongly reduce speed in urgent conflict.",
        "primitive": 4,
        "longitudinal_action": "decelerate",
        "lateral_action": "follow_lane",
    },
    1: {
        "name": "KEEP_LANE_ACCELERATE",
        "description": "Keep lane and accelerate if front gap is sufficient.",
        "primitive": 3,
        "longitudinal_action": "accelerate",
        "lateral_action": "follow_lane",
    },
    2: {
        "name": "KEEP_LANE_DECELERATE",
        "description": "Keep lane and decelerate to increase headway.",
        "primitive": 4,
        "longitudinal_action": "decelerate",
        "lateral_action": "follow_lane",
    },
    3: {
        "name": "LEFT_KEEP_SPEED",
        "description": "Change to left lane while keeping speed.",
        "primitive": 0,
        "longitudinal_action": "keep",
        "lateral_action": "left",
    },
    4: {
        "name": "LEFT_ACCELERATE",
        "description": "Change to left lane and accelerate.",
        "primitive": 0,
        "longitudinal_action": "accelerate",
        "lateral_action": "left",
    },
    5: {
        "name": "LEFT_DECELERATE",
        "description": "Change to left lane and decelerate.",
        "primitive": 0,
        "longitudinal_action": "decelerate",
        "lateral_action": "left",
    },
    6: {
        "name": "RIGHT_KEEP_SPEED",
        "description": "Change to right lane while keeping speed.",
        "primitive": 2,
        "longitudinal_action": "keep",
        "lateral_action": "right",
    },
    7: {
        "name": "RIGHT_ACCELERATE",
        "description": "Change to right lane and accelerate.",
        "primitive": 2,
        "longitudinal_action": "accelerate",
        "lateral_action": "right",
    },
    8: {
        "name": "RIGHT_DECELERATE",
        "description": "Change to right lane and decelerate.",
        "primitive": 2,
        "longitudinal_action": "decelerate",
        "lateral_action": "right",
    },
}


PAIR_TO_DISCRETE: Dict[Tuple[str, str], int] = {
    ("keep", "left"): 0,
    ("decelerate", "left"): 0,
    ("accelerate", "left"): 0,
    ("keep", "follow_lane"): 1,
    ("accelerate", "follow_lane"): 3,
    ("decelerate", "follow_lane"): 4,
    ("keep", "right"): 2,
    ("decelerate", "right"): 2,
    ("accelerate", "right"): 2,
}


DISCRETE_TO_PAIR: Dict[int, Tuple[str, str]] = {
    0: ("keep", "left"),
    1: ("keep", "follow_lane"),
    2: ("keep", "right"),
    3: ("accelerate", "follow_lane"),
    4: ("decelerate", "follow_lane"),
}


def load_openai_env_file(path: str) -> Dict[str, str]:
    loaded: Dict[str, str] = {}
    if not path:
        return loaded
    if not os.path.exists(path):
        return loaded

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ[key] = value
            loaded[key] = value
    return loaded


@dataclass
class SanDRALLMConfig:
    enabled: bool = False
    risk_threshold: float = 1.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    temperature: float = 0.2
    request_timeout: float = 12.0
    max_retries: int = 1
    log_decisions: bool = True
    top_k: int = 3
    formal_check_enabled: bool = True
    rule_min_ttc: float = 2.0
    rule_min_headway_time: float = 1.5
    rule_lane_change_front_gap: float = 12.0
    rule_lane_change_rear_gap: float = 8.0
    rule_verify_horizon_s: float = 2.0
    use_shapely: bool = True
    use_py_rss: bool = True


class LightweightRuleVerifier:
    def __init__(self, config: SanDRALLMConfig) -> None:
        self.config = config
        self.use_shapely = bool(config.use_shapely and SHAPELY_AVAILABLE)
        self.use_py_rss = bool(config.use_py_rss and PY_RSS_AVAILABLE)
        self.lane_width_default = 4.0
        self.dt = 0.4

    def backend_summary(self) -> str:
        backends = ["python-geometry"]
        if self.use_shapely:
            backends.append("shapely")
        if self.use_py_rss:
            backends.append("py_rss_anticipate")
        return "+".join(backends)

    @staticmethod
    def _vehicle_length(vehicle) -> float:
        return float(getattr(vehicle, "LENGTH", 5.0))

    @staticmethod
    def _vehicle_width(vehicle) -> float:
        return float(getattr(vehicle, "WIDTH", 2.0))

    @staticmethod
    def _lane_id(vehicle) -> Optional[int]:
        lane_index = getattr(vehicle, "lane_index", None)
        if isinstance(lane_index, (tuple, list)) and len(lane_index) >= 3:
            try:
                return int(lane_index[2])
            except Exception:
                return None
        return None

    @staticmethod
    def _vehicle_xy(vehicle) -> Tuple[float, float]:
        x = float(vehicle.position[0]) if hasattr(vehicle, "position") else 0.0
        y = float(vehicle.position[1]) if hasattr(vehicle, "position") else 0.0
        return x, y

    @staticmethod
    def _vehicle_speed(vehicle) -> float:
        return float(getattr(vehicle, "speed", 0.0))

    @staticmethod
    def _vehicle_heading(vehicle) -> float:
        return float(getattr(vehicle, "heading", 0.0))

    def _lane_count(self, vehicle) -> Optional[int]:
        lane_index = getattr(vehicle, "lane_index", None)
        road = getattr(vehicle, "road", None)
        if road is None:
            return None
        try:
            _from, _to, _id = lane_index
            return len(road.network.graph[_from][_to])
        except Exception:
            return None

    def _target_lane_id(self, ego_vehicle, action_id: int) -> Optional[int]:
        lane_id = self._lane_id(ego_vehicle)
        if lane_id is None:
            return None
        if int(action_id) == 0:  # LANE_LEFT in highway-env means lane_id - 1
            lane_id = lane_id - 1
        elif int(action_id) == 2:  # LANE_RIGHT means lane_id + 1
            lane_id = lane_id + 1

        lane_count = self._lane_count(ego_vehicle)
        if lane_count is not None and (lane_id < 0 or lane_id >= lane_count):
            return None
        return lane_id

    def _target_lane_center_y(self, ego_vehicle, target_lane_id: Optional[int]) -> Optional[float]:
        if target_lane_id is None:
            return None
        lane_index = getattr(ego_vehicle, "lane_index", None)
        road = getattr(ego_vehicle, "road", None)
        if road is None or not isinstance(lane_index, (tuple, list)) or len(lane_index) < 3:
            return None
        try:
            _from, _to, _ = lane_index
            target_lane = road.network.get_lane((_from, _to, int(target_lane_id)))
            x, y = self._vehicle_xy(ego_vehicle)
            if hasattr(target_lane, "local_coordinates") and hasattr(target_lane, "position"):
                lon, _ = target_lane.local_coordinates(np.array([x, y]))
                p = target_lane.position(lon, 0.0)
                return float(p[1])
        except Exception:
            return None
        return None

    def _predict_xy(self, vehicle, t: float) -> Tuple[float, float]:
        x, y = self._vehicle_xy(vehicle)
        speed = self._vehicle_speed(vehicle)
        heading = self._vehicle_heading(vehicle)
        vx = speed * math.cos(heading)
        vy = speed * math.sin(heading)
        return x + vx * t, y + vy * t

    def _ego_predicted_xy(
        self,
        ego_vehicle,
        action_id: int,
        t: float,
        target_lane_center_y: Optional[float],
    ) -> Tuple[float, float]:
        x, y = self._predict_xy(ego_vehicle, t)
        if int(action_id) not in (0, 2):
            return x, y
        if target_lane_center_y is None:
            return x, y
        lane_change_time = 1.2
        ratio = min(1.0, max(0.0, t / lane_change_time))
        y0 = self._vehicle_xy(ego_vehicle)[1]
        return x, y0 + (target_lane_center_y - y0) * ratio

    def _boxes_overlap(self, v1, p1: Tuple[float, float], v2, p2: Tuple[float, float]) -> bool:
        l1, w1 = self._vehicle_length(v1), self._vehicle_width(v1)
        l2, w2 = self._vehicle_length(v2), self._vehicle_width(v2)
        dx = abs(float(p1[0]) - float(p2[0]))
        dy = abs(float(p1[1]) - float(p2[1]))
        return dx <= (l1 + l2) / 2.0 and dy <= (w1 + w2) / 2.0

    def _shapely_overlap(
        self,
        v1,
        p1: Tuple[float, float],
        h1: float,
        v2,
        p2: Tuple[float, float],
        h2: float,
    ) -> bool:
        l1, w1 = self._vehicle_length(v1), self._vehicle_width(v1)
        l2, w2 = self._vehicle_length(v2), self._vehicle_width(v2)
        poly1 = shp_box(-l1 / 2.0, -w1 / 2.0, l1 / 2.0, w1 / 2.0)
        poly2 = shp_box(-l2 / 2.0, -w2 / 2.0, l2 / 2.0, w2 / 2.0)
        poly1 = shp_rotate(poly1, h1, origin=(0, 0), use_radians=True)
        poly2 = shp_rotate(poly2, h2, origin=(0, 0), use_radians=True)
        poly1 = shp_translate(poly1, xoff=float(p1[0]), yoff=float(p1[1]))
        poly2 = shp_translate(poly2, xoff=float(p2[0]), yoff=float(p2[1]))
        return bool(poly1.intersects(poly2))

    def _classify_lane_relation(
        self,
        ego_lane_id: Optional[int],
        other_lane_id: Optional[int],
        ego_y: float,
        other_y: float,
    ) -> int:
        # 0: same lane, -1: left-adjacent, +1: right-adjacent, 99: unrelated
        if ego_lane_id is not None and other_lane_id is not None:
            d = int(other_lane_id) - int(ego_lane_id)
            if d == 0:
                return 0
            if d == -1:
                return -1
            if d == 1:
                return 1
            return 99
        lane_width = self.lane_width_default
        dy = float(other_y - ego_y)
        if abs(dy) < 0.5 * lane_width:
            return 0
        if -1.5 * lane_width < dy <= -0.5 * lane_width:
            return -1
        if 0.5 * lane_width <= dy < 1.5 * lane_width:
            return 1
        return 99

    @staticmethod
    def _safe_div(a: float, b: float, default: float = float("inf")) -> float:
        if abs(b) < 1e-6:
            return default
        return a / b

    def verify_action(self, env, car_index: int, action_id: int) -> Dict[str, object]:
        result: Dict[str, object] = {
            "action": int(action_id),
            "safe": True,
            "score": 0.0,
            "violated_rules": [],
            "backend": self.backend_summary(),
            "meta": {},
        }
        if not bool(self.config.formal_check_enabled):
            return result

        if not hasattr(env, "controlled_vehicles") or car_index >= len(env.controlled_vehicles):
            result["safe"] = False
            result["score"] = 100.0
            result["violated_rules"] = ["invalid_ego_vehicle"]
            return result

        ego = env.controlled_vehicles[car_index]
        road = getattr(env, "road", None)
        if road is None:
            result["safe"] = False
            result["score"] = 100.0
            result["violated_rules"] = ["missing_road"]
            return result

        ego_x, ego_y = self._vehicle_xy(ego)
        ego_speed = self._vehicle_speed(ego)
        ego_lane_id = self._lane_id(ego)
        target_lane_id = self._target_lane_id(ego, int(action_id))
        is_lane_change = int(action_id) in (0, 2)

        if is_lane_change and target_lane_id is None:
            result["safe"] = False
            result["score"] = 100.0
            result["violated_rules"].append("target_lane_unreachable")
            result["meta"] = {"target_lane_id": None}
            return result

        target_lane_center_y = self._target_lane_center_y(ego, target_lane_id)

        min_front_gap = float(self.config.rule_lane_change_front_gap)
        min_rear_gap = float(self.config.rule_lane_change_rear_gap)
        min_headway_time = float(self.config.rule_min_headway_time)
        min_ttc = float(self.config.rule_min_ttc)
        horizon = max(float(self.config.rule_verify_horizon_s), self.dt)
        steps = int(max(1.0, horizon / self.dt))

        nearest_front_dx = float("inf")
        nearest_front_speed = 0.0
        lane_change_front_dx = float("inf")
        lane_change_rear_dx = float("inf")
        collision_predicted = False

        neighbors = []
        for other in getattr(road, "vehicles", []):
            if other is ego:
                continue
            ox, oy = self._vehicle_xy(other)
            if abs(ox - ego_x) > 120.0 and abs(oy - ego_y) > 20.0:
                continue
            neighbors.append(other)

            rel = self._classify_lane_relation(
                ego_lane_id,
                self._lane_id(other),
                ego_y,
                oy,
            )
            dx0 = ox - ego_x
            if rel == 0 and dx0 > 0.0 and dx0 < nearest_front_dx:
                nearest_front_dx = dx0
                nearest_front_speed = self._vehicle_speed(other)

            if is_lane_change and target_lane_id is not None:
                target_rel = None
                other_lane = self._lane_id(other)
                if other_lane is not None and target_lane_id is not None:
                    target_rel = int(other_lane) - int(target_lane_id)
                if target_rel == 0:
                    if dx0 > 0.0 and dx0 < lane_change_front_dx:
                        lane_change_front_dx = dx0
                    if dx0 <= 0.0 and abs(dx0) < lane_change_rear_dx:
                        lane_change_rear_dx = abs(dx0)

        # Rule 1: keep-lane headway and TTC.
        if not is_lane_change and math.isfinite(nearest_front_dx):
            headway_time = self._safe_div(nearest_front_dx, max(ego_speed, 0.1))
            rel_speed = ego_speed - nearest_front_speed
            ttc = self._safe_div(nearest_front_dx, rel_speed) if rel_speed > 0.0 else float("inf")
            result["meta"]["headway_time"] = float(headway_time)
            result["meta"]["ttc"] = float(ttc)
            if headway_time < min_headway_time:
                result["violated_rules"].append("min_headway_time")
            if ttc < min_ttc:
                result["violated_rules"].append("min_ttc")

        # Rule 2: lane-change gap check.
        if is_lane_change:
            dynamic_front_gap = max(min_front_gap, ego_speed * 0.8 + 4.0)
            dynamic_rear_gap = max(min_rear_gap, ego_speed * 0.4 + 3.0)
            result["meta"]["lane_change_front_gap"] = float(lane_change_front_dx)
            result["meta"]["lane_change_rear_gap"] = float(lane_change_rear_dx)
            if lane_change_front_dx < dynamic_front_gap:
                result["violated_rules"].append("lane_change_front_gap")
            if lane_change_rear_dx < dynamic_rear_gap:
                result["violated_rules"].append("lane_change_rear_gap")

        # Rule 3: collision prediction over short horizon.
        ego_heading = self._vehicle_heading(ego)
        for k in range(1, steps + 1):
            t = min(horizon, float(k) * self.dt)
            ego_p = self._ego_predicted_xy(ego, int(action_id), t, target_lane_center_y)
            for other in neighbors:
                other_p = self._predict_xy(other, t)
                if self.use_shapely:
                    overlap = self._shapely_overlap(
                        ego, ego_p, ego_heading, other, other_p, self._vehicle_heading(other)
                    )
                else:
                    overlap = self._boxes_overlap(ego, ego_p, other, other_p)
                if overlap:
                    collision_predicted = True
                    break
            if collision_predicted:
                break

        if collision_predicted:
            result["violated_rules"].append("predicted_collision")

        # Optional RSS check hook (best-effort only).
        if self.use_py_rss:
            try:
                mod = importlib.import_module("py_rss_anticipate")
                fn = None
                for fn_name in ("is_safe", "check_safety", "verify"):
                    candidate = getattr(mod, fn_name, None)
                    if callable(candidate):
                        fn = candidate
                        break
                if fn is not None:
                    rss_safe = bool(fn(ego_speed=ego_speed, action_id=int(action_id)))
                    if not rss_safe:
                        result["violated_rules"].append("rss_unsafe")
            except Exception:
                # Keep best-effort behavior. Geometry rules remain active.
                pass

        # Score and safety label.
        score = 0.0
        for rule in result["violated_rules"]:
            if rule == "predicted_collision":
                score += 100.0
            elif rule in ("lane_change_front_gap", "lane_change_rear_gap"):
                score += 20.0
            elif rule in ("min_ttc", "min_headway_time", "rss_unsafe"):
                score += 10.0
            else:
                score += 5.0

        result["score"] = float(score)
        result["safe"] = len(result["violated_rules"]) == 0
        result["meta"]["target_lane_id"] = target_lane_id
        return result


class SanDRALLMDecider:
    def __init__(self, config: SanDRALLMConfig) -> None:
        self.config = config
        self._warned_unavailable = False
        self.client = None
        self.formal_verifier = LightweightRuleVerifier(config)

        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = config.base_url or os.getenv("OPENAI_BASE_URL")
        if not base_url:
            base_url = config.api_base or os.getenv("OPENAI_API_BASE")

        self.model_name = config.model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

        if config.enabled and OpenAI is not None and api_key:
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            try:
                self.client = OpenAI(**kwargs)
            except Exception:
                self.client = None

    def _available_discrete_actions(self, action_mask: Optional[np.ndarray]) -> List[int]:
        if action_mask is None:
            return [0, 1, 2, 3, 4]
        mask = np.asarray(action_mask, dtype=np.float32).reshape(-1)
        available = [int(i) for i, m in enumerate(mask.tolist()) if m > 0.5]
        if not available:
            return [1]
        return available

    def _fallback_action(self, available: List[int], rl_action: int) -> int:
        if rl_action in available:
            return int(rl_action)
        for candidate in [4, 1, 3, 0, 2]:
            if candidate in available:
                return int(candidate)
        return int(available[0])

    def _build_candidate_semantic_actions(self, available_actions: List[int]) -> List[Dict[str, object]]:
        available_set = set(int(a) for a in available_actions)
        candidates: List[Dict[str, object]] = []
        for semantic_id, item in SANDRA_SEMANTIC_ACTIONS.items():
            primitive = int(item["primitive"])
            if primitive not in available_set:
                continue
            candidates.append(
                {
                    "semantic_action_id": int(semantic_id),
                    "action_name": str(item["name"]),
                    "description": str(item["description"]),
                    "primitive_action": primitive,
                    "longitudinal_action": str(item["longitudinal_action"]),
                    "lateral_action": str(item["lateral_action"]),
                }
            )
        return candidates

    @staticmethod
    def _get_nearby_summary(env, ego_vehicle, max_neighbors: int = 8) -> List[Dict[str, float]]:
        summary = []
        road = getattr(env, "road", None)
        if road is None:
            return summary

        x0 = float(ego_vehicle.position[0])
        y0 = float(ego_vehicle.position[1])
        candidates = []
        for v in getattr(road, "vehicles", []):
            if v is ego_vehicle:
                continue
            try:
                dx = float(v.position[0] - x0)
                dy = float(v.position[1] - y0)
                dist = float(np.hypot(dx, dy))
            except Exception:
                continue
            candidates.append((dist, v, dx, dy))

        candidates.sort(key=lambda x: x[0])
        for dist, v, dx, dy in candidates[:max_neighbors]:
            summary.append(
                {
                    "dx": round(dx, 3),
                    "dy": round(dy, 3),
                    "distance": round(dist, 3),
                    "speed": round(float(getattr(v, "speed", 0.0)), 3),
                    "lane_index": str(getattr(v, "lane_index", "N/A")),
                }
            )
        return summary

    def _build_prompt(
        self,
        car_no: int,
        risk_value: float,
        ego_vehicle,
        nearby_summary: List[Dict[str, float]],
        candidate_actions: List[Dict[str, object]],
        rl_action: int,
    ) -> Tuple[str, str]:
        top_k = max(1, int(self.config.top_k))
        system_prompt = (
            "You are the SanDRA driving decision module. "
            "Return a ranked list of candidate actions from best to worst. "
            "Do not output STOP."
        )
        user_payload = {
            "car_no": int(car_no),
            "risk_value": round(float(risk_value), 5),
            "risk_threshold": round(float(self.config.risk_threshold), 5),
            "ego_state": {
                "x": round(float(ego_vehicle.position[0]), 3),
                "y": round(float(ego_vehicle.position[1]), 3),
                "speed": round(float(getattr(ego_vehicle, "speed", 0.0)), 3),
                "lane_index": str(getattr(ego_vehicle, "lane_index", "N/A")),
            },
            "nearby_vehicles": nearby_summary,
            "rl_action": int(rl_action),
            "top_k": top_k,
            "candidate_actions": candidate_actions,
            "required_output_format": {
                "best_combination": {"semantic_action_id": "int", "reason": "string"},
                "second_best_combination": {"semantic_action_id": "int", "reason": "string"},
                "third_best_combination": {"semantic_action_id": "int", "reason": "string"},
                "alternative": "ranked_actions=[{semantic_action_id:int, reason:string}, ...]",
            },
        }
        user_prompt = (
            "Rank the candidate actions by intent, comfort, and progress, then output JSON only.\n"
            + json.dumps(user_payload, ensure_ascii=True)
        )
        return system_prompt, user_prompt

    def _query_llm(self, system_prompt: str, user_prompt: str) -> Optional[dict]:
        if self.client is None:
            return None
        retries = max(1, int(self.config.max_retries))
        for _ in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=float(self.config.temperature),
                    timeout=float(self.config.request_timeout),
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                if not content:
                    continue
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return None

    @staticmethod
    def _dedupe_keep_order(items: List[int]) -> List[int]:
        out: List[int] = []
        seen = set()
        for item in items:
            v = int(item)
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    def _decode_to_discrete(self, item: Dict[str, object]) -> Optional[int]:
        if not isinstance(item, dict):
            return None
        if "semantic_action_id" in item:
            try:
                sid = int(item["semantic_action_id"])
                if sid in SANDRA_SEMANTIC_ACTIONS:
                    return int(SANDRA_SEMANTIC_ACTIONS[sid]["primitive"])
            except Exception:
                pass
        if "action_id" in item:
            try:
                aid = int(item["action_id"])
                if aid in SANDRA_SEMANTIC_ACTIONS:
                    return int(SANDRA_SEMANTIC_ACTIONS[aid]["primitive"])
            except Exception:
                pass
        if "primitive_action" in item:
            try:
                return int(item["primitive_action"])
            except Exception:
                pass
        if "mapped_highway_action" in item:
            try:
                return int(item["mapped_highway_action"])
            except Exception:
                pass
        lon = str(item.get("longitudinal_action", "")).strip().lower()
        lat = str(item.get("lateral_action", "")).strip().lower()
        if lon == "stop":
            return None
        if lon and lat:
            return PAIR_TO_DISCRETE.get((lon, lat), None)
        return None

    def _parse_ranked_actions(
        self,
        llm_output: Optional[Dict[str, object]],
        available_actions: List[int],
        rl_action: int,
    ) -> Tuple[List[int], List[Dict[str, object]]]:
        top_k = max(1, int(self.config.top_k))
        parse_notes: List[Dict[str, object]] = []
        available_set = set(int(a) for a in available_actions)
        ranked_actions: List[int] = []

        if isinstance(llm_output, dict):
            if isinstance(llm_output.get("ranked_actions"), list):
                for i, item in enumerate(llm_output.get("ranked_actions", [])):
                    action = self._decode_to_discrete(item if isinstance(item, dict) else {})
                    parse_notes.append({"rank": i + 1, "decoded_action": action})
                    if action is not None and int(action) in available_set:
                        ranked_actions.append(int(action))

            key_order = ["best_combination", "second_best_combination", "third_best_combination"]
            for i, key in enumerate(key_order[:top_k]):
                if key not in llm_output:
                    continue
                item = llm_output.get(key)
                action = self._decode_to_discrete(item if isinstance(item, dict) else {})
                parse_notes.append({"rank_key": key, "decoded_action": action})
                if action is not None and int(action) in available_set:
                    ranked_actions.append(int(action))

        ranked_actions = self._dedupe_keep_order(ranked_actions)

        # Fill to top_k with fallback ordering so downstream verification is always deterministic.
        fallback_pref = []
        if rl_action in available_set:
            fallback_pref.append(int(rl_action))
        if top_k > len(fallback_pref):
            if rl_action == 4:
                fallback_pref.extend([1, 3, 0, 2])
            else:
                fallback_pref.extend([4, 1, 3, 0, 2])
        for act in fallback_pref:
            if act in available_set and act not in ranked_actions:
                ranked_actions.append(int(act))
            if len(ranked_actions) >= top_k:
                break

        if len(ranked_actions) < top_k:
            for act in available_actions:
                if int(act) not in ranked_actions:
                    ranked_actions.append(int(act))
                if len(ranked_actions) >= top_k:
                    break
        return ranked_actions, parse_notes

    def decide_action(
        self,
        env,
        car_index: int,
        risk_value: float,
        action_mask: Optional[np.ndarray],
        rl_action: int,
    ) -> Tuple[int, str, Dict[str, object]]:
        """
        Returns:
            selected_action (int): chosen primitive action id
            source (str): decision source label
            meta (dict): ranked actions + verification details
        """
        available = self._available_discrete_actions(action_mask)
        fallback = self._fallback_action(available, rl_action)
        meta: Dict[str, object] = {
            "ranked_actions": [],
            "verification_results": [],
            "parse_notes": [],
            "backend": self.formal_verifier.backend_summary(),
        }

        if not hasattr(env, "controlled_vehicles") or car_index >= len(env.controlled_vehicles):
            meta["ranked_actions"] = [int(fallback)]
            return int(fallback), "fallback_invalid_env", meta

        ego_vehicle = env.controlled_vehicles[car_index]
        llm_output = None
        llm_used = False
        if self.client is not None and bool(self.config.enabled):
            candidate_actions = self._build_candidate_semantic_actions(available)
            nearby_summary = self._get_nearby_summary(env, ego_vehicle)
            system_prompt, user_prompt = self._build_prompt(
                car_no=car_index + 1,
                risk_value=risk_value,
                ego_vehicle=ego_vehicle,
                nearby_summary=nearby_summary,
                candidate_actions=candidate_actions,
                rl_action=int(rl_action),
            )
            llm_output = self._query_llm(system_prompt, user_prompt)
            llm_used = llm_output is not None
        elif bool(self.config.enabled) and not self._warned_unavailable:
            print(
                "[SANDRA] LLM 客户端不可用（缺少 openai 包或 API Key/Base URL），将使用规则回退策略。",
                flush=True,
            )
            self._warned_unavailable = True

        ranked_actions, parse_notes = self._parse_ranked_actions(llm_output, available, int(rl_action))
        if len(ranked_actions) == 0:
            ranked_actions = [int(fallback)]

        meta["ranked_actions"] = [int(a) for a in ranked_actions]
        meta["parse_notes"] = parse_notes

        # Safety verification over ranked candidates:
        verification_results: List[Dict[str, object]] = []
        selected_action = None
        selected_rank = None
        for rank, action in enumerate(ranked_actions, start=1):
            verification = self.formal_verifier.verify_action(env, car_index, int(action))
            verification["rank"] = int(rank)
            verification_results.append(verification)
            if verification.get("safe", False):
                selected_action = int(action)
                selected_rank = int(rank)
                break

        if selected_action is None:
            # No safe action according to lightweight formal checks:
            # choose minimal-score action from ranking, then fallback.
            if len(verification_results) > 0:
                best = min(verification_results, key=lambda x: float(x.get("score", 1e9)))
                selected_action = int(best.get("action", fallback))
                source = "sandra_ranked_no_safe"
            else:
                selected_action = int(fallback)
                source = "fallback_no_verification"
            selected_rank = None
        else:
            source = "sandra_ranked_verified"

        if not llm_used:
            source = f"fallback_ranked+{source}"

        meta["verification_results"] = verification_results
        meta["selected_action"] = int(selected_action)
        meta["selected_rank"] = selected_rank
        meta["llm_used"] = bool(llm_used)
        return int(selected_action), source, meta

    def available_discrete_actions(self, action_mask: Optional[np.ndarray]) -> List[int]:
        return self._available_discrete_actions(action_mask)
