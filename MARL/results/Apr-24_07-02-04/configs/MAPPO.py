import torch as th
from torch import nn
import configparser
import copy
import random

config_dir = 'configs/configs_ppo.ini'
config = configparser.ConfigParser()
config.read(config_dir)
torch_seed = config.getint('MODEL_CONFIG', 'torch_seed')
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True

from torch.optim import Adam, RMSprop

import numpy as np
import os, logging
from copy import deepcopy
from single_agent.Memory_common import OnPolicyReplayMemory
from single_agent.Model_common import ActorNetwork, CriticNetwork
from common.utils import index_to_one_hot, to_tensor_var, VideoRecorder
from sandra_llm_decider import SanDRALLMConfig, SanDRALLMDecider


class MAPPO:
    """
    An multi-agent learned with PPO
    reference: https://github.com/ChenglongChen/pytorch-DRL
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=1, target_tau=1.,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=20,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.0001, critic_lr=0.0001, test_seeds=0,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 use_cuda=True, traffic_density=1, reward_type="global_R",
                 sandra_enabled=False,
                 sandra_risk_threshold=1.0,
                 sandra_openai_api_key=None,
                 sandra_openai_api_base=None,
                 sandra_openai_base_url=None,
                 sandra_openai_model_name=None,
                 sandra_temperature=0.2,
                 sandra_request_timeout=12.0,
                 sandra_max_retries=1,
                 sandra_log_decisions=True,
                 sandra_use_safety_filter=True,
                 sandra_rewrite_penalty=0.25,
                 sandra_min_risk_improve=0.0,
                 sandra_preview_horizon=3,
                 sandra_block_on_crash=True,
                 sandra_log_candidates=False,
                 sandra_top_k=3,
                 sandra_formal_check_enabled=True,
                 sandra_rule_min_ttc=2.0,
                 sandra_rule_min_headway_time=1.5,
                 sandra_rule_lane_change_front_gap=12.0,
                 sandra_rule_lane_change_rear_gap=8.0,
                 sandra_rule_verify_horizon_s=2.0,
                 sandra_use_shapely=True,
                 sandra_use_py_rss=True,
                 risk_alert_enabled=True, risk_alert_threshold=1.0,
                 render_train=False, render_mode="human"):

        assert traffic_density in [1, 2, 3]
        assert reward_type in ["greedy", "regionalR", "global_R"]
        self.reward_type = reward_type
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_state, self.action_mask = self.env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.test_seeds = test_seeds
        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.traffic_density = traffic_density
        self.sandra_enabled = bool(sandra_enabled)
        self.sandra_risk_threshold = float(sandra_risk_threshold)
        self.sandra_log_decisions = bool(sandra_log_decisions)
        self.sandra_use_safety_filter = bool(sandra_use_safety_filter)
        self.sandra_rewrite_penalty = float(sandra_rewrite_penalty)
        self.sandra_min_risk_improve = float(sandra_min_risk_improve)
        self.sandra_preview_horizon = max(1, int(sandra_preview_horizon))
        self.sandra_block_on_crash = bool(sandra_block_on_crash)
        self.sandra_log_candidates = bool(sandra_log_candidates)
        self.sandra_top_k = max(1, int(sandra_top_k))
        self.sandra_formal_check_enabled = bool(sandra_formal_check_enabled)
        self.sandra_rule_min_ttc = float(sandra_rule_min_ttc)
        self.sandra_rule_min_headway_time = float(sandra_rule_min_headway_time)
        self.sandra_rule_lane_change_front_gap = float(sandra_rule_lane_change_front_gap)
        self.sandra_rule_lane_change_rear_gap = float(sandra_rule_lane_change_rear_gap)
        self.sandra_rule_verify_horizon_s = float(sandra_rule_verify_horizon_s)
        self.sandra_use_shapely = bool(sandra_use_shapely)
        self.sandra_use_py_rss = bool(sandra_use_py_rss)
        self.risk_alert_enabled = bool(risk_alert_enabled)
        self.risk_alert_threshold = float(risk_alert_threshold)
        self.memory = OnPolicyReplayMemory(memory_capacity)
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.use_cuda = use_cuda and th.cuda.is_available()
        self.roll_out_n_steps = roll_out_n_steps
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.clip_param = clip_param
        self.render_train = render_train
        self.render_mode = render_mode

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                  self.action_dim, self.actor_output_act)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)
        # to ensure target network and learning network has the same weights
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

        self.episode_rewards = [0]
        self.average_speed = [0]
        self.epoch_steps = [0]
        init_agents = len(self.env.controlled_vehicles)
        self.current_episode_risk_sum = 0.0
        self.current_episode_risk_steps = 0
        self.current_episode_agent_risk_sum = np.zeros((init_agents,), dtype=np.float32)
        self.episode_mean_risk = []
        self.episode_agent_risk = []
        self.last_step_agent_risk = np.zeros((init_agents,), dtype=np.float32)
        self.sandra_override_steps = 0

        sandra_cfg = SanDRALLMConfig(
            enabled=self.sandra_enabled,
            risk_threshold=self.sandra_risk_threshold,
            api_key=sandra_openai_api_key,
            api_base=sandra_openai_api_base,
            base_url=sandra_openai_base_url,
            model_name=sandra_openai_model_name,
            temperature=float(sandra_temperature),
            request_timeout=float(sandra_request_timeout),
            max_retries=int(sandra_max_retries),
            log_decisions=self.sandra_log_decisions,
            top_k=self.sandra_top_k,
            formal_check_enabled=self.sandra_formal_check_enabled,
            rule_min_ttc=self.sandra_rule_min_ttc,
            rule_min_headway_time=self.sandra_rule_min_headway_time,
            rule_lane_change_front_gap=self.sandra_rule_lane_change_front_gap,
            rule_lane_change_rear_gap=self.sandra_rule_lane_change_rear_gap,
            rule_verify_horizon_s=self.sandra_rule_verify_horizon_s,
            use_shapely=self.sandra_use_shapely,
            use_py_rss=self.sandra_use_py_rss,
        )
        self.sandra_decider = SanDRALLMDecider(sandra_cfg)

    # agent interact with the environment to collect experience
    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            print("正在重置环境 (env.reset)... 这可能需要一点时间")
            self.env_state, self.action_mask = self.env.reset()
            print("环境重置成功！")
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        done = True
        average_speed = 0

        self.n_agents = len(self.env.controlled_vehicles)
        if self.current_episode_agent_risk_sum.shape[0] != self.n_agents:
            self.current_episode_agent_risk_sum = np.zeros((self.n_agents,), dtype=np.float32)
            self.last_step_agent_risk = np.zeros((self.n_agents,), dtype=np.float32)
        if self.render_train:
            try:
                try:
                    self.env.render(mode=self.render_mode)
                except TypeError:
                    self.env.render()
            except Exception as e:
                if "video system not initialized" in str(e):
                    print(">>> 检测到 pygame 已关闭，跳过本次渲染。")
                    if hasattr(self.env, "viewer"):
                        self.env.viewer = None
                else:
                    raise

        # take n steps
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = self.exploration_action(self.env_state, self.action_mask, self.n_agents)
            risk_before_action = self._get_current_risk_vector(self.n_agents)
            action = self._maybe_apply_sandra_overrides(action, self.action_mask, risk_before_action)
            next_state, global_reward, done, info = self.env.step(tuple(action))
            self.action_mask = info["action_mask"]
            if self.render_train:
                try:
                    try:
                        self.env.render(mode=self.render_mode)
                    except TypeError:
                        self.env.render()
                except Exception as e:
                    if "video system not initialized" in str(e):
                        print(">>> 检测到 pygame 已关闭，跳过本次渲染。")
                        if hasattr(self.env, "viewer"):
                            self.env.viewer = None
                    else:
                        raise
            executed_action = self._extract_executed_action(info, action, self.n_agents)
            actions.append([index_to_one_hot(a, self.action_dim) for a in executed_action])
            self.episode_rewards[-1] += global_reward
            self.epoch_steps[-1] += 1
            if self.reward_type == "greedy":
                reward = info["agents_rewards"]
            elif self.reward_type == "regionalR":
                reward = info["regional_rewards"]
            elif self.reward_type == "global_R":
                reward = [global_reward] * self.n_agents
            rewards.append(reward)
            average_speed += info["average_speed"]
            step_risk = self._extract_agent_risk_vector(info, "agent_risk_pre", self.n_agents)
            self.current_episode_risk_sum += float(np.mean(step_risk))
            self.current_episode_risk_steps += 1
            self.current_episode_agent_risk_sum += step_risk
            self.last_step_agent_risk = step_risk
            self._log_high_risk_agents(step_risk)
            final_state = next_state
            self.env_state = next_state

            self.n_steps += 1
            if done:
                self.env_state, self.action_mask = self.env.reset()
                break

        # discount reward
        if done:
            final_value = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
            self.episode_rewards.append(0)
            self.average_speed[-1] = average_speed / self.epoch_steps[-1]
            self.average_speed.append(0)
            self.epoch_steps.append(0)
            if self.current_episode_risk_steps > 0:
                self.episode_mean_risk.append(self.current_episode_risk_sum / self.current_episode_risk_steps)
                self.episode_agent_risk.append(
                    (self.current_episode_agent_risk_sum / self.current_episode_risk_steps).tolist()
                )
            else:
                self.episode_mean_risk.append(0.0)
                self.episode_agent_risk.append([0.0] * self.n_agents)
            self.current_episode_risk_sum = 0.0
            self.current_episode_risk_steps = 0
            self.current_episode_agent_risk_sum = np.zeros((self.n_agents,), dtype=np.float32)
        else:
            self.episode_done = False
            final_action = self.action(final_state, self.action_mask, self.n_agents)
            final_value = self.value(final_state, final_action)

        if self.reward_scale > 0:
            rewards = np.array(rewards) / self.reward_scale

        rewards = np.array(rewards, dtype=np.float32)
        for agent_id in range(self.n_agents):
            rewards[:, agent_id] = self._discount_reward(rewards[:, agent_id], final_value[agent_id])

        rewards = rewards.tolist()
        self.memory.push(states, actions, rewards)

    def _extract_agent_risk_vector(self, info, key, n_agents):
        risk = np.asarray(info.get(key, np.zeros((n_agents,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if risk.shape[0] < n_agents:
            risk = np.pad(risk, (0, n_agents - risk.shape[0]), mode="constant")
        elif risk.shape[0] > n_agents:
            risk = risk[:n_agents]
        return risk

    def _extract_executed_action(self, info, fallback_action, n_agents):
        executed = info.get("new_action", fallback_action)
        if isinstance(executed, np.ndarray):
            executed = executed.reshape(-1).tolist()
        elif isinstance(executed, tuple):
            executed = list(executed)
        elif isinstance(executed, list):
            executed = list(executed)
        else:
            executed = [int(executed)]
        if len(executed) < n_agents:
            executed.extend([1] * (n_agents - len(executed)))
        elif len(executed) > n_agents:
            executed = executed[:n_agents]
        return [int(a) for a in executed]

    def _extract_bool_vector(self, raw, n_agents):
        if isinstance(raw, np.ndarray):
            values = raw.reshape(-1).tolist()
        elif isinstance(raw, tuple):
            values = list(raw)
        elif isinstance(raw, list):
            values = list(raw)
        elif raw is None:
            values = []
        else:
            values = [raw]
        if len(values) < n_agents:
            values.extend([False] * (n_agents - len(values)))
        elif len(values) > n_agents:
            values = values[:n_agents]
        return [bool(v) for v in values]

    def _get_current_risk_vector(self, n_agents):
        env_obj = getattr(self.env, "unwrapped", self.env)
        if hasattr(env_obj, "_evaluate_controlled_vehicle_risk"):
            try:
                current = env_obj._evaluate_controlled_vehicle_risk()
                risk = np.asarray(
                    current.get("total", np.zeros((n_agents,), dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
                if risk.shape[0] < n_agents:
                    risk = np.pad(risk, (0, n_agents - risk.shape[0]), mode="constant")
                elif risk.shape[0] > n_agents:
                    risk = risk[:n_agents]
                return risk
            except Exception:
                pass
        if self.last_step_agent_risk.shape[0] == n_agents:
            return self.last_step_agent_risk.copy()
        return np.zeros((n_agents,), dtype=np.float32)

    def _preview_joint_action(self, joint_actions, base_np_state=None, base_py_state=None, horizon=1):
        env_obj = getattr(self.env, "unwrapped", self.env)
        if not hasattr(env_obj, "step"):
            return None

        np_state_snapshot = np.random.get_state()
        py_state_snapshot = random.getstate()
        try:
            if base_np_state is not None:
                np.random.set_state(base_np_state)
            if base_py_state is not None:
                random.setstate(base_py_state)

            env_shadow = copy.deepcopy(env_obj)
            n_agents = len(joint_actions)
            horizon = max(1, int(horizon))

            trial_actions = [int(a) for a in joint_actions]
            executed_first = None
            risk_seq = []
            last_info = {}
            scenario_crashed = False

            for _ in range(horizon):
                _, _, done, info = env_shadow.step(tuple(int(a) for a in trial_actions))
                last_info = info
                executed = self._extract_executed_action(info, trial_actions, n_agents)
                if executed_first is None:
                    executed_first = executed
                post_risk = self._extract_agent_risk_vector(info, "agent_risk_post", n_agents)
                risk_seq.append(post_risk)

                if bool(info.get("scenario_crashed", False)):
                    scenario_crashed = True
                if done:
                    break
                trial_actions = executed

            if executed_first is None:
                return None

            if len(risk_seq) == 0:
                risk_matrix = np.zeros((1, n_agents), dtype=np.float32)
            else:
                risk_matrix = np.vstack(risk_seq).astype(np.float32)

            vehicle_crashed = [False] * n_agents
            vehicles = getattr(env_shadow, "controlled_vehicles", [])
            for i in range(min(n_agents, len(vehicles))):
                vehicle_crashed[i] = bool(getattr(vehicles[i], "crashed", False))

            agents_dones = self._extract_bool_vector(last_info.get("agents_dones", []), n_agents)
            done_reason = str(last_info.get("done_reason", "running"))
            if done_reason == "scenario_crashed":
                scenario_crashed = True
            return {
                "executed": executed_first,
                "post_risk": risk_matrix[-1],
                "max_risk": np.max(risk_matrix, axis=0),
                "risk_matrix": risk_matrix,
                "info": last_info,
                "scenario_crashed": bool(scenario_crashed),
                "done_reason": done_reason,
                "agents_dones": agents_dones,
                "agent_crashed": vehicle_crashed,
            }
        except Exception:
            return None
        finally:
            np.random.set_state(np_state_snapshot)
            random.setstate(py_state_snapshot)

    def _choose_sandra_action_with_filter(
        self,
        joint_actions,
        agent_id,
        llm_action,
        available_actions,
        risk_before,
        ranked_actions=None,
        formal_results=None,
    ):
        if not self.sandra_use_safety_filter:
            return int(llm_action), "llm_direct", None

        # Keep deterministic/random-consistent previews.
        base_np_state = np.random.get_state()
        base_py_state = random.getstate()

        ranked_seed = []
        if ranked_actions is not None:
            for a in ranked_actions:
                ai = int(a)
                if ai in available_actions and ai not in ranked_seed:
                    ranked_seed.append(ai)
        if int(llm_action) in available_actions and int(llm_action) not in ranked_seed:
            ranked_seed.append(int(llm_action))
        for a in available_actions:
            ai = int(a)
            if ai not in ranked_seed:
                ranked_seed.append(ai)
        candidates = ranked_seed

        formal_map = {}
        if isinstance(formal_results, list):
            for item in formal_results:
                if not isinstance(item, dict):
                    continue
                try:
                    a = int(item.get("action"))
                except Exception:
                    continue
                formal_map[a] = item
        evaluations = []
        for candidate in candidates:
            trial_joint = [int(a) for a in joint_actions]
            trial_joint[agent_id] = int(candidate)
            preview = self._preview_joint_action(
                trial_joint,
                base_np_state=base_np_state,
                base_py_state=base_py_state,
                horizon=self.sandra_preview_horizon,
            )
            if preview is None:
                continue
            executed_agent = int(preview["executed"][agent_id])
            predicted_risk = float(preview["post_risk"][agent_id])
            worst_risk = float(preview["max_risk"][agent_id])
            rewrite = 1.0 if executed_agent != int(candidate) else 0.0
            scenario_crashed = bool(preview.get("scenario_crashed", False))
            agent_crashed = bool(preview.get("agent_crashed", [False] * (agent_id + 1))[agent_id])
            hard_unsafe = bool(self.sandra_block_on_crash and (scenario_crashed or agent_crashed))
            formal_item = formal_map.get(int(candidate), None)
            formal_unsafe = False
            formal_score = 0.0
            formal_violations = []
            if isinstance(formal_item, dict):
                formal_unsafe = not bool(formal_item.get("safe", True))
                formal_score = float(formal_item.get("score", 0.0))
                formal_violations = list(formal_item.get("violated_rules", []))

            # Use a conservative score: prioritize lower worst-case risk, then final risk.
            score = (
                worst_risk
                + 0.3 * predicted_risk
                + self.sandra_rewrite_penalty * rewrite
                + 0.3 * formal_score
            )
            if hard_unsafe:
                score += 1000.0
            if formal_unsafe:
                score += 100.0
            item = {
                "candidate": int(candidate),
                "executed": int(executed_agent),
                "predicted_risk": predicted_risk,
                "worst_risk": worst_risk,
                "score": score,
                "rewrite": bool(rewrite > 0.0),
                "scenario_crashed": scenario_crashed,
                "agent_crashed": agent_crashed,
                "hard_unsafe": hard_unsafe,
                "formal_unsafe": formal_unsafe,
                "formal_score": formal_score,
                "formal_violations": formal_violations,
                "ltl": str(formal_item.get("ltl", "")) if isinstance(formal_item, dict) else "",
                "done_reason": str(preview.get("done_reason", "running")),
                "horizon": int(preview.get("risk_matrix", np.zeros((1, 1))).shape[0]),
                "llm_ranked": bool(int(candidate) in [int(x) for x in ranked_seed[: max(1, self.sandra_top_k)]]),
            }
            evaluations.append(item)

        if len(evaluations) == 0:
            return int(llm_action), "llm_no_preview", None

        safe_candidates = [x for x in evaluations if not x["hard_unsafe"] and not x["formal_unsafe"]]
        if len(safe_candidates) > 0:
            best = min(safe_candidates, key=lambda x: x["score"])
            source = "safety_filter"
        else:
            semi_safe = [x for x in evaluations if not x["hard_unsafe"]]
            if len(semi_safe) > 0:
                best = min(semi_safe, key=lambda x: x["score"])
                source = "safety_filter_no_formal_safe"
            else:
            # Emergency fallback when all candidates are predicted unsafe.
            # Prefer decelerate/idle executable actions if present.
                emergency_order = [4, 1, 3, 0, 2]
                best = None
                for preferred in emergency_order:
                    preferred_items = [x for x in evaluations if int(x["executed"]) == int(preferred)]
                    if len(preferred_items) > 0:
                        best = min(preferred_items, key=lambda x: x["score"])
                        break
                if best is None:
                    best = min(evaluations, key=lambda x: x["score"])
                source = "safety_filter_no_safe"

        if (risk_before - best["predicted_risk"]) < self.sandra_min_risk_improve:
            if len(safe_candidates) > 0:
                rl_candidate = int(joint_actions[agent_id])
                rl_items = [x for x in safe_candidates if int(x["candidate"]) == rl_candidate]
                if len(rl_items) > 0:
                    rl_best = min(rl_items, key=lambda x: x["score"])
                    if float(rl_best["score"]) <= float(best["score"]) + 1e-6:
                        best = rl_best
                        source = f"{source}_keep_rl"
            source = f"{source}_no_improve"

        best["all_candidates"] = evaluations
        return int(best["executed"]), source, best


    def _maybe_apply_sandra_overrides(self, rl_actions, action_mask, risk_vector):
        if not self.sandra_enabled:
            return rl_actions
        env_obj = getattr(self.env, "unwrapped", self.env)
        if not hasattr(env_obj, "controlled_vehicles"):
            return rl_actions

        final_actions = [int(a) for a in rl_actions]
        for agent_id in range(min(len(final_actions), len(risk_vector))):
            risk_value = float(risk_vector[agent_id])
            if risk_value < self.sandra_risk_threshold:
                continue
            agent_mask = None
            if action_mask is not None and agent_id < len(action_mask):
                agent_mask = action_mask[agent_id]
            available_actions = self.sandra_decider.available_discrete_actions(agent_mask)
            llm_action, llm_source, llm_meta = self.sandra_decider.decide_action(
                env=env_obj,
                car_index=agent_id,
                risk_value=risk_value,
                action_mask=agent_mask,
                rl_action=final_actions[agent_id],
            )
            ranked_actions = [int(a) for a in llm_meta.get("ranked_actions", [llm_action])]
            formal_results = llm_meta.get("verification_results", [])
            if self.sandra_log_decisions:
                print(
                    "[SANDRA RANK] car_no=%d backend=%s ranked=%s selected=%d source=%s"
                    % (
                        agent_id + 1,
                        str(llm_meta.get("backend", "unknown")),
                        str(ranked_actions),
                        int(llm_meta.get("selected_action", llm_action)),
                        llm_source,
                    ),
                    flush=True,
                )
                for item in formal_results:
                    if not isinstance(item, dict):
                        continue
                    print(
                        "[SANDRA VERIFY] car_no=%d rank=%d action=%d safe=%s score=%.4f violations=%s"
                        % (
                            agent_id + 1,
                            int(item.get("rank", -1)),
                            int(item.get("action", -1)),
                            str(bool(item.get("safe", False))),
                            float(item.get("score", 0.0)),
                            str(item.get("violated_rules", [])),
                        ),
                        flush=True,
                    )
            sandra_action, filter_source, filter_detail = self._choose_sandra_action_with_filter(
                joint_actions=final_actions,
                agent_id=agent_id,
                llm_action=llm_action,
                available_actions=available_actions,
                risk_before=risk_value,
                ranked_actions=ranked_actions,
                formal_results=formal_results,
            )
            source = f"{llm_source}+{filter_source}"
            if int(sandra_action) != int(final_actions[agent_id]):
                self.sandra_override_steps += 1
                if self.sandra_log_decisions:
                    print(
                        "[SANDRA SWITCH] episode=%d step=%d car_no=%d risk=%.4f rl_action=%d sandra_action=%d source=%s"
                        % (
                            self.n_episodes + 1,
                            int(self.epoch_steps[-1]) if len(self.epoch_steps) > 0 else 0,
                            agent_id + 1,
                            risk_value,
                            int(final_actions[agent_id]),
                            int(sandra_action),
                            source,
                        ),
                        flush=True,
                    )
            if self.sandra_log_decisions and filter_detail is not None:
                print(
                    "[SANDRA FILTER] car_no=%d risk_before=%.4f selected=%d predicted_risk=%.4f worst_risk=%.4f rewrite=%s crash=%s formal_unsafe=%s score=%.4f source=%s"
                    % (
                        agent_id + 1,
                        risk_value,
                        int(filter_detail["executed"]),
                        float(filter_detail["predicted_risk"]),
                        float(filter_detail.get("worst_risk", filter_detail["predicted_risk"])),
                        str(filter_detail["rewrite"]),
                        str(bool(filter_detail.get("hard_unsafe", False))),
                        str(bool(filter_detail.get("formal_unsafe", False))),
                        float(filter_detail["score"]),
                        filter_source,
                    ),
                    flush=True,
                )
            if self.sandra_log_decisions and self.sandra_log_candidates and filter_detail is not None:
                for cand in filter_detail.get("all_candidates", []):
                    print(
                        "[SANDRA CAND] car_no=%d cand=%d exec=%d pred=%.4f worst=%.4f rewrite=%s crash=%s formal=%s score=%.4f done=%s hz=%d ltl=%s violations=%s"
                        % (
                            agent_id + 1,
                            int(cand["candidate"]),
                            int(cand["executed"]),
                            float(cand["predicted_risk"]),
                            float(cand.get("worst_risk", cand["predicted_risk"])),
                            str(cand.get("rewrite", False)),
                            str(bool(cand.get("hard_unsafe", False))),
                            str(bool(cand.get("formal_unsafe", False))),
                            float(cand["score"]),
                            str(cand.get("done_reason", "running")),
                            int(cand.get("horizon", 1)),
                            str(cand.get("ltl", "")),
                            str(cand.get("formal_violations", [])),
                        ),
                        flush=True,
                    )
            final_actions[agent_id] = int(sandra_action)
        return final_actions

    def _log_high_risk_agents(self, step_risk):
        if not self.risk_alert_enabled:
            return
        if step_risk.size == 0:
            return
        risk_indices = np.where(step_risk >= self.risk_alert_threshold)[0]
        if risk_indices.size == 0:
            return

        current_episode = self.n_episodes + 1
        current_step = int(self.epoch_steps[-1]) if len(self.epoch_steps) > 0 else 0
        for idx in risk_indices.tolist():
            vehicle_slot = int(idx)
            vehicle_no = vehicle_slot + 1
            vehicle_id = vehicle_slot
            lane_index = "N/A"
            position_x = 0.0
            position_y = 0.0
            if vehicle_slot < len(self.env.controlled_vehicles):
                vehicle = self.env.controlled_vehicles[vehicle_slot]
                vehicle_id = int(getattr(vehicle, "id", vehicle_slot))
                lane_index = str(getattr(vehicle, "lane_index", "N/A"))
                try:
                    position_x = float(vehicle.position[0])
                    position_y = float(vehicle.position[1])
                except Exception:
                    pass
            print(
                "[RISK ALERT] episode=%d step=%d car_no=%d vehicle_id=%d risk=%.4f threshold=%.4f lane=%s pos=(%.2f, %.2f)"
                % (
                    current_episode,
                    current_step,
                    vehicle_no,
                    vehicle_id,
                    float(step_risk[vehicle_slot]),
                    self.risk_alert_threshold,
                    lane_index,
                    position_x,
                    position_y,
                ),
                flush=True,
            )

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)

        for agent_id in range(self.n_agents):
            # update actor network
            self.actor_optimizer.zero_grad()
            values = self.critic_target(states_var[:, agent_id, :], actions_var[:, agent_id, :]).detach()
            advantages = rewards_var[:, agent_id, :] - values

            action_log_probs = self.actor(states_var[:, agent_id, :])
            action_log_probs = th.sum(action_log_probs * actions_var[:, agent_id, :], 1)
            old_action_log_probs = self.actor_target(states_var[:, agent_id, :]).detach()
            old_action_log_probs = th.sum(old_action_log_probs * actions_var[:, agent_id, :], 1)
            ratio = th.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            # PPO's pessimistic surrogate (L^CLIP)
            actor_loss = -th.mean(th.min(surr1, surr2))
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # update critic network
            self.critic_optimizer.zero_grad()
            target_values = rewards_var[:, agent_id, :]
            values = self.critic(states_var[:, agent_id, :], actions_var[:, agent_id, :])
            if self.critic_loss == "huber":
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            else:
                critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        # update actor target network and critic target network
        if self.n_episodes % self.target_update_steps == 0 and self.n_episodes > 0:
            self._soft_update_target(self.actor_target, self.actor)
            self._soft_update_target(self.critic_target, self.critic)

    # predict softmax action based on state
    def _softmax_action(self, state, action_mask, n_agents):
        state_var = to_tensor_var([state], self.use_cuda)

        softmax_action = []
        for agent_id in range(n_agents):
            logits = self.actor(state_var[:, agent_id, :])
            pi = th.softmax(logits, dim=-1).detach().cpu().numpy()[0]

            if action_mask is not None:
                mask = np.asarray(action_mask[agent_id], dtype=np.float32)
                pi = pi * mask

                if pi.sum() <= 1e-8:
                # 如果 mask 后全为 0，退化成对合法动作均匀采样
                    valid_count = mask.sum()
                    if valid_count > 0:
                        pi = mask / valid_count
                    else:
                        pi = np.ones_like(pi) / len(pi)
                else:
                    pi = pi / pi.sum()
 
            softmax_action.append(pi)

        return softmax_action

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state, action_mask, n_agents):
        softmax_actions = self._softmax_action(state, action_mask, n_agents)
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    # choose an action based on state for execution
    def action(self, state, action_mask, n_agents):
        softmax_actions = self._softmax_action(state, action_mask, n_agents)
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action], self.use_cuda)

        values = [0] * self.n_agents
        for agent_id in range(self.n_agents):
            value_var = self.critic(state_var[:, agent_id, :], action_var[:, agent_id, :])

            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values

    # evaluation the learned agent
    def evaluation(self, env, output_dir, eval_episodes=1, is_train=True):
        rewards = []
        infos = []
        avg_speeds = []
        steps = []
        vehicle_speed = []
        vehicle_position = []
        video_recorder = None
        seeds = [int(s) for s in self.test_seeds.split(',')]

        for i in range(eval_episodes):
            avg_speed = 0
            step = 0
            rewards_i = []
            infos_i = []
            done = False
            if is_train:
                if self.traffic_density == 1:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 1)
                elif self.traffic_density == 2:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 2)
                elif self.traffic_density == 3:
                    state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i], num_CAV=i + 4)
            else:
                state, action_mask = env.reset(is_training=False, testing_seeds=seeds[i])

            n_agents = len(env.controlled_vehicles)
            rendered_frame = env.render(mode="rgb_array")
            video_filename = os.path.join(output_dir,
                                          "testing_episode{}".format(self.n_episodes + 1) + '_{}'.format(i) +
                                          '.mp4')
            # Init video recording
            if video_filename is not None:
                print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape,
                                                                      5))
                video_recorder = VideoRecorder(video_filename,
                                               frame_size=rendered_frame.shape, fps=5)
                video_recorder.add_frame(rendered_frame)
            else:
                video_recorder = None

            while not done:
                step += 1
                action = self.action(state, action_mask, n_agents)
                state, reward, done, info = env.step(action)
                action_mask = info["action_mask"]
                avg_speed += info["average_speed"]
                rendered_frame = env.render(mode="rgb_array")
                if video_recorder is not None:
                    video_recorder.add_frame(rendered_frame)

                rewards_i.append(reward)
                infos_i.append(info)

            vehicle_speed.append(info["vehicle_speed"])
            vehicle_position.append(info["vehicle_position"])
            rewards.append(rewards_i)
            infos.append(infos_i)
            steps.append(step)
            avg_speeds.append(avg_speed / step)

        if video_recorder is not None:
            video_recorder.release()
        # 训练过程中做评估时，不要关闭 env_eval，
        # 否则会触发 pygame.quit()，把训练窗口也一起关掉
        if not is_train:
            env.close()
        else:
            # 只把评估环境的 viewer 置空，避免下次评估复用旧 viewer
            if hasattr(env, "viewer") and env.viewer is not None:
                try:
                    env.viewer = None
                except Exception:
                    pass
        return rewards, (vehicle_speed, vehicle_position), steps, avg_speeds

    # discount roll out rewards
    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    # soft update the actor target network or critic target network
    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = th.load(file_path)
            print('Checkpoint loaded: {}'.format(file_path))
            # logging.info('Checkpoint loaded: {}'.format(file_path))
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.actor.train()
            else:
                self.actor.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        th.save({'global_step': global_step,
                 'model_state_dict': self.actor.state_dict(),
                 'optimizer_state_dict': self.actor_optimizer.state_dict()},
                file_path)
