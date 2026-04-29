import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np
from copy import deepcopy

from ..Agent_common import Agent
from ..Model_common import ActorNetwork, CriticNetwork
from ..utils_common import index_to_one_hot, to_tensor_var


class PPO(Agent):
    """
    An agent learned with PPO using Advantage Actor-Critic framework
    - Actor takes state as input
    - Critic takes both state and action as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy
    - adam seems better than rmsprop for ppo
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=1, target_tau=1.,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="adam", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):
        super(PPO, self).__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda)

        self.roll_out_n_steps = roll_out_n_steps
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.clip_param = clip_param

        self.actor = ActorNetwork(
            self.state_dim, self.actor_hidden_size,
            self.action_dim, self.actor_output_act
        )
        self.critic = CriticNetwork(
            self.state_dim, self.action_dim,
            self.critic_hidden_size, 1
        )

        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)
        else:
            raise ValueError(f"Unsupported optimizer_type: {self.optimizer_type}")

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

        self.env_state = self._normalize_state(self.env_state)

    def _normalize_state(self, state):
        if isinstance(state, tuple):
            if len(state) >= 1:
                state = state[0]
        return state

    def _reset_env(self, env=None):
        if env is None:
            env = self.env
        out = env.reset()
        return self._normalize_state(out)

    def _step_env(self, env, action=None):
        # 兼容两种调用方式：
        # self._step_env(action)
        # self._step_env(env, action)
        if action is None:
            action = env
            env = self.env

        out = env.step(action)

        if len(out) == 5:
            next_state, reward, terminated, truncated, info = out
            done = terminated or truncated
        elif len(out) == 4:
            next_state, reward, done, info = out
        else:
            raise ValueError(f"Unexpected number of values returned by env.step(): {len(out)}")

        next_state = self._normalize_state(next_state)
        return next_state, reward, done, info

    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self._reset_env()
            self.n_steps = 0

        self.env_state = self._normalize_state(self.env_state)

        states = []
        actions = []
        rewards = []

        done = False
        final_state = self.env_state

        for _ in range(self.roll_out_n_steps):
            current_state = self._normalize_state(self.env_state)
            states.append(current_state)

            action = self.exploration_action(current_state)
            next_state, reward, done, _ = self._step_env(action)

            actions.append(action)

            if done and self.done_penalty is not None:
                reward = self.done_penalty
            rewards.append(reward)

            final_state = next_state
            self.env_state = next_state

            if done:
                self.env_state = self._reset_env()
                break

        if done:
            final_value = 0.0
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.episode_done = False
            final_action = self.action(final_state)
            final_value = self.value(final_state, final_action)

        rewards = self._discount_reward(rewards, final_value)
        self.n_steps += 1
        self.memory.push(states, actions, rewards)

    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            return

        batch = self.memory.sample(self.batch_size)

        states = [self._normalize_state(s) for s in batch.states]
        states_var = to_tensor_var(np.array(states), self.use_cuda).view(-1, self.state_dim)

        one_hot_actions = index_to_one_hot(batch.actions, self.action_dim)
        actions_var = to_tensor_var(one_hot_actions, self.use_cuda).view(-1, self.action_dim)

        rewards_var = to_tensor_var(np.array(batch.rewards), self.use_cuda).view(-1, 1)

        self.actor_optimizer.zero_grad()
        values = self.critic_target(states_var, actions_var).detach()
        advantages = rewards_var - values

        action_log_probs = self.actor(states_var)
        action_log_probs = th.sum(action_log_probs * actions_var, dim=1)

        old_action_log_probs = self.actor_target(states_var).detach()
        old_action_log_probs = th.sum(old_action_log_probs * actions_var, dim=1)

        ratio = th.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * advantages.squeeze(-1)
        surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages.squeeze(-1)

        actor_loss = -th.mean(th.min(surr1, surr2))
        actor_loss.backward()

        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        target_values = rewards_var
        values = self.critic(states_var, actions_var)

        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)

        critic_loss.backward()

        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.critic_optimizer.step()

        if self.n_steps % self.target_update_steps == 0 and self.n_steps > 0:
            super(PPO, self)._soft_update_target(self.actor_target, self.actor)
            super(PPO, self)._soft_update_target(self.critic_target, self.critic)

    def _softmax_action(self, state):
        state = self._normalize_state(state)
        state_var = to_tensor_var(np.array([state]), self.use_cuda)
        softmax_action_var = th.exp(self.actor(state_var))

        if self.use_cuda:
            softmax_action = softmax_action_var.data.cpu().numpy()[0]
        else:
            softmax_action = softmax_action_var.data.numpy()[0]

        return softmax_action

    def exploration_action(self, state):
        softmax_action = self._softmax_action(state)
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1.0 * self.n_steps / self.epsilon_decay)

        if np.random.rand() < epsilon:
            action = np.random.choice(self.action_dim)
        else:
            action = np.argmax(softmax_action)

        return action

    def action(self, state):
        softmax_action = self._softmax_action(state)
        action = np.argmax(softmax_action)
        return action

    def value(self, state, action):
        state = self._normalize_state(state)
        state_var = to_tensor_var(np.array([state]), self.use_cuda)

        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var(np.array([action]), self.use_cuda)

        value_var = self.critic(state_var, action_var)

        if self.use_cuda:
            value = value_var.data.cpu().numpy()[0]
        else:
            value = value_var.data.numpy()[0]

        return value