from MAPPO import MAPPO
from common.utils import agg_double_list, copy_file_ppo, init_dir

import os
import sys

# 把本地 highway-env 放到 sys.path 最前面，确保优先导入项目里的版本
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HIGHWAY_ENV_PATH = os.path.abspath(os.path.join(CURRENT_DIR, "..", "highway-env"))
if HIGHWAY_ENV_PATH not in sys.path:
    sys.path.insert(0, HIGHWAY_ENV_PATH)

import gym
import numpy as np

# ===== NumPy 2.0 compatibility patch =====
if not hasattr(np, "bool"):
    np.bool = np.bool_
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = np.complex128
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "long"):
    np.long = int
# =========================================

import matplotlib.pyplot as plt
import highway_env
import argparse
import configparser
from datetime import datetime

print(f">>> Added local path to sys.path: {HIGHWAY_ENV_PATH}", flush=True)
print(f">>> Successfully imported highway_env from: {highway_env.__file__}", flush=True)


def parse_args():
    """
    Description for this experiment:
        + hard: 7-steps, curriculum
        + seed = 0
    """
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs_ppo.ini'
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using MA2C'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, required=False,
                        default='evaluate', help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--model-dir', type=str, required=False,
                        default='results/Jan-01_11_22_31', help="pretrained model path")
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(0, 600, 20)]),
                        help="random seeds for evaluation, split by ,")
    parser.add_argument('--render-train', action='store_true',
                        help='Render the highway-env window during training')
    parser.add_argument('--render-mode', type=str, required=False, default='human',
                        choices=['human', 'rgb_array'],
                        help='Render mode used during training')
    args = parser.parse_args()
    return args


def train(args):
    base_dir = args.base_dir
    config_dir = args.config_dir
    config = configparser.ConfigParser()
    config.read(config_dir)

    # create an experiment folder
    now = datetime.utcnow().strftime("%b-%d_%H-%M-%S")
    output_dir = base_dir + now

    print(f">>> [Step 1] 正在初始化目录: {output_dir}", flush=True)
    dirs = init_dir(output_dir)
    copy_file_ppo(dirs['configs'])

    if os.path.exists(args.model_dir):
        model_dir = args.model_dir
    else:
        model_dir = dirs['models']

    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')
    TARGET_UPDATE_STEPS = config.getint('MODEL_CONFIG', 'TARGET_UPDATE_STEPS')
    TARGET_TAU = config.getfloat('MODEL_CONFIG', 'TARGET_TAU')

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    MAX_EPISODES = config.getint('TRAIN_CONFIG', 'MAX_EPISODES')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    EVAL_INTERVAL = config.getint('TRAIN_CONFIG', 'EVAL_INTERVAL')
    EVAL_EPISODES = config.getint('TRAIN_CONFIG', 'EVAL_EPISODES')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')

    # init env
    print(">>> [Step 2] 正在初始化环境 (gym.make)...", flush=True)
    env = gym.make('merge-multi-agent-v0', disable_env_checker=True)
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['LANE_CHANGE_COST'] = config.getfloat('ENV_CONFIG', 'LANE_CHANGE_COST')
    env.config['MERGE_SUCCESS_REWARD'] = config.getfloat('ENV_CONFIG', 'MERGE_SUCCESS_REWARD')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    if args.render_train:
        env.config['offscreen_rendering'] = False
        env.config['real_time_rendering'] = True

    assert env.T % ROLL_OUT_N_STEPS == 0
    print(">>> 环境初始化完成。", flush=True)

    # 保持下面这一长串参数不动
    print(">>> [Step 3] 正在创建 MAPPO 实例...", flush=True)
    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds

    print(">>> [Step 2.5] 正在初始化评估环境 (env_eval)...", flush=True)
    env_eval = gym.make('merge-multi-agent-v0', disable_env_checker=True)
    env_eval.config['offscreen_rendering'] = True
    env_eval.config['real_time_rendering'] = False
    env_eval.config['seed'] = config.getint('ENV_CONFIG', 'seed') + 1
    env_eval.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env_eval.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env_eval.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env_eval.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env_eval.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env_eval.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env_eval.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env_eval.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env_eval.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    env_eval.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    print(">>> 评估环境初始化完成。", flush=True)
    # --------------------------------------

    mappo = MAPPO(env=env, memory_capacity=MEMORY_CAPACITY,
                  state_dim=state_dim, action_dim=action_dim,
                  batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
                  roll_out_n_steps=ROLL_OUT_N_STEPS,
                  actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                  actor_lr=actor_lr, critic_lr=critic_lr, reward_scale=reward_scale,
                  target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
                  reward_gamma=reward_gamma, reward_type=reward_type,
                  max_grad_norm=MAX_GRAD_NORM, test_seeds=test_seeds,
                  episodes_before_train=EPISODES_BEFORE_TRAIN, traffic_density=traffic_density,
                  render_train=args.render_train, render_mode=args.render_mode
                  )

    # load the model if exist
    print(">>> [Step 4] 正在加载模型权重 (mappo.load)...", flush=True)
    mappo.load(model_dir, train_mode=True)
    print(">>> 模型加载/初始化完成。", flush=True)

    env.seed = env.config['seed']
    episodes = []
    eval_rewards = []
    best_eval_reward = -100

    print(">>> [Step 5] 进入主训练循环...", flush=True)
    while mappo.n_episodes < MAX_EPISODES:
        # 每隔 10 个 episode 打印一次进度，防止屏幕太乱
        if mappo.n_episodes % 10 == 0:
            print(f"    - 当前 Episode: {mappo.n_episodes}, 正在交互中...", flush=True)

        mappo.interact()

        if mappo.n_episodes >= EPISODES_BEFORE_TRAIN:
            mappo.train()

        if mappo.episode_done and ((mappo.n_episodes + 1) % EVAL_INTERVAL == 0):
            print(f">>> 正在进行 Episode {mappo.n_episodes + 1} 的性能评估...", flush=True)
            rewards, _, _, _ = mappo.evaluation(
                env_eval,
                dirs['train_videos'],
                EVAL_EPISODES,
                is_train=True
            )
            rewards_mu, rewards_std = agg_double_list(rewards)
            print("Episode %d, Average Reward %.2f" % (mappo.n_episodes + 1, rewards_mu), flush=True)

            # ... 保存逻辑保持不变 ...
            episodes.append(mappo.n_episodes + 1)
            eval_rewards.append(rewards_mu)
            if rewards_mu > best_eval_reward:
                mappo.save(dirs['models'], 100000)
                mappo.save(dirs['models'], mappo.n_episodes + 1)
                best_eval_reward = rewards_mu
            else:
                mappo.save(dirs['models'], mappo.n_episodes + 1)
            np.save(output_dir + '/{}'.format('episode_rewards'), np.array(mappo.episode_rewards))
            np.save(output_dir + '/{}'.format('eval_rewards'), np.array(eval_rewards))
            np.save(output_dir + '/{}'.format('average_speed'), np.array(mappo.average_speed))

    # save the model
    mappo.save(dirs['models'], MAX_EPISODES + 2)

    plt.figure()
    plt.plot(episodes, eval_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend(["MAPPO"])
    plt.show()


def evaluate(args):
    if os.path.exists(args.model_dir):
        model_dir = args.model_dir + '/models/'
    else:
        raise Exception("Sorry, no pretrained models")
    config_dir = args.model_dir + '/configs/configs_ppo.ini'
    config = configparser.ConfigParser()
    config.read(config_dir)

    video_dir = args.model_dir + '/eval_videos'
    eval_logs = args.model_dir + '/eval_logs'

    # model configs
    BATCH_SIZE = config.getint('MODEL_CONFIG', 'BATCH_SIZE')
    MEMORY_CAPACITY = config.getint('MODEL_CONFIG', 'MEMORY_CAPACITY')
    ROLL_OUT_N_STEPS = config.getint('MODEL_CONFIG', 'ROLL_OUT_N_STEPS')
    reward_gamma = config.getfloat('MODEL_CONFIG', 'reward_gamma')
    actor_hidden_size = config.getint('MODEL_CONFIG', 'actor_hidden_size')
    critic_hidden_size = config.getint('MODEL_CONFIG', 'critic_hidden_size')
    MAX_GRAD_NORM = config.getfloat('MODEL_CONFIG', 'MAX_GRAD_NORM')
    ENTROPY_REG = config.getfloat('MODEL_CONFIG', 'ENTROPY_REG')
    reward_type = config.get('MODEL_CONFIG', 'reward_type')
    TARGET_UPDATE_STEPS = config.getint('MODEL_CONFIG', 'TARGET_UPDATE_STEPS')
    TARGET_TAU = config.getfloat('MODEL_CONFIG', 'TARGET_TAU')

    # train configs
    actor_lr = config.getfloat('TRAIN_CONFIG', 'actor_lr')
    critic_lr = config.getfloat('TRAIN_CONFIG', 'critic_lr')
    EPISODES_BEFORE_TRAIN = config.getint('TRAIN_CONFIG', 'EPISODES_BEFORE_TRAIN')
    reward_scale = config.getfloat('TRAIN_CONFIG', 'reward_scale')

    # init env
    env = gym.make('merge-multi-agent-v0', disable_env_checker=True)
    env.config['seed'] = config.getint('ENV_CONFIG', 'seed')
    env.config['simulation_frequency'] = config.getint('ENV_CONFIG', 'simulation_frequency')
    env.config['duration'] = config.getint('ENV_CONFIG', 'duration')
    env.config['policy_frequency'] = config.getint('ENV_CONFIG', 'policy_frequency')
    env.config['COLLISION_REWARD'] = config.getint('ENV_CONFIG', 'COLLISION_REWARD')
    env.config['HIGH_SPEED_REWARD'] = config.getint('ENV_CONFIG', 'HIGH_SPEED_REWARD')
    env.config['HEADWAY_COST'] = config.getint('ENV_CONFIG', 'HEADWAY_COST')
    env.config['HEADWAY_TIME'] = config.getfloat('ENV_CONFIG', 'HEADWAY_TIME')
    env.config['MERGING_LANE_COST'] = config.getint('ENV_CONFIG', 'MERGING_LANE_COST')
    env.config['LANE_CHANGE_COST'] = config.getfloat('ENV_CONFIG', 'LANE_CHANGE_COST')
    env.config['MERGE_SUCCESS_REWARD'] = config.getfloat('ENV_CONFIG', 'MERGE_SUCCESS_REWARD')
    env.config['traffic_density'] = config.getint('ENV_CONFIG', 'traffic_density')
    traffic_density = config.getint('ENV_CONFIG', 'traffic_density')
    env.config['action_masking'] = config.getboolean('MODEL_CONFIG', 'action_masking')
    if args.render_train:
        env.config['offscreen_rendering'] = False
        env.config['real_time_rendering'] = True

    assert env.T % ROLL_OUT_N_STEPS == 0
    state_dim = env.n_s
    action_dim = env.n_a
    test_seeds = args.evaluation_seeds
    seeds = [int(s) for s in test_seeds.split(',')]

    mappo = MAPPO(env=env, memory_capacity=MEMORY_CAPACITY,
                  state_dim=state_dim, action_dim=action_dim,
                  batch_size=BATCH_SIZE, entropy_reg=ENTROPY_REG,
                  roll_out_n_steps=ROLL_OUT_N_STEPS,
                  actor_hidden_size=actor_hidden_size, critic_hidden_size=critic_hidden_size,
                  actor_lr=actor_lr, critic_lr=critic_lr, reward_scale=reward_scale,
                  target_update_steps=TARGET_UPDATE_STEPS, target_tau=TARGET_TAU,
                  reward_gamma=reward_gamma, reward_type=reward_type,
                  max_grad_norm=MAX_GRAD_NORM, test_seeds=test_seeds,
                  episodes_before_train=EPISODES_BEFORE_TRAIN, traffic_density=traffic_density,
                  render_train=args.render_train, render_mode=args.render_mode
                  )

    # load the model if exist
    mappo.load(model_dir, train_mode=False)
    rewards, (vehicle_speed, vehicle_position), steps, avg_speeds = mappo.evaluation(env, video_dir, len(seeds),
                                                                                     is_train=False)
    rewards_mu, rewards_std = agg_double_list(rewards)
    success_rate = sum(np.array(steps) == 100) / len(steps)
    avg_speeds_mu, avg_speeds_std = agg_double_list(avg_speeds)

    print("Evaluation Reward and std %.2f, %.2f " % (rewards_mu, rewards_std))
    print("Collision Rate %.2f" % (1 - success_rate))
    print("Average Speed and std %.2f , %.2f " % (avg_speeds_mu, avg_speeds_std))

    np.save(eval_logs + '/{}'.format('eval_rewards'), np.array(rewards))
    np.save(eval_logs + '/{}'.format('eval_steps'), np.array(steps))
    np.save(eval_logs + '/{}'.format('eval_avg_speeds'), np.array(avg_speeds))
    np.save(eval_logs + '/{}'.format('vehicle_speed'), np.array(vehicle_speed))
    np.save(eval_logs + '/{}'.format('vehicle_position'), np.array(vehicle_position))


if __name__ == "__main__":
    args = parse_args()
    # train or eval
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)
