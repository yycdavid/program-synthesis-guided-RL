import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import os
from tqdm import tqdm
import pickle
import json

import environment
from environment.ant import AntCraftEnv
import trainers
from misc.util import Struct, OutputManager, create_dir, make_one_hot
from stable_baselines3 import PPO, SAC
from pyvirtualdisplay import Display
from gym import wrappers
import gym
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

import difflib
import importlib
import uuid

import seaborn
from stable_baselines3.common.utils import set_random_seed

# Register custom envs
import utils.import_envs  # noqa: F401 pytype: disable=import-error
from utils.exp_manager import ExperimentManager
from utils.utils import ALGOS, StoreDict

seaborn.set()


def get_args():
    parser = argparse.ArgumentParser(
        description='Main experiment script for ANT environment')
    parser.add_argument('--mode', default="train",
                        help='Choose a mode: train or ...')
    parser.add_argument('--exp', default="None",
                        help='Select experiment ...')
    parser.add_argument('--continue_on', default="None",
                        help='Do training based on a previous exp ...')
    parser.add_argument(
        '--goal_d',
        type=float,
        default=1.0,
        help='Goal distance for Antg env')
    parser.add_argument(
        '--r_scale',
        type=float,
        default=1.0,
        help='Scaling factor for forward reward')

    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        default="ppo",
        type=str,
        required=False,
        choices=list(
            ALGOS.keys()))
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="environment ID")
    parser.add_argument(
        "-tb",
        "--tensorboard-log",
        help="Tensorboard log dir",
        default="",
        type=str)
    parser.add_argument(
        "-i",
        "--trained-agent",
        help="Path to a pretrained agent to continue training",
        default="",
        type=str)
    parser.add_argument(
        "--truncate-last-trajectory",
        help="When using HER with online sampling the last trajectory "
        "in the replay buffer will be truncated after reloading the replay buffer.",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "-n",
        "--n-timesteps",
        help="Overwrite the number of timesteps",
        default=-1,
        type=int)
    parser.add_argument(
        "--num-threads",
        help="Number of threads for PyTorch (-1 to use default)",
        default=-1,
        type=int)
    parser.add_argument(
        "--log-interval",
        help="Override log interval (default: -1, no change)",
        default=-1,
        type=int)
    parser.add_argument(
        "--eval-freq",
        help="Evaluate the agent every n steps (if negative, no evaluation)",
        default=10000,
        type=int)
    parser.add_argument(
        "--eval-episodes",
        help="Number of episodes to use for evaluation",
        default=5,
        type=int)
    parser.add_argument(
        "--save-freq",
        help="Save the model every n steps (if negative, no checkpoint)",
        default=-1,
        type=int)
    parser.add_argument(
        "--save-replay-buffer",
        help="Save the replay buffer too (when applicable)",
        action="store_true",
        default=False)
    parser.add_argument(
        "-f",
        "--log-folder",
        help="Log folder",
        type=str,
        default="logs")
    parser.add_argument(
        "--seed",
        help="Random generator seed",
        type=int,
        default=-1)
    parser.add_argument(
        "--vec-env",
        help="VecEnv type",
        type=str,
        default="dummy",
        choices=[
            "dummy",
            "subproc"])
    parser.add_argument(
        "--n-trials",
        help="Number of trials for optimizing hyperparameters",
        type=int,
        default=10)
    parser.add_argument(
        "-optimize",
        "--optimize-hyperparameters",
        action="store_true",
        default=False,
        help="Run hyperparameters search")
    parser.add_argument(
        "--n-jobs",
        help="Number of parallel jobs when optimizing hyperparameters",
        type=int,
        default=1)
    parser.add_argument(
        "--sampler",
        help="Sampler to use when optimizing hyperparameters",
        type=str,
        default="tpe",
        choices=["random", "tpe", "skopt"],
    )
    parser.add_argument(
        "--pruner",
        help="Pruner to use when optimizing hyperparameters",
        type=str,
        default="median",
        choices=["halving", "median", "none"],
    )
    parser.add_argument(
        "--n-startup-trials",
        help="Number of trials before using optuna sampler",
        type=int,
        default=10)
    parser.add_argument(
        "--n-evaluations",
        help="Number of evaluations for hyperparameter optimization",
        type=int,
        default=20)
    parser.add_argument(
        "--storage",
        help="Database storage path if distributed optimization should be used",
        type=str,
        default=None)
    parser.add_argument(
        "--study-name",
        help="Study name for distributed optimization",
        type=str,
        default=None)
    parser.add_argument(
        "--verbose",
        help="Verbose mode (0: no output, 1: INFO)",
        default=1,
        type=int)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional keyword argument to pass to the env constructor")
    parser.add_argument(
        "-params",
        "--hyperparams",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",
    )
    parser.add_argument(
        "-uuid",
        "--uuid",
        action="store_true",
        default=False,
        help="Ensure that the run has a unique ID")

    return parser.parse_args()


def main_test(args):
    exp_folder = os.path.join("logs/sac", args.exp)
    model_file = os.path.join(exp_folder, "Antg-v1.zip")
    # Load model
    #model = SAC.load("Ant-v2.zip")
    model = SAC.load(model_file)

    env_kwargs = get_env_arg(args)

    # Eval model
    env = gym.make('Antg-v1', **env_kwargs)
    episode_rewards, episode_lengths, final_dists, mean_reward, std_reward, mean_dist, std_dist = evaluate_policy(
        model, env, n_eval_episodes=10, return_episode_rewards=True, with_dist=True)
    print("Mean reward:")
    print(mean_reward)
    print("std reward:")
    print(std_reward)
    print("Mean final dist:")
    print(mean_dist)
    # Save stats
    stats_file = os.path.join(exp_folder, 'stats.json')
    stats_dict = {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_dist': mean_dist,
        'std_dist': std_dist,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'final_dists': final_dists,
    }
    with open(stats_file, 'w') as outfile:
        json.dump(stats_dict, outfile)

    # Record video
    obs = env.reset()
    env = DummyVecEnv([lambda: env])
    obs = env.reset()
    video_folder = 'logs/videos/'
    video_length = 50

    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="agent-sac-g-" +
        args.exp)
    env.reset()

    for i in range(video_length + 1):
        action, _states = model.predict(obs, deterministic=True)
        #action = [env.action_space.sample()]
        obs, reward, done, info = env.step(action)
        # if done:
        #    print("Done")
        #    obs = env.reset()

    print(obs[0][-2:])

    env.close()


def main_train_bl(args):
    # Going through custom gym packages to let them register in the global
    # registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    # pytype: disable=module-attr
    registered_envs = set(gym.envs.registry.env_specs.keys())

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(
                env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(
            f"{env_id} not found in gym registry, you maybe meant {closest_match}?")

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f"_{uuid.uuid4()}" if args.uuid else ""
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2 ** 32 - 1, dtype="int64").item()

    set_random_seed(args.seed)

    # Setting num threads to 1 makes things run faster on cpu
    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    if args.trained_agent != "":
        assert args.trained_agent.endswith(".zip") and os.path.isfile(
            args.trained_agent
        ), "The trained_agent must be a valid path to a .zip file"

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")

    env_kwargs = get_env_arg(args)

    if args.exp == "None":
        exp_name = None
    else:
        exp_name = args.exp

    exp_manager = ExperimentManager(
        args,
        args.algo,
        env_id,
        args.log_folder,
        args.tensorboard_log,
        args.n_timesteps,
        args.eval_freq,
        args.eval_episodes,
        args.save_freq,
        args.hyperparams,
        env_kwargs,
        args.trained_agent,
        args.optimize_hyperparameters,
        args.storage,
        args.study_name,
        args.n_trials,
        args.n_jobs,
        args.sampler,
        args.pruner,
        n_startup_trials=args.n_startup_trials,
        n_evaluations=args.n_evaluations,
        truncate_last_trajectory=args.truncate_last_trajectory,
        uuid_str=uuid_str,
        seed=args.seed,
        log_interval=args.log_interval,
        save_replay_buffer=args.save_replay_buffer,
        verbose=args.verbose,
        vec_env_type=args.vec_env,
        exp_name=exp_name,
    )

    # Prepare experiment and launch hyperparameter optimization if needed
    model = exp_manager.setup_experiment()

    # Normal training
    if model is not None:
        exp_manager.learn(model)
        exp_manager.save_trained_model(model)
    else:
        exp_manager.hyperparameters_optimization()


def get_env_arg(args):
    if args.continue_on != "None":
        exp_folder = os.path.join("logs/sac", args.continue_on)
        save_path = os.path.join(exp_folder, 'end_states.pickle')
        with open(save_path, 'rb') as f:
            end_states = pickle.load(f)
        env_kwargs = end_states

    else:
        env_kwargs = {}

    env_kwargs['goal_distance'] = args.goal_d
    env_kwargs['forward_reward_scale'] = args.r_scale

    return env_kwargs


def main_save_end(args):
    exp_folder = os.path.join("logs/sac", args.exp)
    model_file = os.path.join(exp_folder, "Antg-v1.zip")
    # Load model
    model = SAC.load(model_file)

    env_kwargs = get_env_arg(args)
    env = gym.make('Antg-v1', **env_kwargs)

    episode_length = 29
    num_states_needed = 5000
    final_poses = []
    final_vels = []

    states_saved = 0
    for j in tqdm(range(num_states_needed)):
        saved = False
        while not saved:
            obs = env.reset()
            ok = True
            for i in range(episode_length):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                # Save ending state if the final 15 steps doesn't fail
                if i > 15 and done:
                    ok = False
                    break

            if ok:
                final_poses.append(obs[:env.model.nq - 2])
                final_vels.append(
                    obs[env.model.nq - 2:env.model.nq + env.model.nv - 2])
                saved = True

    end_states = {
        'final_poses': final_poses,
        'final_vels': final_vels,
    }
    save_path = os.path.join(exp_folder, 'end_states_old.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(end_states, f)


def main_inspect(args):
    exp_folder = os.path.join("logs/sac", args.exp)
    model_file = os.path.join(exp_folder, "Antg-v1.zip")
    # Load model
    #model = SAC.load("Ant-v2.zip")
    model = SAC.load(model_file)
    import pdb
    pdb.set_trace()

    env_kwargs = get_env_arg(args)

    n_angles = 8
    thetas = (np.arange(n_angles) / n_angles * 2 * np.pi).tolist()
    for (i, theta) in enumerate(thetas):
        env_kwargs['set_goal'] = theta

        # Eval model
        env = gym.make('Antg-v1', **env_kwargs)
        episode_rewards, episode_lengths, final_dists, mean_reward, std_reward, mean_dist, std_dist = evaluate_policy(
            model, env, n_eval_episodes=10, return_episode_rewards=True, with_dist=True)
        print("Mean reward:")
        print(mean_reward)
        print("std reward:")
        print(std_reward)
        print("Mean final dist:")
        print(mean_dist)
        # Save stats
        stats_file = os.path.join(exp_folder, 'stats_{}.json'.format(i))
        stats_dict = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_dist': mean_dist,
            'std_dist': std_dist,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_dists': final_dists,
        }
        with open(stats_file, 'w') as outfile:
            json.dump(stats_dict, outfile)

        # Record video
        obs = env.reset()
        env = DummyVecEnv([lambda: env])
        obs = env.reset()
        video_folder = 'logs/videos/'
        video_length = 48

        env = VecVideoRecorder(
            env,
            video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
            name_prefix="agent-sac-g-" +
            args.exp +
            "-angle_{}".format(i))
        env.reset()

        for i in range(video_length + 1):
            action, _states = model.predict(obs, deterministic=True)
            #action = [env.action_space.sample()]
            obs, reward, done, info = env.step(action)
            # if done:
            #    print("Done")
            #    obs = env.reset()

        print(obs[0][-2:])

        env.close()


def main_filter(args):
    exp_model_folder = os.path.join("logs/sac", args.exp)
    model_file = os.path.join(exp_model_folder, "Antg-v1.zip")
    # Load model
    model = SAC.load(model_file)

    exp_folder = os.path.join("logs/sac", args.continue_on)
    save_path = os.path.join(exp_folder, 'end_states_old.pickle')
    with open(save_path, 'rb') as f:
        end_states = pickle.load(f)

    total_num = len(end_states['final_poses'])
    selected = []

    for i in tqdm(range(total_num)):
        env_kwargs = {}
        env_kwargs['final_poses'] = [end_states['final_poses'][i]]
        env_kwargs['final_vels'] = [end_states['final_vels'][i]]

        env_kwargs['goal_distance'] = args.goal_d
        env_kwargs['forward_reward_scale'] = args.r_scale

        env = gym.make('Antg-v1', **env_kwargs)

        ok = True
        obs = env.reset()
        for j in range(15):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done:
                ok = False

        if ok:
            selected.append(i)

    print(total_num)
    print("Selected: {}".format(len(selected)))

    end_states_new = {
        'final_poses': [end_states['final_poses'][i] for i in selected],
        'final_vels': [end_states['final_vels'][i] for i in selected],
    }
    save_path = os.path.join(exp_folder, 'end_states.pickle')
    with open(save_path, 'wb') as f:
        pickle.dump(end_states_new, f)


if __name__ == '__main__':
    try:
        args = get_args()
        if args.mode == "train":
            main_train_bl(args)
        elif args.mode == "test":
            main_test(args)
        elif args.mode == "inspect":
            main_inspect(args)
        elif args.mode == "end":
            main_save_end(args)
        elif args.mode == "filter":
            main_filter(args)
        else:
            print("Running mode not supported")

    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
