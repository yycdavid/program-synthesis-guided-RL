import argparse
import sys
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import os
from tqdm import tqdm
from torch.distributions import Normal
import pickle

import environment
from environment.box import BoxWorld
from environment.craft import CraftWorldHardHard
import trainers
from misc.util import Struct, OutputManager, create_dir, make_one_hot
from models.cvae import ConvVAE
from models.rnn import MDNRNN
from models.simple_nn import Controller, NNModel, NNConvModel
from models.world_models import WorldModel


BETA = 1.0
EPSILON = 1e-6
Z_DIM = 100
RNN_DIM = 256


def get_args():
    parser = argparse.ArgumentParser(description='World models Training')
    parser.add_argument(
        '--seed',
        type=int,
        default=1995,
        metavar='S',
        help='random seed, default is 1995, do not change default')
    parser.add_argument(
        '--train_seed',
        type=int,
        default=1995,
        metavar='S',
        help='random seed to identify training folder, default is 1995, do not change default')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Config file')
    parser.add_argument('--mode', default="train",
                        help='Choose a mode: train or ...')
    parser.add_argument(
        '--number_of_saves',
        type=int,
        default=50,
        help='Number of save points for training')
    parser.add_argument(
        '--test_iter',
        type=str,
        default='all',
        help='Test the model saved at which iteration')
    parser.add_argument(
        '--vm_from_exp',
        type=str,
        default='self',
        help='Load v and m model from another experiment')
    parser.add_argument(
        '--no_reuse_key',
        action='store_true',
        default=False,
        help='Allow reusing keys in box environment')
    parser.add_argument(
        '--update_test_setting',
        action='store_true',
        default=False,
        help='Use the specified test setting in test set')

    args = parser.parse_args()
    args.reuse_key = not args.no_reuse_key

    return args


def configure(file_name):
    # load config
    root_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(root_dir, "experiments", file_name)
    with open(file_path) as config_f:
        config = Struct(**yaml.load(config_f))

    return config


def get_v_model(model_type, d_map_in, d_flat, z_dim, side_len):
    if model_type == "ConvVAE":
        model = ConvVAE(d_map_in, d_flat, z_dim, side_len)
    else:
        assert(False, "V model not valid")
    return model


def get_m_model(
        model_type,
        z_dim,
        action_dim,
        hidden_dim=256,
        lstm_hidden_units=256,
        device='cpu'):
    if model_type == "MDNRNN":
        model = MDNRNN(
            z_dim,
            action_dim,
            hidden_dim=hidden_dim,
            hidden_units=lstm_hidden_units,
            device=device)
    else:
        assert(False, "M model not valid")
    return model


def generate_training_data(exp_env, num_rolls=100, num_steps=100):
    maps = []
    flat_inputs = []
    actions = []
    is_box = isinstance(exp_env, BoxWorld)

    if is_box:
        n_goals = exp_env.max_goal_length
        goal_start = 1
        map_width = exp_env.n
        map_height = exp_env.n

    else:
        n_goals = len(exp_env.grabbable_indices)
        goal_start = exp_env.grabbable_indices[0]
        map_width = exp_env.MAP_WIDTH
        map_height = exp_env.MAP_HEIGHT

    for _ in tqdm(range(num_rolls)):
        # Sample scenario
        if is_box:
            goal = exp_env.random.choice(n_goals) + 1
        elif isinstance(exp_env, CraftWorldHardHard):
            tasks = list(exp_env.task_probs.keys())
            task_probs = [exp_env.task_probs[t] for t in tasks]
            goal = exp_env.random.choice(tasks, p=task_probs)
        else:
            goal = exp_env.random.choice(exp_env.grabbable_indices)
        scenario = exp_env.sample_scenario_with_goal(goal)
        scenario.init(None)

        for k in range(num_steps):
            # Store inputs
            state = scenario.get_state()
            map, flat_input = state.features_world_model()
            task_input = make_one_hot(goal - goal_start, C=n_goals)
            flat_input = np.concatenate((flat_input, task_input))

            maps.append(map)
            flat_inputs.append(flat_input)

            # Random action
            action = np.random.choice(exp_env.n_actions)
            scenario.step(action)
            actions.append(action)

    return maps, flat_inputs, actions


def train_v(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = configure(args.config)

    # Create results directory and logging
    results_path = config.results_dir + config.name
    create_dir(results_path)
    manager = OutputManager(
        config.results_dir + config.name,
        filename='log_world_models_v.txt')
    manager.say("Log starts, exp: {}".format(config.name))

    # Set up environment
    exp_env = environment.load(config, args.seed)

    # Create model
    if isinstance(exp_env, BoxWorld):
        v_model = get_v_model(
            config.v_model.name,
            d_map_in=exp_env.map_feature_dim,
            d_flat=exp_env.flat_feature_dim,
            z_dim=Z_DIM,
            side_len=exp_env.n)
    else:
        v_model = get_v_model(
            config.v_model.name,
            d_map_in=exp_env.map_feature_dim,
            d_flat=exp_env.flat_feature_dim,
            z_dim=Z_DIM,
            side_len=exp_env.MAP_WIDTH)
    device = torch.device("cuda" if config.model.device == "cuda" else "cpu")
    v_model.to(device)

    maps, flat_inputs, actions = generate_training_data(
        exp_env, num_rolls=config.v_model.train_rolls, num_steps=100)
    rollout_data = {
        'maps': maps,
        'flat_inputs': flat_inputs,
        'actions': actions,
    }

    maps = rollout_data['maps']
    flat_inputs = rollout_data['flat_inputs']

    train_data = []
    # convert the data to tensors
    manager.say("Converting data to tensors")
    for i in range(len(maps)):
        map_input = torch.tensor(maps[i], dtype=torch.float32).to(device)
        flat_input = torch.tensor(
            flat_inputs[i],
            dtype=torch.float32).to(device)
        train_data.append([map_input, flat_input])

    trainloader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=64)

    # Train
    optimizer = optim.Adam(v_model.parameters(), lr=1e-3)

    v_model.train()
    train_loss = 0
    loss_hist = []
    batch_i = 0
    model_save_file = os.path.join(results_path, 'v_model.pt')
    manager.say("Training starts")
    for epoch in range(config.v_model.train_epochs):
        for i, data in tqdm(enumerate(trainloader)):
            map_input, flat_input = data

            optimizer.zero_grad()
            (map_recon, flat_recon), mu, logvar = v_model(map_input, flat_input)
            loss = loss_function(
                map_recon,
                flat_recon,
                map_input,
                flat_input,
                mu,
                logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loss_hist.append(loss.detach().data.cpu().numpy())

            if batch_i == 0 or batch_i % 100 == 99:
                manager.say('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, train_loss / 100))
                train_loss = 0.0

                manager.say("Storing at %s" % model_save_file)
                torch.save(v_model.state_dict(), model_save_file)

            batch_i += 1

    save_losses(loss_hist, results_path)


def mdn_loss_function(out_pi, out_sigma, out_mu, y):
    # y: (L, N, D), pi&sigma&mu: (L, N, n_gaussians, D)
    y = y.unsqueeze(2)  # (L, N, 1, D)
    result = Normal(loc=out_mu, scale=out_sigma)
    result = torch.exp(result.log_prob(y))
    result = torch.sum(result * out_pi, dim=2)
    result = -torch.log(EPSILON + result)
    return torch.mean(result)


def train_m(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = configure(args.config)

    # Create results directory and logging
    results_path = config.results_dir + config.name
    create_dir(results_path)
    manager = OutputManager(
        config.results_dir + config.name,
        filename='log_world_models_m.txt')
    manager.say("Log starts, exp: {}".format(config.name))

    # Set up environment
    exp_env = environment.load(config, args.seed)

    maps, flat_inputs, actions = generate_training_data(
        exp_env, num_rolls=config.v_model.train_rolls, num_steps=100)
    rollout_data = {
        'maps': maps,
        'flat_inputs': flat_inputs,
        'actions': actions,
    }

    # Load v model
    if isinstance(exp_env, BoxWorld):
        v_model = get_v_model(
            config.v_model.name,
            d_map_in=exp_env.map_feature_dim,
            d_flat=exp_env.flat_feature_dim,
            z_dim=Z_DIM,
            side_len=exp_env.n)
    else:
        v_model = get_v_model(
            config.v_model.name,
            d_map_in=exp_env.map_feature_dim,
            d_flat=exp_env.flat_feature_dim,
            z_dim=Z_DIM,
            side_len=exp_env.MAP_WIDTH)
    v_model_path = os.path.join(results_path, 'v_model.pt')
    m_model_path = os.path.join(results_path, 'm_model.pt')
    v_model.load_state_dict(torch.load(v_model_path))
    device = torch.device("cuda" if config.model.device == "cuda" else "cpu")
    device_v = torch.device("cpu")
    v_model.to(device_v)

    # Process training rollouts using v model
    manager.say("Processing training data with v model")
    maps = rollout_data['maps']
    flat_inputs = rollout_data['flat_inputs']
    actions = rollout_data['actions']
    zs = []
    actions_one_hot = []
    for i in tqdm(range(len(maps))):
        map_input = torch.tensor(maps[i], dtype=torch.float32).to(device_v)
        flat_input = torch.tensor(
            flat_inputs[i],
            dtype=torch.float32).to(device_v)
        z = v_model(
            map_input.unsqueeze(0),
            flat_input.unsqueeze(0),
            encode=True)
        zs.append(z.detach())

        # One hot actions
        action_one_hot = make_one_hot(actions[i], C=exp_env.n_actions)
        actions_one_hot.append(
            torch.tensor(
                action_one_hot,
                dtype=torch.float32).to(device_v).unsqueeze(0))

    # Divide data per rollout
    seq_len = 100
    assert len(zs) % seq_len == 0, 'Data should be a multiple of 100'
    zs_by_rollout = []
    actions_by_rollout = []
    for i_rollout in range(int(len(zs) / seq_len)):
        start_idx = i_rollout * seq_len
        z_rollout = torch.cat(
            zs[start_idx:start_idx + seq_len], dim=0).detach()  # (L, H)
        zs_by_rollout.append(z_rollout)

        a_rollout = torch.cat(
            actions_one_hot[start_idx:start_idx + seq_len], dim=0).detach()  # (L, H_a)
        actions_by_rollout.append(a_rollout)

    # Define M Model
    m_model = get_m_model(
        config.m_model.name,
        z_dim=Z_DIM,
        action_dim=exp_env.n_actions)
    m_model.to(device)

    # Train M model
    train_data = []
    for i in range(len(zs_by_rollout)):
        train_data.append([zs_by_rollout[i], actions_by_rollout[i]])

    trainloader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=16)

    optimizer = optim.Adam(m_model.parameters(), lr=1e-3)

    m_model.train()
    train_loss = 0
    loss_hist = []
    batch_i = 0
    model_save_file = os.path.join(results_path, 'm_model.pt')
    manager.say("Training starts")
    for epoch in range(config.m_model.train_epochs):
        for i, (zs, actions) in tqdm(enumerate(trainloader)):
            # zs: (N, L, H_z), actions: (N, L, H_a)
            zs, actions = zs.to(device), actions.to(device)

            optimizer.zero_grad()
            input_combined = torch.transpose(
                torch.cat((zs, actions), dim=2), 0, 1)  # (L, N, H_z+H_a)
            pi, sigma, mu = m_model(
                input_combined, reset=True)  # (L, N, n_gaussians, H_z)

            # Shift target z
            last_z = zs[:, -1:, :]
            target = torch.cat((zs[:, 1:seq_len, :], last_z), dim=1)
            target = torch.transpose(target, 0, 1)

            loss = mdn_loss_function(pi, sigma, mu, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loss_hist.append(loss.detach().data.cpu().numpy())

            if batch_i == 0 or batch_i % 100 == 99:
                manager.say('[%d, %5d] loss: %.3f' %
                            (epoch + 1, i + 1, train_loss / 100))
                train_loss = 0.0

                manager.say("Storing at %s" % model_save_file)
                torch.save(m_model.state_dict(), model_save_file)

            batch_i += 1

    save_losses(loss_hist, results_path, model_type='m')


def plot_stats_over_batches(data, filename):
    fig = plt.figure()
    x = np.linspace(0, len(data), len(data))
    plt.plot(x, data)
    plt.savefig(filename)
    plt.close()


def save_losses(loss_hist, result_dir, model_type='v'):
    save_file = os.path.join(
        result_dir,
        "{}_model_training_loss.txt".format(model_type))
    np.savetxt(save_file, loss_hist)

    save_file = os.path.join(
        result_dir,
        "{}_model_training_loss.png".format(model_type))
    plot_stats_over_batches(loss_hist, save_file)


def loss_function(map_recon, flat_recon, map_input, flat_input, mu, logvar):
    batch_size = map_input.size(0)
    map_loss = F.binary_cross_entropy(map_recon, map_input, reduction='sum')
    flat_loss = F.binary_cross_entropy(flat_recon, flat_input, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = (map_loss + flat_loss + BETA * kld) / batch_size
    return total_loss


def train_c(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    config = configure(args.config)

    # Create results directory and logging
    if args.seed != 1995:
        results_path = config.results_dir + config.name + '_seed_{}'.format(args.seed)
    else:
        results_path = config.results_dir + config.name
    create_dir(results_path)
    manager = OutputManager(
        results_path,
        filename='log_world_models_c.txt')
    manager.say("Log starts, exp: {}".format(config.name))

    # Set up environment
    exp_env = environment.load(config, args.seed)

    # Load v model
    if isinstance(exp_env, BoxWorld):
        v_model = get_v_model(
            config.v_model.name,
            d_map_in=exp_env.map_feature_dim,
            d_flat=exp_env.flat_feature_dim,
            z_dim=Z_DIM,
            side_len=exp_env.n)
    else:
        v_model = get_v_model(
            config.v_model.name,
            d_map_in=exp_env.map_feature_dim,
            d_flat=exp_env.flat_feature_dim,
            z_dim=Z_DIM,
            side_len=exp_env.MAP_WIDTH)
    if args.vm_from_exp == 'self':
        v_model_path = os.path.join(results_path, 'v_model.pt')
    else:
        exp_path = os.path.join(config.results_dir, args.vm_from_exp)
        v_model_path = os.path.join(exp_path, 'v_model.pt')
    v_model.load_state_dict(torch.load(v_model_path))
    device = torch.device("cuda" if config.model.device == "cuda" else "cpu")
    v_model.to(device)

    # Load m model
    if args.vm_from_exp == 'self':
        m_model_path = os.path.join(results_path, 'm_model.pt')
    else:
        exp_path = os.path.join(config.results_dir, args.vm_from_exp)
        m_model_path = os.path.join(exp_path, 'm_model.pt')
    m_model = get_m_model(
        config.m_model.name,
        z_dim=Z_DIM,
        action_dim=exp_env.n_actions,
        device=config.model.device)
    m_model.load_state_dict(torch.load(m_model_path))
    m_model.to(device)

    # Define C model
    if isinstance(exp_env, BoxWorld):
        c_model = NNConvModel(config)
    else:
        c_model = NNModel(config)

    c_model.to(device)
    if config.c_model.train_alg == 'es':
        num_params = (Z_DIM + RNN_DIM * 2) * exp_env.n_actions

    # Train
    trainer = trainers.load(config, exp_env, manager)
    m_model.eval()
    v_model.eval()
    if config.c_model.train_alg == "ac":
        if config.c_model.input == 'encoded':
            model = WorldModel(
                v_model,
                m_model,
                c_model,
                c_input_dim=Z_DIM +
                RNN_DIM *
                2)
        elif config.c_model.input == 'combined':
            model = WorldModel(
                v_model,
                m_model,
                c_model,
                c_input_dim=Z_DIM +
                RNN_DIM *
                2 +
                exp_env.n_features,
                c_input_mode=config.c_model.input)
        elif config.c_model.input == 'scaled_h_only':
            if isinstance(exp_env, BoxWorld):
                model = WorldModel(
                    v_model,
                    m_model,
                    c_model,
                    c_input_dim=Z_DIM + RNN_DIM,
                    c_input_mode=config.c_model.input,
                    h_scale=config.c_model.h_scale,
                    z_scale=config.c_model.z_scale,
                    critic_orig=config.c_model.critic_orig,
                    is_box=True)
            else:
                model = WorldModel(
                    v_model,
                    m_model,
                    c_model,
                    c_input_dim=Z_DIM +
                    RNN_DIM +
                    exp_env.n_features,
                    c_input_mode=config.c_model.input,
                    h_scale=config.c_model.h_scale,
                    z_scale=config.c_model.z_scale,
                    critic_orig=config.c_model.critic_orig)
        else:
            model = WorldModel(
                v_model,
                m_model,
                c_model,
                c_input_dim=exp_env.n_features,
                c_input_mode=config.c_model.input)
        trainer.train(model, number_of_saves=args.number_of_saves)
    else:
        assert False, "Training algorithm not supported"


def test_stats(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config
    config = configure(args.config)
    #config.model.device = 'cpu'

    # Create results directory and logging
    if args.train_seed != 1995:
        results_path = config.results_dir + config.name + '_seed_{}'.format(args.train_seed)
    else:
        results_path = config.results_dir + config.name
    create_dir(results_path)
    manager = OutputManager(
        results_path,
        filename='log_test.txt')
    manager.say("Log starts, exp: {}".format(config.name))

    # Set up environment
    exp_env = environment.load(config, args.seed)

    # Load v model
    if isinstance(exp_env, BoxWorld):
        v_model = get_v_model(
            config.v_model.name,
            d_map_in=exp_env.map_feature_dim,
            d_flat=exp_env.flat_feature_dim,
            z_dim=Z_DIM,
            side_len=exp_env.n)
    else:
        v_model = get_v_model(
            config.v_model.name,
            d_map_in=exp_env.map_feature_dim,
            d_flat=exp_env.flat_feature_dim,
            z_dim=Z_DIM,
            side_len=exp_env.MAP_WIDTH)
    if args.vm_from_exp == 'self':
        v_model_path = os.path.join(results_path, 'v_model.pt')
    else:
        exp_path = os.path.join(config.results_dir, args.vm_from_exp)
        v_model_path = os.path.join(exp_path, 'v_model.pt')
    v_model.load_state_dict(torch.load(v_model_path))
    device = torch.device("cuda" if config.model.device == "cuda" else "cpu")
    #device = torch.device("cpu")
    v_model.to(device)

    # Load m model
    if args.vm_from_exp == 'self':
        m_model_path = os.path.join(results_path, 'm_model.pt')
    else:
        exp_path = os.path.join(config.results_dir, args.vm_from_exp)
        m_model_path = os.path.join(exp_path, 'm_model.pt')
    m_model = get_m_model(
        config.m_model.name,
        z_dim=Z_DIM,
        action_dim=exp_env.n_actions,
        device=config.model.device)
    m_model.load_state_dict(torch.load(m_model_path))
    m_model.to(device)

    # Define C model
    if isinstance(exp_env, BoxWorld):
        c_model = NNConvModel(config)
    else:
        c_model = NNModel(config)

    c_model.to(device)

    trainer = trainers.load(config, exp_env, manager)
    c_model.eval()
    m_model.eval()
    v_model.eval()
    if config.c_model.train_alg == "ac":
        if config.c_model.input == 'encoded':
            model = WorldModel(
                v_model,
                m_model,
                c_model,
                c_input_dim=Z_DIM +
                RNN_DIM *
                2)

        elif config.c_model.input == 'scaled_h_only':
            if isinstance(exp_env, BoxWorld):
                model = WorldModel(
                    v_model,
                    m_model,
                    c_model,
                    c_input_dim=Z_DIM + RNN_DIM,
                    c_input_mode=config.c_model.input,
                    h_scale=config.c_model.h_scale,
                    z_scale=config.c_model.z_scale,
                    critic_orig=config.c_model.critic_orig,
                    is_box=True)
            else:
                model = WorldModel(
                    v_model,
                    m_model,
                    c_model,
                    c_input_dim=Z_DIM +
                    RNN_DIM +
                    exp_env.n_features,
                    c_input_mode=config.c_model.input,
                    h_scale=config.c_model.h_scale,
                    z_scale=config.c_model.z_scale,
                    critic_orig=config.c_model.critic_orig)

        else:
            model = WorldModel(
                v_model,
                m_model,
                c_model,
                c_input_dim=Z_DIM +
                RNN_DIM *
                2 +
                exp_env.n_features,
                c_input_mode=config.c_model.input)

        if args.mode == 'test_stats':
            trainer.test_statistics(
                model,
                None,
                models_to_test=args.test_iter,
                seed=args.seed,
                reuse_key=args.reuse_key,
                update_test_setting=args.update_test_setting)
        else:
            trainer.test(
                model,
                None,
                model_to_test=args.test_iter,
                reuse_key=args.reuse_key,
                update_test_setting=args.update_test_setting)
    else:
        assert False, "Training algorithm not supported"


if __name__ == '__main__':
    try:
        args = get_args()
        if args.mode == "train_v":
            train_v(args)
        elif args.mode == "train_m":
            train_m(args)
        elif args.mode == "train_c":
            train_c(args)
        elif args.mode == "test_stats":
            test_stats(args)
        elif args.mode == "test":
            test_stats(args)
        else:
            print("Running mode not supported")

    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
