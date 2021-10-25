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
from tqdm import tqdm
import os
import json

import environment
from environment.craft import CraftState
from misc.util import Struct, OutputManager, create_dir
from models.cvae import CVAE


def get_args():
    parser = argparse.ArgumentParser(
        description='Training and testing of CVAE')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1994, metavar='S',
                        help='random seed (default: 1994)')
    parser.add_argument('--mode', default="train",
                        help='Choose a mode: train or reconstruction')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Config file')
    parser.add_argument(
        '--net_path',
        type=str,
        default='net.pth',
        help='Path for storing the trained network')
    parser.add_argument('--dhid', type=int, default=200,
                        help='Hidden dim of CVAE')
    parser.add_argument('--drep', type=int, default=100,
                        help='Rep dim of CVAE')
    parser.add_argument('--num_rolls', type=int, default=2000,
                        help='Number of rollouts to collect training data')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def configure(file_name):
    # load config
    root_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(root_dir, "experiments", file_name)
    with open(file_path) as config_f:
        config = Struct(**yaml.load(config_f))

    return config


def get_model(is_cuda, dinp=83, dcond=181, dhid=200, drep=100):
    model = CVAE(dinp, dhid, drep, dinp, dcond, is_cuda)
    return model


def main(args):
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    config = configure(args.config)
    # Create results directory and logging
    results_path = config.results_dir + config.name
    create_dir(results_path)
    manager = OutputManager(config.results_dir + config.name)
    manager.say("Log starts, exp: {}".format(config.name))

    # Get environment
    exp_env = environment.load(config, args.seed)

    # Generate training data
    if config.world.name == "BoxWorld":
        abstract_states, abstract_partial_states, tasks, unobs_fractions = generate_training_data_box(
            exp_env, manager, num_rolls=args.num_rolls, num_steps=100)
    else:
        abstract_states, abstract_partial_states, tasks, unobs_fractions = generate_training_data(
            exp_env, manager, num_rolls=args.num_rolls, num_steps=100)

    train_data = []
    #if config.world.name == "BoxWorld":
    dinp = len(abstract_states[0])
    dcond = len(abstract_partial_states[0]) + len(tasks[0]) + 1
    print("dinp: {}".format(dinp))
    print("dcond: {}".format(dcond))
    # convert the data to tensors
    for i in range(len(abstract_states)):
        if config.world.name == "BoxWorld":
            abstract_inp = torch.tensor(
                abstract_states[i],
                dtype=torch.float32).to(device)
            abstract_partial_inp = torch.tensor(
                abstract_partial_states[i],
                dtype=torch.float32).to(device)
            task = torch.tensor(tasks[i], dtype=torch.float32).to(device)
            unobs_fraction = torch.tensor(
                [unobs_fractions[i]], dtype=torch.float32).to(device)
            train_data.append(
                [abstract_inp, abstract_partial_inp, task, unobs_fraction])

        else:
            abstract_inp = torch.tensor(
                abstract_states[i],
                dtype=torch.float32).to(device)
            abstract_partial_inp = torch.tensor(
                abstract_partial_states[i],
                dtype=torch.float32).to(device)
            task = torch.tensor(tasks[i], dtype=torch.float32).to(device)
            unobs_fraction = torch.tensor(
                [unobs_fractions[i]], dtype=torch.float32).to(device)
            train_data.append([abstract_inp,
                               abstract_partial_inp, task, unobs_fraction])

    trainloader = torch.utils.data.DataLoader(
        train_data, shuffle=True, batch_size=64)

    # Train
    if config.world.name == "BoxWorld":
        # dinp = 462, dcond = 577
        model = get_model(
            is_cuda,
            dinp=dinp,
            dcond=dcond,
            dhid=args.dhid,
            drep=args.drep)
    else:
        model = get_model(
            is_cuda,
            dinp=dinp,
            dcond=dcond)
    if is_cuda:
        model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    train_loss = 0
    recon_loss_hist = []
    ep = 0
    for epoch in range(10000):
        for i, data in enumerate(trainloader):
            if config.world.name == "BoxWorld":
                abstract_state, abstract_partial_state, task, unobs_fraction = data
            else:
                abstract_state, abstract_partial_state, task, unobs_fraction = data
            bsz = abstract_state.size(0)

            inp = abstract_state.view(bsz, -1)
            cond_inp = torch.cat(
                (abstract_partial_state.view(
                    bsz, -1), task, unobs_fraction), dim=-1)

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(inp, cond_inp)

            loss = loss_function(recon_batch, inp, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            recon_loss_hist.append(loss.detach().data.cpu().numpy())

            if ep == 0 or ep % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, train_loss / 100))
                train_loss = 0.0

                PATH = args.net_path
                print("Storing at %s" % PATH)
                torch.save(model.state_dict(), PATH)

                save_losses(recon_loss_hist)

            ep += 1


def loss_function(recon_x, x, mu, logvar):
    '''
    the loss consists of two parts, the binary Cross-Entropy loss for
    the reconstruction loss and the KL divergence loss for the variational
    inference.
    '''
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return BCE + KLD


def plot_stats_over_episodes(data, filename):
    fig = plt.figure()
    x = np.linspace(0, len(data), len(data))
    plt.plot(x, data)
    plt.savefig(filename)
    plt.close()


def save_losses(recon_loss_hist):
    filename = "recon_loss1.txt"
    np.savetxt(filename, recon_loss_hist)

    filename = "recon_loss1.png"
    plot_stats_over_episodes(recon_loss_hist, filename)


def main_color(args):
    config = configure(args.config)
    # Get environment
    exp_env = environment.load(config, args.seed)

    goal_colors = [i for i in range(exp_env.n_things - 1)]
    exp_env.plot_solution_graph(
        goal_colors, [
            [1]], [1], plot_dir='resources/box', file_name='all_colors.png')


def test(args):
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    config = configure(args.config)
    # Create results directory and logging
    results_path = config.results_dir + config.name
    create_dir(results_path)
    manager = OutputManager(config.results_dir + config.name)
    manager.say("Log starts, exp: {}".format(config.name))

    # Get environment
    exp_env = environment.load(config, args.seed)

    # Generate training data
    if config.world.name == "BoxWorld":
        abstract_states, abstract_partial_states, tasks, unobs_fractions, states = generate_training_data_box(
            exp_env, manager, num_rolls=1, num_steps=100, get_raw_states=True)
    else:
        abstract_states, abstract_partial_states, tasks, unobs_fractions, states = generate_training_data(
            exp_env, manager, num_rolls=1, num_steps=100, get_raw_states=True)

    test_data = []
    if config.world.name == "BoxWorld":
        dinp = len(abstract_states[0])
        dcond = len(abstract_partial_states[0]) + len(tasks[0]) + 1
        print("dinp: {}".format(dinp))
        print("dcond: {}".format(dcond))
    # convert data to tensors
    for i in range(len(abstract_states)):
        if config.world.name == "BoxWorld":
            abstract_inp = torch.tensor(
                abstract_states[i],
                dtype=torch.float32).to(device)
            abstract_partial_inp = torch.tensor(
                abstract_partial_states[i],
                dtype=torch.float32).to(device)
            task = torch.tensor(tasks[i], dtype=torch.float32).to(device)
            unobs_fraction = torch.tensor(
                [unobs_fractions[i]], dtype=torch.float32).to(device)
            test_data.append(
                [abstract_inp, abstract_partial_inp, task, unobs_fraction])

        else:
            abstract_inp = torch.tensor(
                abstract_states[i],
                dtype=torch.float32).to(device)
            abstract_partial_inp = torch.tensor(
                abstract_partial_states[i],
                dtype=torch.float32).to(device)
            task = torch.tensor(tasks[i], dtype=torch.float32).to(device)
            unobs_fraction = torch.tensor(
                [unobs_fractions[i]], dtype=torch.float32).to(device)
            test_data.append([abstract_inp,
                              abstract_partial_inp, task, unobs_fraction])

    testloader = torch.utils.data.DataLoader(
        test_data, shuffle=False, batch_size=64)

    # Test
    if config.world.name == "BoxWorld":
        # dinp = 462, dcond = 577
        model = get_model(is_cuda, dinp=dinp, dcond=dcond)
    else:
        model = get_model(is_cuda)
    if is_cuda:
        model.to(device)

    PATH = args.net_path
    print(PATH)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()

    dataiter = iter(testloader)
    if config.world.name == "BoxWorld":
        abstract_states, abstract_partial_states, tasks, unobs_fractions = dataiter.next()
    else:
        abstract_states, abstract_partial_states, tasks, unobs_fractions = dataiter.next()

    bsz = abstract_states.size(0)
    inp = abstract_states.view(bsz, -1)
    cond_inp = torch.cat(
        (abstract_partial_states.view(
            bsz, -1), tasks, unobs_fractions), dim=-1)

    recon_batch, mu, logvar = model(inp, cond_inp)

    loss = loss_function(recon_batch, inp, mu, logvar)
    print(loss)
    view_id = 2
    print("\ntask")
    task_id = torch.argmax(tasks[view_id]) + exp_env.grabbable_indices[0]
    print(exp_env.cookbook.index.get(task_id.item()))
    print("\nreconstructed abstract state")
    print(exp_env.parse_abstract_state(recon_batch[view_id]))
    print("\noriginal abstract state")
    print(exp_env.parse_abstract_state(abstract_states[view_id]))

    abs_partial_state = abstract_partial_states.view(bsz, -1)[view_id]
    part_len = int(len(abs_partial_state) / 2)
    print("\nblocked partial")
    print(exp_env.parse_abstract_state(abs_partial_state[:part_len]))
    print("\nnot blocked partial")
    print(exp_env.parse_abstract_state(abs_partial_state[part_len:]))

    # Setup pygame
    spec_file = os.path.join(config.icons, 'spec.json')
    with open(spec_file) as json_file:
        object_param_list = json.load(json_file)
    exp_env.setup_pygame(config.icons, object_param_list)

    exp_env.visualize_pretty(states[view_id], results_path)

    for k in range(2):
        z = Variable(torch.FloatTensor(1, 100).normal_()).to(device)
        z = torch.cat((z[0], cond_inp[view_id]), 0)
        recon = model.decode(z.unsqueeze(0))

        print("\nconditional reconstruction", k)
        print(exp_env.parse_abstract_state(recon[0]))


def generate_training_data_box(
        exp_env,
        manager,
        num_rolls=2000,
        num_steps=100,
        get_raw_states=False):
    abstract_states = []
    abstract_partial_states = []
    tasks = []
    unobs_fractions = []
    if get_raw_states:
        raw_states = []

    abstract_partial_maps = []

    n_goals = exp_env.max_goal_length

    map_width = exp_env.n
    map_height = exp_env.n
    num_samples_per_roll = 5
    for _ in tqdm(range(num_rolls)):
        # Sample scenario
        goal_length = exp_env.random.choice(exp_env.max_goal_length) + 1
        #manager.say("Task goal: get {}".format(exp_env.cookbook.index.get(goal)))
        scenario = exp_env.sample_scenario_with_goal(goal_length)
        scenario.init(None)

        # Which steps to take samples
        steps_to_sample = exp_env.random.choice(
            num_steps, size=num_samples_per_roll, replace=False)
        steps_to_sample = set(steps_to_sample)

        # run rollout
        for k in range(num_steps):
            state = scenario.get_state()
            if k in steps_to_sample:
                # Collect
                # compute abstraction of the partial map
                abstract_partial_map = exp_env.get_abstract_partial_state(
                    state)
                encoded_partial = exp_env.encode_abstract_partial_state(
                    abstract_partial_map)
                abstract_partial_states.append(encoded_partial)
                abstract_partial_maps.append(abstract_partial_map)

                # compute output abstract map
                abstract_map = exp_env.get_abstract_state(state)
                encoded_abstract = exp_env.encode_abstract_state(abstract_map)
                abstract_states.append(encoded_abstract)

                tasks.append(make_one_hot(goal_length - 1, C=n_goals))
                unobs_fractions.append(
                    1.0 - np.sum(state.mask) / (map_width * map_height))

                if get_raw_states:
                    raw_states.append(state)

            action = np.random.choice(
                exp_env.n_actions)
            _, new_state, _, _ = scenario.step(action)

    if get_raw_states:
        return abstract_states, abstract_partial_states, tasks, unobs_fractions, raw_states
    else:
        return abstract_states, abstract_partial_states, tasks, unobs_fractions


def generate_training_data(exp_env, manager, num_rolls=100, num_steps=100, get_raw_states=False):
    abstract_states = []
    abstract_partial_states = []
    tasks = []
    unobs_fractions = []
    if get_raw_states:
        raw_states = []

    n_goals = len(exp_env.grabbable_indices)
    goal_start = exp_env.grabbable_indices[0]

    map_width = exp_env.MAP_WIDTH
    map_height = exp_env.MAP_HEIGHT
    for _ in range(num_rolls):
        # Sample scenario
        if len(exp_env.usable_indices) == 3:
            # CraftWorldHardHard
            task_probs = [exp_env.task_probs[task] for task in exp_env.grabbable_indices]
            goal = exp_env.random.choice(exp_env.grabbable_indices, p=task_probs)
        else:
            goal = exp_env.random.choice(exp_env.grabbable_indices)
        #manager.say("Task goal: get {}".format(exp_env.cookbook.index.get(goal)))
        scenario = exp_env.sample_scenario_with_goal(goal)
        scenario.init(None)

        # Start state
        state = scenario.get_state()

        # compute abstraction of the partial map
        abstract_partial_map = exp_env.get_abstract_partial_state(
            state)
        abstract_partial_states.append(abstract_partial_map)

        # compute output abstract map
        abstract_map = exp_env.get_abstract_state(state)
        abstract_states.append(abstract_map)

        tasks.append(make_one_hot(goal - goal_start, C=n_goals))
        unobs_fractions.append(
            1.0 - np.sum(state.mask) / (map_width * map_height))

        if get_raw_states:
            raw_states.append(state)

        # get rollout using a random agent
        for k in range(num_steps):
            action = np.random.choice(
                exp_env.n_actions - 1)  # remove use action
            _, new_state, _, _ = scenario.step(action)

            # resample scenario at every timestep (so that we don't have same
            # full map and abstract map everytime)
            scenario = exp_env.sample_scenario_with_goal(goal)
            scenario.init(None)
            state = scenario.get_state()
            state.pos = new_state.pos  # make the agent be at the same pos as before
            state.mask = new_state.mask.copy()

            # compute abstraction of the partial map
            abstract_partial_map = exp_env.get_abstract_partial_state(
                state)
            abstract_partial_states.append(abstract_partial_map)

            # compute output abstract map
            abstract_map = exp_env.get_abstract_state(state)
            abstract_states.append(abstract_map)

            tasks.append(make_one_hot(goal - goal_start, C=n_goals))

            unobs_fractions.append(
                1.0 - np.sum(state.mask) / (map_width * map_height))

            if get_raw_states:
                raw_states.append(state)

    if get_raw_states:
        return abstract_states, abstract_partial_states, tasks, unobs_fractions, raw_states
    else:
        return abstract_states, abstract_partial_states, tasks, unobs_fractions


def get_neighboring_states(pos, map_width, map_height, dist=1):
    neighbors = set()
    for a in range(-dist, dist + 1):
        x = pos[0] + a
        if x < 1 or x >= map_width - 1:
            continue
        for b in range(-dist, dist + 1):
            y = pos[1] + b
            if y < 1 or y >= map_height - 1:
                continue
            neighbors.add((x, y))
    return neighbors


def get_neighboring_states_box(pos, map_width, map_height, dist=1):
    neighbors = set()
    for a in range(-dist, dist + 1):
        x = pos[0] + a
        if x < 0 or x >= map_width:
            continue
        for b in range(-dist, dist + 1):
            y = pos[1] + b
            if y < 0 or y >= map_height:
                continue
            neighbors.add((x, y))
    return neighbors


def get_partial_map(full_map, obs_squares):
    len_x, len_y, len_z = full_map.shape

    obs_x = [a for (a, b) in obs_squares]
    obs_y = [b for (a, b) in obs_squares]

    partial_map = full_map.copy()
    partial_map[:, :, -1] = 1
    partial_map[obs_x, obs_y, -1] = 0

    unobs_x, unobs_y = (partial_map[:, :, -1] == 1).nonzero()
    partial_map[unobs_x, unobs_y, :len_z - 1] = 0

    return partial_map


def make_one_hot(label, C=10):
    one_hot = np.zeros(C)
    one_hot[label] = 1

    return one_hot


if __name__ == '__main__':
    try:
        args = get_args()
        if args.mode == "train":
            main(args)
        elif args.mode == "test":
            test(args)
        elif args.mode == "color":
            main_color(args)
        else:
            print("Running mode not supported")

    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
