import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.util import make_one_hot
from .simple_nn import NNConvModel


class WorldModel(object):
    """Wrapper for world models with actor critic for controller, to have the same interface as normal NNModel or Modular AC models"""

    def __init__(
            self,
            v_model,
            m_model,
            c_model,
            c_input_dim=100,
            c_input_mode='scaled_h_only',
            h_scale=5,
            z_scale=30,
            critic_orig=False,
            is_box=False):
        super(WorldModel, self).__init__()
        self.v_model = v_model
        self.m_model = m_model
        self.c_model = c_model

        print("Input dimension for C model: {}".format(c_input_dim))
        # If is_box, this is the dimension of the world model inputs
        self.c_input_dim = c_input_dim
        self.device = self.c_model.device
        self.c_input_mode = c_input_mode
        self.critic_orig = critic_orig

        self.h_scale = h_scale
        self.z_scale = z_scale

        self.is_box = is_box
        if self.is_box:
            assert isinstance(
                self.c_model, NNConvModel), "Box world can only use NNConvModel"

    def parameters(self):
        return self.c_model.parameters()

    def state_dict(self):
        return self.c_model.state_dict()

    def train(self):
        self.c_model.train()

    def load_state_dict(self, state_dict):
        self.c_model.load_state_dict(state_dict)

    def prepare(self, world, trainer):
        self.c_model.prepare(
            world,
            trainer,
            world_models=True,
            input_dim=self.c_input_dim,
            critic_orig=self.critic_orig)

    def init(self, tasks_this_batch):
        self.c_model.init(tasks_this_batch)
        self.tasks_this_batch = tasks_this_batch
        self.m_model.reset(len(tasks_this_batch))

    def act(self, states, commands):
        # Preprocess with v_model and m_model
        exp_env = states[0].world
        if self.is_box:
            n_goals = exp_env.max_goal_length
            goal_start = 1
        else:
            n_goals = len(exp_env.grabbable_indices)
            goal_start = exp_env.grabbable_indices[0]
        batch_size = len(states)
        maps = []
        flat_inputs = []
        for i in range(len(states)):
            map, flat_input = states[i].features_world_model()
            task_input = make_one_hot(
                self.tasks_this_batch[i] - goal_start, C=n_goals)
            flat_input = np.concatenate((flat_input, task_input))

            maps.append(map)
            flat_inputs.append(flat_input)

        map_inputs = torch.tensor(
            maps, dtype=torch.float32).to(
            self.c_model.device)
        flat_inputs = torch.tensor(
            flat_inputs,
            dtype=torch.float32).to(
            self.c_model.device)
        zs = self.v_model(
            map_inputs,
            flat_inputs,
            encode=True)  # (batch_size, d_z)

        # Get hidden from last step of m_model
        h, c = self.m_model.hidden  # (num_layers, batch_size, hidden_units)
        h_flatten = torch.transpose(h, 0, 1).view(batch_size, -1)
        c_flatten = torch.transpose(c, 0, 1).view(batch_size, -1)
        if self.c_input_mode == 'encoded':
            featurized_states = torch.cat(
                [zs, h_flatten, c_flatten], dim=1).detach()  # (batch_size, -1)

        elif self.c_input_mode == 'combined':
            raw_input = [state.features() for state in states]
            raw_input = torch.tensor(raw_input).to(self.device)
            featurized_states = torch.cat(
                [zs, h_flatten, c_flatten, raw_input], dim=1).detach()

        elif self.c_input_mode == 'scaled_h_only':
            if self.is_box:
                featurized_states = torch.cat(
                    [zs / self.z_scale, h_flatten / self.h_scale], dim=1).detach()
            else:
                raw_input = [state.features() for state in states]
                raw_input = torch.tensor(raw_input).to(self.device)
                featurized_states = torch.cat(
                    [zs / self.z_scale, h_flatten / self.h_scale, raw_input], dim=1).detach()

        else:
            raw_input = [state.features() for state in states]
            raw_input = torch.tensor(raw_input).to(self.device)
            featurized_states = raw_input

        # c_model forward
        if self.is_box:
            actions, log_probs, critic_scores, neg_entropies = self.c_model.act(
                states, commands, world_model_states=featurized_states, critic_orig=self.critic_orig)
        else:
            actions, log_probs, critic_scores, neg_entropies = self.c_model.act(
                featurized_states, commands, already_featurize=True, critic_orig=self.critic_orig)

        # Update m_model hidden
        actions_one_hot = [
            make_one_hot(
                action,
                C=exp_env.n_actions) for action in actions]
        actions_input = torch.tensor(
            actions_one_hot,
            dtype=torch.float32).to(
            self.device)
        combined = torch.cat(
            (zs, actions_input), dim=1).unsqueeze(0)  # (1, N, d)

        _ = self.m_model(combined, encode=True)

        return actions, log_probs, critic_scores, neg_entropies


def act_world_models(c_model, v_model, m_model, states, tasks):
    exp_env = states[0].world
    n_goals = len(exp_env.grabbable_indices)
    goal_start = exp_env.grabbable_indices[0]
    batch_size = len(states)

    maps = []
    flat_inputs = []
    for i in range(len(states)):
        map, flat_input = states[i].features_world_model()
        task_input = make_one_hot(tasks[i] - goal_start, C=n_goals)
        flat_input = np.concatenate((flat_input, task_input))

        maps.append(map)
        flat_inputs.append(flat_input)

    map_inputs = torch.tensor(maps, dtype=torch.float32)
    flat_inputs = torch.tensor(flat_inputs, dtype=torch.float32)
    zs = v_model(map_inputs, flat_inputs, encode=True)  # (batch_size, d_z)

    # Get hidden from last step of m_model
    h, c = m_model.hidden  # (num_layers, batch_size, hidden_units)
    h_flatten = torch.transpose(h, 0, 1).view(batch_size, -1)
    c_flatten = torch.transpose(c, 0, 1).view(batch_size, -1)
    ctrl_input = torch.cat([zs, h_flatten, c_flatten],
                           dim=1)  # (batch_size, -1)

    # Forward c_model, get actions
    scores = c_model(ctrl_input)  # (N, num_actions)
    probs = F.softmax(scores, dim=1).detach().cpu().numpy()
    actions = [np.random.choice(exp_env.n_actions, p=probs[i, :])
               for i in range(batch_size)]

    # Update m_model hidden
    actions_one_hot = [
        make_one_hot(
            action,
            C=exp_env.n_actions) for action in actions]
    actions_input = torch.tensor(actions_one_hot, dtype=torch.float32)
    combined = torch.cat((zs, actions_input), dim=1).unsqueeze(0)  # (1, N, d)

    _ = m_model(combined, encode=True)

    return actions
