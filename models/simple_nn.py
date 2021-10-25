import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict, namedtuple
from .modular_ac import MLP
from misc.util import make_one_hot


class Controller(nn.Module):
    def __init__(self, input_dim, action_space):
        super(Controller, self).__init__()

        self.fc1 = nn.Linear(input_dim, action_space)

    def forward(self, x):
        x = self.fc1(x)
        return x


class NNModel(nn.Module):
    def __init__(self, config):
        super(NNModel, self).__init__()
        self.world = None
        self.config = config
        self.d_hidden_actor = 128
        self.d_hidden_critic = 32
        self.device = torch.device(
            "cuda" if config.model.device == "cuda" else "cpu")

    def prepare(
            self,
            world,
            trainer,
            world_models=False,
            input_dim=100,
            critic_orig=False):
        assert self.world is None
        self.world = world
        self.trainer = trainer

        # Number of tasks (goals)
        self.n_tasks = len(trainer.tasks)
        # Number of actors, one per task
        self.n_modules = self.n_tasks
        # Number of low-level actions
        self.n_actions = world.n_actions
        if world_models:
            self.n_features = input_dim
        else:
            # Number of input features
            self.n_features = world.n_features

        # Create actor and critic networks
        self.actors = {}
        self.critics = {}

        for task in trainer.tasks:
            actor = MLP(
                self.n_features,
                self.n_actions,
                self.d_hidden_actor).to(
                self.device)
            self.actors[task] = actor
            self.add_module("actor_{}".format(task), actor)

            if critic_orig:
                critic_input_dim = world.n_features
            else:
                critic_input_dim = self.n_features
            critic = MLP(
                critic_input_dim,
                1,
                self.d_hidden_critic).to(
                self.device)
            self.critics[task] = critic
            self.add_module("critic_{}".format(task), critic)

    def init(self, tasks_this_batch):
        by_task = defaultdict(list)
        # Group by module
        for (i, task) in enumerate(tasks_this_batch):
            by_task[task].append(i)
        self.by_task = by_task
        self.tasks_this_batch = tasks_this_batch

    def featurize(self, state):
        return state.features()

    def act(
            self,
            states,
            commands,
            already_featurize=False,
            critic_orig=False):
        if already_featurize:
            featurized_states = states
        else:
            featurized_states = [self.featurize(state) for state in states]
        batch_size = len(states)

        actions = [None] * batch_size
        log_probs_chosen = torch.zeros(batch_size).to(self.device)
        neg_entropies = torch.zeros(batch_size).to(self.device)
        critic_scores = torch.zeros(batch_size).to(self.device)

        # Run forward for each module
        for task, indices in self.by_task.items():
            if already_featurize:
                packed_inputs = featurized_states[indices]
            else:
                # Pack states together
                packed_inputs = torch.tensor([featurized_states[i] for i in indices]).to(
                    self.device)  # (N_command, d_feature)
            actor = self.actors[task]
            scores = actor(packed_inputs)
            log_probs = F.log_softmax(scores, dim=1)  # (N_command, d_action)
            neg_entropy = torch.sum(
                torch.exp(log_probs) * log_probs,
                1)  # (N_command,)

            # Sample actions
            probs = np.exp(log_probs.detach().cpu().numpy())
            for j, (pr, i) in enumerate(zip(probs, indices)):
                if not commands[i] is None:
                    if self.training or True:
                        a = np.random.choice(self.n_actions, p=pr)
                    else:
                        # print("Choose most probable action")
                        a = np.argmax(pr)
                    actions[i] = a
                    log_probs_chosen[i] = log_probs[j, a]
                    neg_entropies[i] = neg_entropy[j]

            # Run critics
            critic = self.critics[task]
            if critic_orig:
                scores_for_task = critic(
                    packed_inputs[:, -self.world.n_features:])
            else:
                scores_for_task = critic(packed_inputs)

            for j, i in enumerate(indices):
                critic_scores[i] = scores_for_task[j]

        return actions, log_probs_chosen.unsqueeze(
            1), critic_scores.unsqueeze(1), neg_entropies.unsqueeze(1)


class NNConvModel(nn.Module):
    def __init__(self, config):
        super(NNConvModel, self).__init__()
        self.world = None
        self.config = config
        self.d_hidden_actor = 128
        self.d_hidden_critic = 32
        self.conv_dim = 32
        self.device = torch.device(
            "cuda" if config.model.device == "cuda" else "cpu")
        self.modulate = self.config.model.modulate

    '''
    Here, the input_dim is the dimension of the inputs from V and M models
    '''

    def prepare(
            self,
            world,
            trainer,
            world_models=False,
            input_dim=100,
            critic_orig=False):
        assert self.world is None
        self.world = world
        self.trainer = trainer

        # Number of tasks (goals)
        self.n_tasks = len(trainer.tasks)
        # Number of low-level actions
        self.n_actions = world.n_actions

        if self.modulate:
            self.grid_feat_dim = world.n_things + 2  # Additional dimensions: dot with key
        else:
            self.grid_feat_dim = world.n_things + 1

        if world_models:
            self.raw_n_features = (world.grid_feat_size - 2) * \
                (world.grid_feat_size - 2) * self.conv_dim + world.n_things
            self.n_features = input_dim + self.raw_n_features
        else:
            # Number of input features
            self.n_features = (world.grid_feat_size - 2) * \
                (world.grid_feat_size - 2) * self.conv_dim + world.n_things

        # CNN input layer, H -> H - 2
        self.conv1 = nn.Conv2d(self.grid_feat_dim, self.conv_dim, 3).to(
            self.device)

        # Create actor and critic networks
        self.actors = {}
        self.critics = {}

        actor = MLP(
            self.n_features,
            self.n_actions,
            self.d_hidden_actor).to(
            self.device)
        self.add_module("actor", actor)
        for task in trainer.tasks:
            self.actors[task] = actor

            if critic_orig:
                assert world_models, "critic_orig can only be used in world models"
                critic_input_dim = self.raw_n_features
            else:
                critic_input_dim = self.n_features
            critic = MLP(
                critic_input_dim,
                1,
                self.d_hidden_critic).to(
                self.device)
            self.critics[task] = critic
            self.add_module("critic_{}".format(task), critic)

    def init(self, tasks_this_batch):
        by_task = defaultdict(list)
        # Group by module
        for (i, task) in enumerate(tasks_this_batch):
            by_task[task].append(i)
        self.by_task = by_task
        self.tasks_this_batch = tasks_this_batch

    def featurize(self, states):
        maps = []
        flat_inputs = []
        batch_size = len(states)
        for i in range(batch_size):
            raw_feats = states[i].features()
            if self.modulate:
                # Get key product
                key_vec = np.zeros(self.world.n_things + 1)
                key_vec[:self.world.n_things] = raw_feats['key_feats']
                key_map = np.dot(raw_feats['grid_feats'], key_vec)
                key_map = np.expand_dims(key_map, axis=2)

                modulated = np.concatenate(
                    (raw_feats['grid_feats'], key_map), axis=2)
                maps.append(np.transpose(modulated, (2, 0, 1)))
            else:
                maps.append(np.transpose(raw_feats['grid_feats'], (2, 0, 1)))

            flat_inputs.append(raw_feats['key_feats'])

        map_inputs = torch.tensor(maps, dtype=torch.float32).to(self.device)
        flat_inputs = torch.tensor(
            flat_inputs,
            dtype=torch.float32).to(
            self.device)  # (N, d)
        z = self.conv1(map_inputs)  # (N, C, H, W)
        z = F.relu(z)
        processed = torch.cat(
            [z.view(batch_size, -1), flat_inputs], dim=1)  # (N, d')

        return processed

    def act(
            self,
            states,
            commands,
            world_model_states=None,
            critic_orig=False):
        featurized_states = self.featurize(states)
        if world_model_states is not None:
            concat_states = torch.cat(
                [featurized_states, world_model_states], dim=1)
        batch_size = len(states)

        actions = [None] * batch_size
        log_probs_chosen = torch.zeros(batch_size).to(self.device)
        neg_entropies = torch.zeros(batch_size).to(self.device)
        critic_scores = torch.zeros(batch_size).to(self.device)

        # Run forward for each module
        for task, indices in self.by_task.items():
            if world_model_states is not None:
                packed_inputs = concat_states[indices]
            else:
                packed_inputs = featurized_states[indices]
            actor = self.actors[task]
            scores = actor(packed_inputs)
            log_probs = F.log_softmax(scores, dim=1)  # (N_command, d_action)
            neg_entropy = torch.sum(
                torch.exp(log_probs) * log_probs,
                1)  # (N_command,)

            # Sample actions
            probs = np.exp(log_probs.detach().cpu().numpy())
            for j, (pr, i) in enumerate(zip(probs, indices)):
                if not commands[i] is None:
                    if self.training or True:
                        a = np.random.choice(self.n_actions, p=pr)
                    else:
                        # print("Choose most probable action")
                        a = np.argmax(pr)
                    actions[i] = a
                    log_probs_chosen[i] = log_probs[j, a]
                    neg_entropies[i] = neg_entropy[j]

            # Run critics
            critic = self.critics[task]
            if critic_orig:
                packed_inputs = featurized_states[indices]
                scores_for_task = critic(packed_inputs)
            else:
                scores_for_task = critic(packed_inputs)

            for j, i in enumerate(indices):
                critic_scores[i] = scores_for_task[j]

        return actions, log_probs_chosen.unsqueeze(
            1), critic_scores.unsqueeze(1), neg_entropies.unsqueeze(1)


class ConditionModel(nn.Module):
    def __init__(self, config):
        super(ConditionModel, self).__init__()
        self.world = None
        self.config = config
        self.d_hidden_actor = 128
        self.d_hidden_critic = 32
        self.device = torch.device(
            "cuda" if config.model.device == "cuda" else "cpu")

    def prepare(self, world, trainer):
        assert self.world is None
        self.world = world
        self.trainer = trainer

        # Number of tasks (goals)
        self.n_tasks = len(trainer.tasks)
        # Number of high-level commands
        self.n_modules = len(world.action_indices)
        self.command_to_idx = {command: i for (i, command) in enumerate(world.action_indices)}
        # Number of low-level actions
        self.n_actions = world.n_actions
        # Number of input features
        self.n_features = world.n_features

        self.critics = {}

        self.actor = MLP(
            self.n_features + self.n_modules,
            self.n_actions,
            self.d_hidden_actor).to(
            self.device)

        for task in trainer.tasks:
            critic = MLP(
                self.n_features,
                1,
                self.d_hidden_critic).to(
                self.device)
            self.critics[task] = critic
            self.add_module("critic_{}".format(task), critic)

    def init(self, tasks_this_batch):
        by_task = defaultdict(list)
        # Group by module
        for (i, task) in enumerate(tasks_this_batch):
            by_task[task].append(i)
        self.by_task = by_task
        self.tasks_this_batch = tasks_this_batch

    def featurize(self, state):
        return state.features()

    def act(self, states, commands):
        featurized_states = [self.featurize(state) for state in states]
        batch_size = len(commands)

        # Group by module
        by_mod = defaultdict(list)
        for (i, command) in enumerate(commands):
            if command is not None:
                by_mod[command].append(i)

        actions = [None] * batch_size
        log_probs_chosen = torch.zeros(batch_size).to(self.device)
        neg_entropies = torch.zeros(batch_size).to(self.device)

        # Run forward for each module
        for command, indices in by_mod.items():
            # Pack states together
            packed_inputs = torch.tensor([featurized_states[i] for i in indices]).to(
                self.device)  # (N_command, d_feature)
            n_command = len(indices)
            cond_inp = torch.zeros(n_command, self.n_modules).to(self.device)
            cond_inp[:, self.command_to_idx[command]] = 1
            packed_inputs = torch.cat((packed_inputs, cond_inp), 1)

            scores = self.actor(packed_inputs)
            log_probs = F.log_softmax(scores, dim=1)  # (N_command, d_action)
            neg_entropy = torch.sum(
                torch.exp(log_probs) * log_probs,
                1)  # (N_command,)

            # Sample actions
            probs = np.exp(log_probs.detach().cpu().numpy())
            for j, (pr, i) in enumerate(zip(probs, indices)):
                if self.training or True:
                    a = np.random.choice(self.n_actions, p=pr)
                else:
                    # print("Choose most probable action")
                    a = np.argmax(pr)
                actions[i] = a
                log_probs_chosen[i] = log_probs[j, a]
                neg_entropies[i] = neg_entropy[j]

        critic_scores = torch.zeros(batch_size).to(self.device)
        # Run forward for each critic
        for task, indices in self.by_task.items():
            # Pack states together
            packed_inputs = torch.tensor([featurized_states[i] for i in indices]).to(
                self.device)  # (N_command, d_feature)
            critic = self.critics[task]
            scores_for_task = critic(packed_inputs)

            # Sample actions
            for j, i in enumerate(indices):
                critic_scores[i] = scores_for_task[j]

        return actions, log_probs_chosen.unsqueeze(
            1), critic_scores.unsqueeze(1), neg_entropies.unsqueeze(1)
