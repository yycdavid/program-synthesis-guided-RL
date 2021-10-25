import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict, namedtuple


class MLP(nn.Module):
    def __init__(self, d_in, d_out, d_hidden, num_hidden=1):
        super(MLP, self).__init__()
        self.fc_layers = []
        assert num_hidden >= 1, "MLP number of hidden layers should be at least 1"

        self.fc_layers.append(nn.Linear(d_in, d_hidden))
        for _ in range(num_hidden - 1):
            self.fc_layers.append(nn.Linear(d_hidden, d_hidden))
        self.fc_layers.append(nn.Linear(d_hidden, d_out))
        self.n_layers = len(self.fc_layers)

        for idx, module in enumerate(self.fc_layers):
            self.add_module(str(idx), module)

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.fc_layers[i](x)
            x = F.relu(x)

        x = self.fc_layers[-1](x)
        return x


class ModularACModel(nn.Module):
    def __init__(self, config):
        super(ModularACModel, self).__init__()
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
        # Number of low-level actions
        self.n_actions = world.n_actions
        # Number of input features
        self.n_features = world.n_features

        # Create actor and critic networks
        self.actors = {}
        self.critics = {}

        for action in world.action_indices:
            actor = MLP(
                self.n_features,
                self.n_actions,
                self.d_hidden_actor).to(
                self.device)
            self.actors[action] = actor
            self.add_module("actor_{}".format(action), actor)

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
            actor = self.actors[command]
            scores = actor(packed_inputs)
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


class ModularACConvModel(nn.Module):
    def __init__(self, config):
        super(ModularACConvModel, self).__init__()
        self.world = None
        self.config = config
        self.d_hidden_actor = 128
        self.d_hidden_critic = 32
        self.conv_dim = 32
        self.device = torch.device(
            "cuda" if config.model.device == "cuda" else "cpu")
        self.modulate = self.config.model.modulate

    def prepare(self, world, trainer):
        assert self.world is None
        self.world = world
        self.trainer = trainer

        # Number of tasks (goals)
        self.n_tasks = len(trainer.tasks)
        # Number of high-level commands
        self.n_modules = len(world.action_indices)
        # Number of low-level actions
        self.n_actions = world.n_actions
        # Number of input features
        if self.modulate:
            # Additional dimensions: dot with key, dot with command
            self.grid_feat_dim = world.n_things + 3
        else:
            self.grid_feat_dim = world.n_things + 1
        self.n_features = (world.grid_feat_size - 2) * \
            (world.grid_feat_size - 2) * self.conv_dim + world.n_things

        # CNN input layer, H -> H - 2
        self.conv1 = nn.Conv2d(self.grid_feat_dim, self.conv_dim, 3).to(
            self.device)

        # Create actor and critic networks
        self.actors = {}
        self.critics = {}

        if self.modulate:
            actor = MLP(
                self.n_features,
                self.n_actions,
                self.d_hidden_actor).to(
                self.device)
            self.add_module("actor", actor)

            for action in world.action_indices:
                self.actors[action] = actor

        else:
            for action in world.action_indices:
                actor = MLP(
                    self.n_features,
                    self.n_actions,
                    self.d_hidden_actor).to(
                    self.device)
                self.actors[action] = actor
                self.add_module("actor_{}".format(action), actor)

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

    def featurize(self, states, commands):
        maps = []
        flat_inputs = []
        batch_size = len(states)
        for i in range(batch_size):
            raw_feats = states[i].features()
            if self.modulate:
                # Get command product
                command_vec = np.zeros(self.world.n_things + 1)
                if not commands[i] is None:
                    command_vec[commands[i]] = 1
                com_map = np.dot(raw_feats['grid_feats'], command_vec)
                com_map = np.expand_dims(com_map, axis=2)

                # Get key product
                key_vec = np.zeros(self.world.n_things + 1)
                key_vec[:self.world.n_things] = raw_feats['key_feats']
                key_map = np.dot(raw_feats['grid_feats'], key_vec)
                key_map = np.expand_dims(key_map, axis=2)

                modulated = np.concatenate(
                    (raw_feats['grid_feats'], com_map, key_map), axis=2)
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

    def act(self, states, commands):
        featurized_states = self.featurize(states, commands)
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
            # (N_command, d_feature)
            packed_inputs = featurized_states[indices]
            actor = self.actors[command]
            scores = actor(packed_inputs)
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
            # (N_command, d_feature)
            packed_inputs = featurized_states[indices]
            critic = self.critics[task]
            scores_for_task = critic(packed_inputs)

            # Sample actions
            for j, i in enumerate(indices):
                critic_scores[i] = scores_for_task[j]

        return actions, log_probs_chosen.unsqueeze(
            1), critic_scores.unsqueeze(1), neg_entropies.unsqueeze(1)
