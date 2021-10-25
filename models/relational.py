import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import defaultdict, namedtuple
from .modular_ac import MLP
import math


class RelationalModel(nn.Module):
    def __init__(self, config):
        super(RelationalModel, self).__init__()
        self.world = None
        self.config = config
        if hasattr(config.model, "original"):
            # Use the original architecture proposed in DRRL
            self.use_orig = config.model.original
        else:
            self.use_orig = False
        if self.use_orig:
            self.d_hidden_actor = 256
            self.conv_dim_1 = 12
            self.conv_dim = 24
        else:
            self.conv_dim = 32
            self.d_hidden_actor = 128
        self.d_hidden_critic = 32
        self.device = torch.device(
            "cuda" if config.model.device == "cuda" else "cpu")
        self.modulate = self.config.model.modulate
        self.n_att_stack = config.model.n_layer
        self.n_heads = config.model.n_heads
        self.att_emb_size = 64
        if hasattr(config.model, "flat"):
            self.use_flat = config.model.flat
        else:
            self.use_flat = False

    def prepare(self, world, trainer):
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

        if self.use_orig:
            # CNN input layer, H -> H - 1
            self.conv1 = nn.Conv2d(
                self.grid_feat_dim,
                self.conv_dim_1,
                2).to(
                self.device)
            self.conv2 = nn.Conv2d(
                self.conv_dim_1,
                self.conv_dim,
                2).to(
                self.device)
        else:
            # CNN input layer, H -> H - 2
            self.conv1 = nn.Conv2d(
                self.grid_feat_dim,
                self.conv_dim,
                3).to(
                self.device)

        # coordinates to be appended to conv outputs
        out_conv_size = world.grid_feat_size - 2
        xmap = np.linspace(-np.ones(out_conv_size),
                           np.ones(out_conv_size),
                           num=out_conv_size,
                           endpoint=True,
                           axis=0)
        xmap = torch.tensor(
            np.expand_dims(
                np.expand_dims(
                    xmap,
                    0),
                0),
            dtype=torch.float32,
            requires_grad=False)  # (1,1,H,W)
        ymap = np.linspace(-np.ones(out_conv_size),
                           np.ones(out_conv_size),
                           num=out_conv_size,
                           endpoint=True,
                           axis=1)
        ymap = torch.tensor(
            np.expand_dims(
                np.expand_dims(
                    ymap,
                    0),
                0),
            dtype=torch.float32,
            requires_grad=False)  # (1,1,H,W)
        self.register_buffer(
            "xymap", torch.cat(
                (xmap, ymap), dim=1).to(
                self.device))  # shape (1, 2, out_conv_size, out_conv_size)

        # Attention modules
        att_elem_size = self.conv_dim + 2  # Add 2 coordinates
        self.attMod = AttentionModule(
            out_conv_size *
            out_conv_size,
            att_elem_size,
            self.att_emb_size,
            self.n_heads).to(
            self.device)

        # Create actor and critic networks
        self.actors = {}
        self.critics = {}
        # Number of input features
        if self.use_flat:
            self.n_features = att_elem_size + world.n_things
        else:
            self.n_features = att_elem_size

        if self.use_orig:
            actor = MLP(
                self.n_features,
                self.n_actions,
                self.d_hidden_actor,
                num_hidden=4).to(
                self.device)
        else:
            actor = MLP(
                self.n_features,
                self.n_actions,
                self.d_hidden_actor).to(
                self.device)
        actor = MLP(
            self.n_features,
            self.n_actions,
            self.d_hidden_actor).to(
            self.device)
        self.add_module("actor", actor)
        for task in trainer.tasks:
            self.actors[task] = actor

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
        if self.use_orig:
            z = self.conv2(z)
        z = F.relu(z)
        return z, flat_inputs

    def act(self, states, commands):
        # Modulate, and conv encode map
        # featurized_maps: (N, C, H, W), flat_inputs: (N, d)
        featurized_maps, flat_inputs = self.featurize(states)

        # Append coordinates
        batch_size = len(states)
        batch_maps = self.xymap.repeat(batch_size, 1, 1, 1)
        c = torch.cat((featurized_maps, batch_maps), 1)  # (N, C, H, W)

        # Relational module, max pool in the end
        a = c.view(c.size(0), c.size(1), -1).transpose(1, 2)  # (N, H*W, C)
        for i_att in range(self.n_att_stack):
            a = self.attMod(a)
        kernelsize = a.shape[1]
        if isinstance(kernelsize, torch.Tensor):
            kernelsize = kernelsize.item()
        # pool out entity dimension, (N, C, 1)
        pooled = F.max_pool1d(a.transpose(1, 2), kernel_size=kernelsize)
        featurized_states = pooled.squeeze(2)  # (N, C)
        if self.use_flat:
            featurized_states = torch.cat((featurized_states, flat_inputs), 1)

        # Pass to actor and critic
        actions = [None] * batch_size
        log_probs_chosen = torch.zeros(batch_size).to(self.device)
        neg_entropies = torch.zeros(batch_size).to(self.device)
        critic_scores = torch.zeros(batch_size).to(self.device)

        # Run forward for each module
        for task, indices in self.by_task.items():
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
            scores_for_task = critic(packed_inputs)

            for j, i in enumerate(indices):
                critic_scores[i] = scores_for_task[j]

        return actions, log_probs_chosen.unsqueeze(
            1), critic_scores.unsqueeze(1), neg_entropies.unsqueeze(1)


class AttentionModule(nn.Module):

    def __init__(self, n_elems, elem_size, emb_size, n_heads):
        super(AttentionModule, self).__init__()
        self.heads = nn.ModuleList(
            AttentionHead(
                n_elems,
                elem_size,
                emb_size) for _ in range(n_heads))
        self.linear1 = nn.Linear(n_heads * elem_size, elem_size)
        self.linear2 = nn.Linear(elem_size, elem_size)

        self.ln = nn.LayerNorm(elem_size, elementwise_affine=True)

    def forward(self, x):
        # concatenate all heads' outputs
        A_cat = torch.cat([head(x) for head in self.heads], -1)
        # projecting down to original element size with 2-layer MLP, layer size
        # = entity size
        mlp_out = F.relu(
            self.linear2(
                F.relu(
                    self.linear1(A_cat))))  # (N, H*W, C)
        # residual connection and final layer normalization
        return self.ln(x + mlp_out)

    def get_att_weights(self, x):
        """Version of forward function that also returns softmax-normalied QK' attention weights"""
        # concatenate all heads' outputs
        A_cat = torch.cat([head(x) for head in self.heads], -1)
        # projecting down to original element size with 2-layer MLP, layer size
        # = entity size
        mlp_out = F.relu(self.linear2(F.relu(self.linear1(A_cat))))
        # residual connection and final layer normalization
        output = self.ln(x + mlp_out)
        attention_weights = [head.attention_weights(
            x).detach() for head in self.heads]
        return [output, attention_weights]


class AttentionHead(nn.Module):
    def __init__(self, n_elems, elem_size, emb_size):
        super(AttentionHead, self).__init__()
        self.sqrt_emb_size = int(math.sqrt(emb_size))
        #queries, keys, values
        self.query = nn.Linear(elem_size, emb_size)
        self.key = nn.Linear(elem_size, emb_size)
        self.value = nn.Linear(elem_size, elem_size)
        # layer norms:
        # In the paper the authors normalize the projected Q,K and V with layer normalization. They don't state
        # explicitly over which dimensions they normalize and how exactly gains and biases are shared. I decided to
        # stick with with the solution from https://github.com/gyh75520/Relational_DRL/ because it makes the most
        # sense to me: 0,1-normalize every projected entity and apply separate gain and bias to each entry in the
        # embeddings. Weights are shared across entites, but not accross Q,K,V
        # or heads.
        self.qln = nn.LayerNorm(emb_size, elementwise_affine=True)
        self.kln = nn.LayerNorm(emb_size, elementwise_affine=True)
        self.vln = nn.LayerNorm(elem_size, elementwise_affine=True)

    def forward(self, x):
        # print(f"input: {x.shape}")
        Q = self.qln(self.query(x))
        K = self.kln(self.key(x))
        V = self.vln(self.value(x))
        # softmax is taken over last dimension (rows) of QK': All the attentional weights going into a column/entity
        # of V thus sum up to 1.
        softmax = F.softmax(
            torch.bmm(
                Q,
                K.transpose(
                    1,
                    2)) /
            self.sqrt_emb_size,
            dim=-
            1)
        # print(f"softmax shape: {softmax.shape} and sum accross batch 1, column 1: {torch.sum(softmax[0,0,:])}")
        return torch.bmm(softmax, V)

    def attention_weights(self, x):
        # print(f"input: {x.shape}")
        Q = self.qln(self.query(x))
        K = self.kln(self.key(x))
        V = self.vln(self.value(x))
        # softmax is taken over last dimension (rows) of QK': All the attentional weights going into a column/entity
        # of V thus sum up to 1.
        softmax = F.softmax(
            torch.bmm(
                Q,
                K.transpose(
                    1,
                    2)) /
            self.sqrt_emb_size,
            dim=-
            1)
        return softmax
