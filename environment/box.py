import numpy as np
import time
from skimage.measure import block_reduce
from misc import array
import copy
from collections import defaultdict
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

import math
import pickle
import pygame
import pygame.freetype
from tqdm import tqdm
from .craft import Commander

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"


# this version does not use black as a key color because it's used for the
# board's outline
COLORS = {
    0: [0, 0, 117],
    1: [230, 190, 255],
    2: [170, 255, 195],
    3: [255, 250, 200],
    4: [255, 216, 177],
    5: [250, 190, 190],
    6: [240, 50, 230],
    7: [145, 30, 180],
    8: [67, 99, 216],
    9: [66, 212, 244],
    10: [60, 180, 75],
    11: [191, 239, 69],
    12: [255, 255, 25],
    13: [245, 130, 49],
    14: [230, 25, 75],
    15: [128, 0, 0],
    16: [154, 99, 36],
    17: [128, 128, 0],
    18: [70, 153, 144],
    19: [100, 70, 0]
}

for key in COLORS.keys():
    COLORS[key] = np.array(COLORS[key], dtype=np.uint8)
AGENT_COLOR = np.array([128, 128, 128], dtype=np.uint8)
GOAL_COLOR = np.array([255, 255, 255], dtype=np.uint8)
BACKGD_COLOR = np.array([220, 220, 220], dtype=np.uint8)

ACTION_LOOKUP = {
    0: 'move up',
    1: 'move down',
    2: 'move left',
    3: 'move right',
}

CHANGE_COORDINATES = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}


def make_one_hot(label, C=10):
    one_hot = np.zeros(C)
    one_hot[label] = 1

    return one_hot


'''
Return a generated abstract state, given a generative model, and a partially observed state
'''


def hallucinate(world, generator, state, goal_length):
    map_width = world.n
    map_height = world.n
    n_goals = world.max_goal_length

    # Prepare conditional inputs
    def to_torch(t): return torch.tensor(t, dtype=torch.float32)

    abstract_partial_map = world.get_abstract_partial_state(state)
    encoded_partial = world.encode_abstract_partial_state(abstract_partial_map)
    abstract_partial_state = to_torch(encoded_partial)

    task = to_torch(make_one_hot(goal_length - 1, C=n_goals))
    unobs_fractions = to_torch(
        [1.0 - np.count_nonzero(state.mask) / (map_width * map_height)])
    cond_inp = torch.cat(
        (abstract_partial_state, task, unobs_fractions)).unsqueeze(0)

    # Run generative model
    z = Variable(torch.FloatTensor(1, generator.drep).normal_())
    z = torch.cat((z, cond_inp), -1)
    recon = generator.decode(z)

    halluc_abs_state = world.parse_abstract_state(recon[0])
    return halluc_abs_state


class BoxWorld(object):
    '''
    coordinate: x is downwards, y is rightwards
    '''

    def __init__(self, config, seed):
        self.n = 12
        self.distractor_length = 1
        self.max_num_distractor = 4
        self.max_goal_length = 4
        self.max_num_same_box = self.max_num_distractor

        if hasattr(config.world, "view_range"):
            self.VIEW_RANGE = config.world.view_range
        else:
            self.VIEW_RANGE = 3
        self.n_actions = 4

        # Penalties and Rewards
        self.reward_gem = 10
        self.FINAL_REWARD = self.reward_gem
        self.reward_dead = 0
        self.reward_correct_key = 1
        self.reward_wrong_key = -1

        self.random = np.random.RandomState(seed)

        self.num_colors = 10
        self.goal_color_id = self.num_colors
        self.colors = {}
        for i in range(self.num_colors):
            self.colors[i] = COLORS[i]

        self.action_indices = list(self.colors.keys())
        self.action_indices.append(self.goal_color_id)
        self.n_things = self.num_colors + 1
        self.grid_feat_size = self.n * 2 - 1

        self.set_world_models_consts()

    def set_world_models_consts(self):
        # Dimensions for world model features
        # 1 for partial observable mask, 1 for agent pos
        self.map_feature_dim = self.n_things + 2
        self.key_feature_dim = self.n_things
        self.task_feature_dim = self.max_goal_length
        self.flat_feature_dim = self.task_feature_dim + self.key_feature_dim

    '''
    goal: is the length of the goal path, how many keys are needed
    '''

    def sample_scenario_with_goal(
            self,
            goal_length,
            plot_solution=False,
            plot_dir='',
            dangle_box=False):
        # Randomly sample number of distractors
        num_distractor = self.random.choice(self.max_num_distractor - 1) + 1

        # Sample initial key color, and colors for intermediate goals
        # goal_colors[0] is initial key color
        goal_colors = self.random.choice(
            self.num_colors, size=goal_length, replace=False)
        # pick colors for distractors
        distractor_possible_colors = [color for color in range(
            len(self.colors)) if color not in goal_colors]
        distractor_colors = [
            self.random.choice(
                distractor_possible_colors,
                size=self.distractor_length,
                replace=False) for _ in range(num_distractor)]

        num_dangles = 0
        if dangle_box:
            used_colors_set = set()
            for d_colors in distractor_colors:
                for c in d_colors:
                    used_colors_set.add(c)
            for c in goal_colors:
                used_colors_set.add(c)

            dangle_possible_colors = [color for color in range(
                len(self.colors)) if color not in used_colors_set]

            assert len(
                dangle_possible_colors) > 0, "No available colors for dangling boxes"
            num_dangles = 2
            dangle_colors = self.random.choice(
                dangle_possible_colors, size=num_dangles, replace=True)
            dangle_target = self.random.choice(
                goal_length, size=num_dangles, replace=True)

        # sample where to branch off distractor branches from goal path
        # this line mainly prevents arbitrary distractor path length
        distractor_roots = self.random.choice(goal_length, size=num_distractor)

        # Sample locations of pairs
        keys, locks, agent_pos = self.sample_pair_locations(
            goal_length + self.distractor_length * num_distractor + num_dangles)

        # rudimentary plot of solution DAG
        if plot_solution:
            self.plot_solution_graph(
                goal_colors,
                distractor_colors,
                distractor_roots,
                plot_dir=plot_dir)

        # Create map
        grid = np.zeros((self.n, self.n, self.num_colors + 1))
        init_key = goal_colors[0]
        # create the goal path
        for i in range(goal_length):
            if i == goal_length - 1:
                key_id = self.goal_color_id  # Final key
            else:
                key_id = goal_colors[i + 1]

            lock_id = goal_colors[i]
            grid[keys[i][0], keys[i][1], key_id] = 1
            grid[locks[i][0], locks[i][1], lock_id] = 1

        # a dead end is the end of a distractor branch, saved as color so it's consistent with world representation
        # iterate over distractor branches to place all distractors
        dead_ends = []
        for i, (distractor_color, root) in enumerate(
                zip(distractor_colors, distractor_roots)):
            # choose x,y locations for keys and locks from keys and locks
            # (previously determined so nothing collides)
            key_distractor = keys[goal_length +
                                  i * self.distractor_length: goal_length +
                                  (i + 1) * self.distractor_length]
            lock_distractor = locks[goal_length +
                                    i *self.distractor_length: goal_length +
                                    (i + 1) * self.distractor_length]
            # determine colors and place key,lock-pairs
            for k, (key, lock) in enumerate(
                    list(zip(key_distractor, lock_distractor))):
                if k == 0:  # first lock has color of root of distractor branch
                    lock_id = goal_colors[root]
                else:
                    lock_id = distractor_color[k - 1]
                key_id = distractor_color[k]
                grid[key[0], key[1], key_id] = 1
                grid[lock[0], lock[1], lock_id] = 1
            # after loop is run through the remaining key_id is the dead end
            dead_ends.append(key_id)

        if dangle_box:
            # Sample dangling boxes
            key_dangle = keys[-num_dangles:]
            lock_dangle = locks[-num_dangles:]
            for i, (key, lock) in enumerate(
                    list(zip(key_dangle, lock_dangle))):
                lock_id = dangle_colors[i]
                if dangle_target[i] == goal_length - 1:
                    key_id = self.goal_color_id
                else:
                    key_id = goal_colors[dangle_target[i] + 1]
                grid[key[0], key[1], key_id] = 1
                grid[lock[0], lock[1], lock_id] = 1

        return BoxScenario(
            grid,
            agent_pos,
            self,
            init_key,
            dead_ends,
            goal_colors)

    def sample_pair_locations(self, num_pair):
        """Generates random key,lock pairs locations in the environment.
        Makes sure the objects don't collide and everything is reachable by the agent.
        The locations can be filled later on with correct or distractor colors.
        Args:
            num_pair: number of key-lock pairs to be generated, results from length of correct path + length of all
            distractor branches.
        Returns:
            x,y-coordinates of all keys, locks, and agent.
        """
        n = self.n  # size of the board excluding boundary
        possibilities = set(range(1, n * (n - 1)))
        keys = []
        locks = []
        for k in range(num_pair):
            key = self.random.choice(list(possibilities))
            key_x, key_y = key // (n - 1), key % (n - 1)
            lock_x, lock_y = key_x, key_y + 1
            to_remove = [key_x * (n - 1) + key_y] + \
                        [key_x * (n - 1) + i + key_y for i in range(1, min(2, n - 2 - key_y) + 1)] + \
                        [key_x * (n - 1) - i + key_y for i in range(1, min(2, key_y) + 1)]

            possibilities -= set(to_remove)
            keys.append([key_x, key_y])
            locks.append([lock_x, lock_y])

        agent_pos = self.random.choice(list(possibilities))
        agent_pos = np.array([agent_pos // (n - 1), agent_pos % (n - 1)])
        return keys, locks, agent_pos

    def plot_solution_graph(
            self,
            goal_colors,
            distractor_colors,
            distractor_roots,
            plot_dir='',
            file_name='solution_graph.png'):
        """Plots game problem as directed graph of colors that were used to render a given environment.
        Not very pretty yet.
        """
        colors = self.colors
        vis = np.ones([len(distractor_roots) + 1,
                       max(len(goal_colors),
                           max([distractor_roots[path] + len(distractor_colors[path]) for path in range(len(
                               distractor_roots))])) + 1, 3], dtype=int) * BACKGD_COLOR[0]  # length of longest path
        for i, col in enumerate(goal_colors):
            vis[0, i, :] = colors[col]
        vis[0, len(goal_colors), :] = GOAL_COLOR
        for j, dist_branch in enumerate(distractor_colors):
            for i_raw, dist_col in enumerate(dist_branch):
                i = i_raw + distractor_roots[j] + 1
                vis[j + 1, i, :] = colors[dist_col]
        plt.title("problem graph")
        plt.imshow(vis)
        plt.yticks(ticks=list(range(len(distractor_roots) +
                                    1)), labels=["solution path"] +
                   [f"distractor path {i + 1}" for i in range(len(distractor_roots))])
        plt.xticks(list(range(len(goal_colors) + 1)))
        plt.xlabel("key #")
        save_file = os.path.join(plot_dir, file_name)
        plt.savefig(save_file, dpi=100)

    def convert_grid_to_image(self, grid):
        img = np.ones((self.n, self.n, 3), dtype=np.uint8) * BACKGD_COLOR

        # Convert colors
        for x in range(self.n):
            for y in range(self.n):
                if not grid[x, y, :].any():
                    continue
                here = grid[x, y, :]
                if here.sum() > 1:
                    print("impossible world configuration:")
                assert here.sum() == 1
                thing = here.argmax()
                if thing == self.goal_color_id:
                    img[x, y, :] = GOAL_COLOR
                else:
                    img[x, y, :] = self.colors[thing]

        # Pad
        img = np.pad(img, [(1, 1), (1, 1), (0, 0)])

        return img

    '''
    Visualize a state
    '''

    def visualize_pretty(
            self,
            state,
            save_dir,
            num=1,
            with_side=True,
            plan=None):
        # Convert grid to colored matrix
        img = self.convert_grid_to_image(state.grid)
        img = img.astype(np.uint8)

        # Add agent
        x, y = state.pos
        img[x + 1, y + 1, :] = AGENT_COLOR

        # Add current key
        if state.key > -1 and state.key != self.goal_color_id:
            img[0, 0, :] = self.colors[state.key]
        elif state.key == self.goal_color_id:
            img[0, 0, :] = GOAL_COLOR

        if plan is not None:
            fig, ax_all = plt.subplots(
                1, 2, gridspec_kw={
                    'width_ratios': [
                        5, 1]})
            ax = ax_all[0]
        else:
            fig, ax = plt.subplots()
        ax.imshow(img, vmin=0, vmax=255, interpolation='none')
        ax.set_axis_off()

        # Add mask
        mask = state.mask
        for i in range(self.n):
            for j in range(self.n):
                if mask[i, j] == 0:
                    ax.text(j + 1, i + 1, 'x',
                            ha="center", va="center",
                            color="black")

        if plan is not None:
            # Plot plan
            ax_plan = ax_all[1]
            plan_img = np.zeros((len(plan), 1, 3), dtype=np.uint8)
            for i, color in enumerate(plan):
                if color == self.goal_color_id:
                    plan_img[i, 0, :] = GOAL_COLOR
                else:
                    plan_img[i, 0, :] = self.colors[color]

            ax_plan.imshow(plan_img, vmin=0, vmax=255, interpolation='none')
            ax_plan.set_axis_off()

        save_file = os.path.join(save_dir, 'pic{}.png'.format(num))
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()

    '''
    Visualize an episode
    '''

    def pretty_rollout(self, rollout, save_dir, plans=None):
        for j, state in tqdm(enumerate(rollout)):
            if not plans[j] is None:
                plan = plans[j]['plan']
            else:
                plan = None
            self.visualize_pretty(state, save_dir, num=j, plan=plan)

        os.system('ffmpeg -y -r 2 -i {}/pic%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" {}/test.mp4'.format(save_dir, save_dir))

    '''
    Get the abstract state. Format:
    - loose: {color}
    - box: {(k1, k2): count}
    - key: color
    '''

    def get_abstract_state(self, state):
        abstract_state = {}
        abstract_state['key'] = state.key
        abstract_state['loose'] = set()
        abstract_state['box'] = defaultdict(lambda: 0)

        grid = state.grid
        cur_cell = 0
        while cur_cell < self.n * self.n:
            cur_x, cur_y = cur_cell // self.n, cur_cell % self.n
            if not grid[cur_x, cur_y, :].any():
                cur_cell += 1

            else:
                if cur_y == self.n - \
                        1 or (not grid[cur_x, cur_y + 1, :].any()):
                    # Loose key
                    abstract_state['loose'].add(grid[cur_x, cur_y, :].argmax())
                    cur_cell += 1
                else:
                    # Box
                    box = (grid[cur_x, cur_y, :].argmax(),
                           grid[cur_x, cur_y + 1, :].argmax())
                    abstract_state['box'][box] += 1
                    cur_cell += 2

        abstract_state['box'] = dict(abstract_state['box'])

        return abstract_state

    def encode_abstract_state(self, abstract_state):
        key_vec = np.zeros(self.n_things)
        if abstract_state['key'] != -1:
            key_vec[abstract_state['key']] = 1

        loose_vec = np.zeros(self.n_things)
        for thing in abstract_state['loose']:
            loose_vec[thing] = 1

        num_boxes = self.n_things * (self.n_things - 1)
        max_val = self.max_num_same_box
        box_vec = np.zeros(num_boxes * max_val)
        cur_idx = 0
        for k1 in range(self.n_things):
            for k2 in range(self.n_things):
                if k1 == k2:
                    continue
                if (k1, k2) in abstract_state['box']:
                    v = abstract_state['box'][(k1, k2)]
                    assert v <= max_val
                    box_vec[cur_idx:cur_idx + v] = 1
                cur_idx += max_val

        return np.concatenate((key_vec, loose_vec, box_vec))

    '''
    Parse the abstract state. Format:
    - loose: {color}
    - box: {(k1, k2): count}
    - key: color
    '''

    def parse_abstract_state(self, encoded_abstract):
        key_vec = encoded_abstract[:self.n_things]
        loose_vec = encoded_abstract[self.n_things:2 * self.n_things]
        max_val = self.max_num_same_box
        box_vec = encoded_abstract[2 * self.n_things:].reshape(-1, max_val)

        abstract_state = {}
        abstract_state['key'] = key_vec.argmax()
        abstract_state['loose'] = set()
        abstract_state['box'] = {}

        for thing in range(self.n_things):
            if loose_vec[thing] > 0.5:
                abstract_state['loose'].add(thing)

        ctr = 0
        for k1 in range(self.n_things):
            for k2 in range(self.n_things):
                if k1 == k2:
                    continue
                count = round(box_vec[ctr].sum().item())
                if count > 0:
                    abstract_state['box'][(k1, k2)] = count
                ctr += 1

        return abstract_state

    '''
    Get the abstract partial state. Format:
    - loose: {color}
    - box: {(k1, k2): count}
    - key: color
    - left_single: {color: count}
    - right_single: {color: count}
    '''

    def get_abstract_partial_state(self, state):
        abstract_state = {}
        abstract_state['key'] = state.key
        abstract_state['loose'] = set()
        abstract_state['box'] = defaultdict(lambda: 0)
        abstract_state['left_single'] = defaultdict(lambda: 0)
        abstract_state['right_single'] = defaultdict(lambda: 0)
        grid = state.grid
        mask = state.mask
        cur_cell = 0
        while cur_cell < self.n * self.n:
            cur_x, cur_y = cur_cell // self.n, cur_cell % self.n
            if not grid[cur_x, cur_y, :].any() or mask[cur_x, cur_y] == 0:
                cur_cell += 1
            else:
                # Visible, and have something
                if cur_y == 0:
                    if grid[cur_x, cur_y + 1, :].any():
                        # Box
                        box = (grid[cur_x, cur_y, :].argmax(),
                               grid[cur_x, cur_y + 1, :].argmax())
                        abstract_state['box'][box] += 1
                        cur_cell += 2
                    else:
                        # Loose key
                        abstract_state['loose'].add(
                            grid[cur_x, cur_y, :].argmax())
                        cur_cell += 1

                else:
                    if mask[cur_x, cur_y - 1] == 0:
                        if grid[cur_x, cur_y + 1, :].any():
                            # Box
                            box = (grid[cur_x, cur_y, :].argmax(),
                                   grid[cur_x, cur_y + 1, :].argmax())
                            abstract_state['box'][box] += 1
                            cur_cell += 2
                        else:
                            # Left side of visible region
                            abstract_state['left_single'][grid[cur_x,
                                                               cur_y, :].argmax()] += 1
                            cur_cell += 1
                    else:
                        # Not at left boundary, and not at left side of visible
                        # region
                        if cur_y == self.n - 1:
                            # Loose key
                            abstract_state['loose'].add(
                                grid[cur_x, cur_y, :].argmax())
                            cur_cell += 1
                        else:
                            if mask[cur_x, cur_y + 1] == 0:
                                # Could be a loose key, or a locked key. At the
                                # right side of visible region
                                abstract_state['right_single'][grid[cur_x,
                                                                    cur_y, :].argmax()] += 1
                                cur_cell += 1
                            else:
                                if grid[cur_x, cur_y + 1, :].any():
                                    # Box
                                    box = (grid[cur_x, cur_y, :].argmax(
                                    ), grid[cur_x, cur_y + 1, :].argmax())
                                    abstract_state['box'][box] += 1
                                    cur_cell += 2
                                else:
                                    # Loose key
                                    abstract_state['loose'].add(
                                        grid[cur_x, cur_y, :].argmax())
                                    cur_cell += 1

        abstract_state['box'] = dict(abstract_state['box'])
        abstract_state['left_single'] = dict(abstract_state['left_single'])
        abstract_state['right_single'] = dict(abstract_state['right_single'])

        return abstract_state

    def encode_abstract_partial_state(self, abstract_state):
        normal_part = {
            'key': abstract_state['key'],
            'loose': abstract_state['loose'],
            'box': abstract_state['box'],
        }
        normal_vec = self.encode_abstract_state(normal_part)
        left_vec = self._encode_single_counts(
            abstract_state['left_single'], max_val=5)
        right_vec = self._encode_single_counts(
            abstract_state['right_single'], max_val=5)
        return np.concatenate((normal_vec, left_vec, right_vec))

    '''
    Parse the abstract partial state. Format:
    - loose: {color}
    - box: {(k1, k2): count}
    - key: color
    - left_single: {color: count}
    - right_single: {color: count}
    '''

    def parse_abstract_partial_state(self, encoded_partial):
        tail_len = self.max_goal_length + 1
        partial_vec = encoded_partial[:-tail_len]

        pass

    def _encode_single_counts(self, counts, max_val=2):
        single_vec = np.zeros(self.n_things * max_val)
        cur_idx = 0
        for k1 in range(self.n_things):
            if k1 in counts:
                v = counts[k1]
                assert v <= max_val
                single_vec[cur_idx:cur_idx + v] = 1
            cur_idx += max_val

        return single_vec


class BoxScenario(object):
    def __init__(
            self,
            grid,
            init_pos,
            world,
            init_key,
            dead_ends,
            correct_keys):

        self.init_grid = grid
        self.init_pos = init_pos
        self.world = world
        self.init_key = init_key
        self.dead_ends = set(dead_ends)
        # This includes the initial key the agent holds, but not include the
        # white key (final goal)
        self.correct_keys = set(correct_keys)
        self.goal_length = len(self.correct_keys)

        # Mask for observed part, 1 means observed
        self.init_mask = self._initial_mask(init_pos)

    def _initial_mask(self, pos):
        mask = np.zeros((self.world.n, self.world.n))
        x, y = pos
        x_min, x_max, y_min, y_max = self._get_current_view(x, y)

        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                mask[i, j] = 1

        return mask

    def _get_current_view(self, x, y):
        x_min = max(x - self.world.VIEW_RANGE, 0)
        x_max = min(x + self.world.VIEW_RANGE, self.world.n - 1)
        y_min = max(y - self.world.VIEW_RANGE, 0)
        y_max = min(y + self.world.VIEW_RANGE, self.world.n - 1)
        return x_min, x_max, y_min, y_max

    '''
    Create initial state.

    - id: Managed by Trainer, only used for pre-sampled scenarios, unique id to identify this scenario.
    - For baseline that doesn't use synthesizer (world model), give synthesizer=None.
    '''

    def init(self, synthesizer, id=0, use_ground_truth=True):
        self.id = id
        state = BoxState(
            self.world,
            self.init_grid,
            self.init_pos,
            self.init_key,
            self.init_mask)
        self.current_state = state
        self.commander = None

        if synthesizer is not None:
            # Synthesize ground truth plan
            max_max_steps = self.world.max_goal_length + 1
            for max_steps in range(1, max_max_steps):
                solved_results = synthesizer.synthesize_plan(
                    self.current_state, max_steps=max_steps)
                if solved_results['solved']:
                    break

            if solved_results['solved']:
                self.ground_truth_plan = solved_results['plan']
                if use_ground_truth:
                    self.commander = Commander()
                    self.commander.set_plan(
                        solved_results['plan'],
                        solved_results['plan'],
                        solved_results['plan'])
            else:
                assert False, "Unable to synthesize a plan for this scenario"

    def use_ground_truth(self, synthesizer):
        self.commander = Commander()
        self.commander.set_plan(
            self.ground_truth_plan,
            self.ground_truth_plan,
            self.ground_truth_plan,)

    def set_reuse_key(self):
        self.reuse_key = True

    """
    Do planning using synthesizer, return:
    - plan_changed: true if new plan is different from the previous plan
    """

    def plan(
            self,
            synthesizer,
            generator,
            print_things=False,
            optimistic=False,
            n_completions=3,
            fail_with_simplest=False):
        assert not optimistic, "No optimistic planning for box world"
        if print_things:
            print("Replan...")

        success = False
        max_max_steps = self.world.max_goal_length + 1
        # Synthesize plan
        max_iter = 1
        current_iter = 0
        while (not success) and (current_iter < max_iter):
            if print_things:
                print("Iter {}".format(current_iter))
            current_iter += 1
            completed_states = []
            for _ in range(n_completions):
                completed_states.append(
                    hallucinate(
                        self.world,
                        generator,
                        self.current_state,
                        self.goal_length))

            best_result = None
            best_n_sat = -1
            for max_steps in range(1, max_max_steps):
                solved_results = synthesizer.plan_max_sat(
                    completed_states,
                    self.current_state,
                    self.world,
                    max_steps=max_steps)
                if print_things:
                    if solved_results is not None:
                        print("n_sat {}".format(solved_results['n_sat']))
                    else:
                        print("n_sat 0")

                if solved_results is not None:
                    if solved_results['n_sat'] > best_n_sat:
                        best_result = solved_results
                        best_n_sat = solved_results['n_sat']

                if best_n_sat >= n_completions:
                    break

            if best_result is not None:
                success = True

        plan_changed = False
        if best_result is not None:
            if self.commander is None:
                self.commander = Commander()
                self.commander.set_plan(
                    best_result['plan'],
                    best_result['plan'],
                    best_result['plan'])
            else:
                if best_result['plan'] != self.commander.plan[self.commander.current_stage:]:
                    self.commander.set_plan(
                        best_result['plan'],
                        best_result['plan'],
                        best_result['plan'])
                    plan_changed = True
            success = True

        if success:
            if print_things:
                print("Replan {}".format(plan_changed))

        else:
            if fail_with_simplest:
                # Use the singe step plan
                if self.commander is None:
                    self.commander = Commander()
                    simplest_plan = [self.world.goal_color_id]
                    self.commander.set_plan(
                        simplest_plan,
                        simplest_plan,
                        simplest_plan)
                    print(
                        "Unable to synthesize a plan for this scenario, with hallucination or optimistic. Use the simplest plan")
                success = True
            else:
                print(
                    "Unable to synthesize a plan for this scenario, with hallucination or optimistic. Use the previous plan")

        return plan_changed, completed_states, success

    def get_plan(self):
        return self.commander.export_remaining_plan()

    def reset(self, keep_plan=True):
        state = BoxState(
            self.world,
            self.init_grid,
            self.init_pos,
            self.init_key,
            self.init_mask)
        self.current_state = state
        if keep_plan:
            self.commander.reset_stage()
        else:
            self.commander = None

    def step(self, action):
        reuse_key = hasattr(self, "reuse_key") and self.reuse_key

        state = self.current_state
        n_key = state.key
        n_grid = state.grid

        change = CHANGE_COORDINATES[action]
        new_position = state.pos + change
        n_x, n_y = new_position

        reward = 0
        finished = False
        # Move player if the field in the moving direction is either
        if np.any(new_position < 0) or np.any(new_position >= self.world.n):
            # at boundary
            possible_move = False

        elif not state.grid[n_x, n_y, :].any():
            # No key, no lock
            possible_move = True

        elif n_y == 0 or (not state.grid[n_x, n_y - 1, :].any()):
            # first condition
            # is to catch keys at left boundary
            # It is a key
            if not state.grid[n_x, n_y + 1, :].any():
                # Key is not locked
                possible_move = True
                n_key = state.grid[n_x, n_y, :].argmax()
                if reuse_key:
                    gone_key = -1

                if n_key == self.world.goal_color_id:
                    # Goal reached
                    reward += self.world.reward_gem
                    finished = True

                elif n_key in self.dead_ends:
                    # reached a dead end, terminate episode
                    reward += self.world.reward_dead
                    finished = True
                    if reuse_key:
                        gone_key = n_key
                        n_key = state.key
                        finished = False

                elif n_key in self.correct_keys:
                    reward += self.world.reward_correct_key

                else:
                    reward += self.world.reward_wrong_key
                    if reuse_key:
                        gone_key = n_key
                        n_key = state.key

                # Update grid
                n_grid = state.grid.copy()
                n_grid[n_x, n_y, n_key] = 0
                if reuse_key and gone_key > -1:
                    n_grid[n_x, n_y, gone_key] = 0

            else:
                # Key is locked
                possible_move = False

        else:
            # It is a lock
            lock_type = state.grid[n_x, n_y, :].argmax()
            if lock_type == state.key:
                # The lock matches the key
                possible_move = True
                # Update grid
                n_grid = state.grid.copy()
                n_grid[n_x, n_y, lock_type] = 0
                n_key = -1
                if reuse_key:
                    n_key = state.key

            else:
                possible_move = False

        if not possible_move:
            n_x, n_y = state.pos

        n_mask = self._update_mask(n_x, n_y, state.mask)
        n_pos = np.array([n_x, n_y])
        new_state = BoxState(
            self.world,
            n_grid,
            n_pos,
            n_key,
            n_mask)
        self.current_state = new_state

        advanced = False
        if self.commander is not None:
            # Update command
            advanced, finished_com = self.commander.advance_command(new_state)

        return reward, new_state, finished, advanced

    def _update_mask(self, n_x, n_y, old_mask):
        new_mask = old_mask.copy()
        x_min, x_max, y_min, y_max = self._get_current_view(n_x, n_y)

        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                new_mask[i, j] = 1

        return new_mask

    def get_state(self):
        return self.current_state

    '''
    Get the current command
    '''

    def query_command(self):
        return self.commander.get_command()


class BoxState(object):
    '''
    mask: 1 means observable
    '''

    def __init__(self, world, grid, pos, key, mask):
        self.world = world
        self.grid = grid
        self.pos = pos
        self.key = key
        self.mask = mask
        self._cached_features = None

    '''
    condition is the key color required
    '''

    def satisfy(self, condition):
        return self.key == condition

    def features(self):
        if self._cached_features is None:
            x, y = self.pos
            bhw = self.world.n - 1

            # Mask grid
            prepared = self._apply_mask()

            # grid_feats: (self.n*2-1, self.n*2-1, n_things+1)
            grid_feats = array.pad_slice(
                prepared, (x - bhw, x + bhw + 1), (y - bhw, y + bhw + 1))

            # key feature. (n_things,)
            key_feature = np.zeros(self.world.n_things)
            if self.key != -1:
                key_feature[self.key] = 1

            features = {
                'grid_feats': grid_feats.astype(np.float32),
                'key_feats': key_feature.astype(np.float32),
            }

            self._cached_features = features

        return self._cached_features

    def features_world_model(self):
        # Add mask
        masked = self._apply_mask()

        # Add agent position
        pos_slice = np.zeros((self.world.n, self.world.n, 1))
        x, y = self.pos
        pos_slice[x, y, :] = 1
        pos_added = np.concatenate((masked, pos_slice), axis=2)
        map_feature = np.transpose(pos_added, (2, 0, 1))

        # key feature. (n_things,)
        key_feature = np.zeros(self.world.n_things)
        if self.key != -1:
            key_feature[self.key] = 1

        return map_feature, key_feature.astype(np.float32)

    '''
    Add mask to the one-hot grid representation
    '''

    def _apply_mask(self):
        expanded_mask = np.expand_dims(self.mask, 2)
        masked = self.grid * expanded_mask
        prepared = np.concatenate((masked, 1 - expanded_mask), axis=2)
        return prepared
