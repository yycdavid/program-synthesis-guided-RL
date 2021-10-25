from .cookbook import Cookbook
import numpy as np
import curses
import time
from synthesizer.solver import CraftSynthesizer
from skimage.measure import block_reduce
from misc import array
import copy
from collections import defaultdict
import torch
from torch.autograd import Variable

from stable_baselines3 import SAC
#from environment.ant import AntCraftEnv
import math
import pickle
import pygame
import pygame.freetype
from tqdm import tqdm
from random_hallucinator import RandomHallucinator

import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

N_TYPES = 4
CONNECTED = 0
WATER = 1
STONE = 2
NOT_ADJACENT = 3

WHITE = (255, 255, 255)
LIGHT = (196, 196, 196)
GREEN = (80, 160, 0)
DARK = (128, 128, 128)
DARK_RED = (139, 0, 0)
BLACK = (0, 0, 0)


def make_one_hot(label, C=10):
    one_hot = np.zeros(C)
    one_hot[label] = 1

    return one_hot


'''
Return a generated abstract state, given a generative model, and a partially observed state
'''


def hallucinate(world, generator, state, goal):
    map_width = world.MAP_WIDTH
    map_height = world.MAP_HEIGHT
    goal_start = world.grabbable_indices[0]
    n_goals = len(world.grabbable_indices)

    # Prepare conditional inputs
    def to_torch(t): return torch.tensor(t, dtype=torch.float32)
    abstract_partial_state = to_torch(world.get_abstract_partial_state(state))
    task = to_torch(make_one_hot(goal - goal_start, C=n_goals))
    unobs_fractions = to_torch(
        [1.0 - np.count_nonzero(state.mask) / (map_width * map_height)])
    cond_inp = torch.cat(
        (abstract_partial_state, task, unobs_fractions)).unsqueeze(0)

    # Run generative model
    z = Variable(torch.FloatTensor(1, 100).normal_())
    z = torch.cat((z, cond_inp), -1)
    recon = generator.decode(z)

    halluc_abs_state = world.parse_abstract_state(recon[0])
    return halluc_abs_state


class CraftWorldHard(object):
    def __init__(self, config, seed):
        self.MAP_WIDTH = 12
        self.MAP_HEIGHT = 12
        self.WINDOW_WIDTH = 5
        self.WINDOW_HEIGHT = 5
        self.N_WORKSHOPS = 3
        if hasattr(config.world, "view_range"):
            self.VIEW_RANGE = config.world.view_range
        else:
            self.VIEW_RANGE = 2

        if hasattr(config.world, "non_det") and config.world.non_det:
            self.non_det = True
        else:
            self.non_det = False

        self.DOWN = 0
        self.UP = 1
        self.LEFT = 2
        self.RIGHT = 3
        self.USE = 4
        self.n_actions = self.USE + 1

        self.PARTIAL_REWARD = 0.2
        self.REPLAN_REWARD = 0.2
        self.FINAL_REWARD = 1.0

        self.cookbook = Cookbook(config.recipes)
        self.n_features = 2 * self.WINDOW_WIDTH * self.WINDOW_HEIGHT * \
            (self.cookbook.n_kinds + 1) + self.cookbook.n_kinds + 4

        self.non_grabbable_indices = self.cookbook.environment
        # grabbable_indices includes both primitives and tools
        self.grabbable_indices = [
            i for i in range(
                self.cookbook.n_kinds) if (
                i not in self.non_grabbable_indices) and i != 0]
        self.primitive_indices = [i for i in self.cookbook.primitives]
        self.workshop_indices = [self.cookbook.index["workshop%d" % i]
                                 for i in range(self.N_WORKSHOPS)]
        self.water_index = self.cookbook.index["water"]
        self.stone_index = self.cookbook.index["stone"]
        self.usable_indices = [
            self.cookbook.index['bridge'],
            self.cookbook.index['axe'],
            self.cookbook.index['ladder']]
        self.artifact_indices = [
            i for i in self.grabbable_indices if i not in self.cookbook.primitives]
        self.action_indices = self.primitive_indices + \
            self.workshop_indices + self.usable_indices + [0]

        self.workshop_set = set(self.workshop_indices)
        self.usable_set = set(self.usable_indices)

        self.set_world_models_consts()

        self.random = np.random.RandomState(seed)

        # Abstract state size
        self.ABS_MAX_REGIONS = 2
        self.ABS_NUM_THINGS = 8
        self.ABS_MAX_VAL = 5
        self.ABS_SIZE = self.ABS_MAX_REGIONS + 1 + self.ABS_NUM_THINGS * self.ABS_MAX_VAL * self.ABS_MAX_REGIONS # 83
        self.ABS_PARTIAL_SIZE = self.ABS_SIZE * 2 + 1 + len(self.grabbable_indices)

        # Add task distribution
        task_count = {}
        for task in self.grabbable_indices:
            task_count[task] = 2
        task_count[self.cookbook.index['gold']] += 15
        task_count[self.cookbook.index['gem']] += 15

        total_count = 0
        for _, cnt in task_count.items():
            total_count += cnt

        self.task_probs = {}
        for task in self.grabbable_indices:
            self.task_probs[task] = task_count[task] / total_count
        # For gem and gold tasks
        self.path_types = ['direct', 'bridge', 'axe', 'ladder']
        path_counts = [2, 5, 5, 5]
        self.path_probs = [cnt / sum(path_counts) for cnt in path_counts]

        self.tools = {
            "bridge",
            "axe",
            "ladder",
        }

        # For ANT
        self.set_ant(config)

    def set_ant(self, config):
        self.use_ant = False
        if hasattr(config.world, "with_ant") and config.world.with_ant:
            self.use_ant = True
            self.ant_cell_size = 2.0
            self.ant_n_steps = config.world.ant_steps
            self.ant_deterministic = config.world.ant_deterministic
            # Load ANT controller
            self.ant_controller = SAC.load(config.world.ant_model)

            # Load init states
            with open(config.world.start_states, 'rb') as f:
                start_states = pickle.load(f)
            init_pool_size = 100
            self.ant_start_states = {}
            self.ant_start_states['final_poses'] = start_states['final_poses'][:init_pool_size]
            self.ant_start_states['final_vels'] = start_states['final_vels'][:init_pool_size]

    '''
    Convert from discrete position in craft map to position in ANT environment
    '''

    def convert_pos_craft_to_ant(self, craft_pos):
        x = craft_pos[0] * self.ant_cell_size + self.ant_cell_size / 2
        y = craft_pos[1] * self.ant_cell_size + self.ant_cell_size / 2
        ant_pos = np.array([x, y])

        return ant_pos

    '''
    Convert from position in ANT environment to discrete position in craft map.
    '''

    def convert_pos_ant_to_craft(self, ant_pos):
        clipped = np.clip(ant_pos, 0.0, self.ant_cell_size * self.MAP_WIDTH)
        discrete = [math.floor(p / self.ant_cell_size) for p in ant_pos]
        craft_pos = np.clip(discrete, 0, self.MAP_WIDTH - 1)

        return craft_pos

    def set_world_models_consts(self):
        # Dimensions for world model features
        # 1 for partial observable mask, 1 for agent pos
        self.map_feature_dim = self.cookbook.n_kinds + 2
        self.max_val_inventory = 5
        self.inventory_feature_dim = len(
            self.grabbable_indices) * self.max_val_inventory
        self.direction_feature_dim = 4
        self.task_feature_dim = len(self.grabbable_indices)
        self.flat_feature_dim = self.direction_feature_dim + \
            self.inventory_feature_dim + self.task_feature_dim

    '''
    goal is an index to the goal item.
    path_type: if not None, specify a path type from ['direct', 'bridge', 'axe', 'ladder']
    '''
    def sample_scenario_with_goal(self, goal, path_type=None):
        # If goal is gem or gold, sample from path_types, fix workshops to be only the required ones.
        assert goal not in self.cookbook.environment
        if goal == self.cookbook.index['gold'] or goal == self.cookbook.index['gem']:
            if path_type is None:
                path_type = self.random.choice(self.path_types, p=self.path_probs)

            if path_type == 'direct':
                # Half of the times, there's only one zone
                num_zones = self.random.choice([1,2])
                if num_zones == 2:
                    # Make boudary
                    region_1, region_2, line = self._random_boundary_line(self.random)
                    boundary_type = self.random.choice(['river', 'wall'])
                    region_1_resources, region_1_workshops = self.cookbook.primitives_all_for(goal)
                    region_2_resources = {}
                    region_2_workshops = set()

                    settings = {
                        "num_regions": 2,
                        "boundary_type": boundary_type,
                        "boundary_line": line,
                        "region_1": region_1,
                        "region_2": region_2,
                        "region_1_resources": region_1_resources,
                        "region_1_workshops": region_1_workshops,
                        "region_2_resources": region_2_resources,
                        "region_2_workshops": region_2_workshops,
                    }
                else:
                    settings = {
                        "num_regions": 1,
                        "resources": {goal: 1},
                        "workshops": set(),
                    }

            else:
                # Make boudary
                region_1, region_2, line = self._random_boundary_line(self.random)
                if path_type == 'bridge':
                    boundary_type = 'river'
                elif path_type == 'axe':
                    boundary_type = 'wall'
                else: # Ladder
                    boundary_type = self.random.choice(['river', 'wall'])
                region_1_resources, region_1_workshops = self.cookbook.primitives_all_for(self.cookbook.index[path_type])
                region_2_resources, region_2_workshops = self.cookbook.primitives_all_for(goal)

                settings = {
                    "num_regions": 2,
                    "boundary_type": boundary_type,
                    "boundary_line": line,
                    "region_1": region_1,
                    "region_2": region_2,
                    "region_1_resources": region_1_resources,
                    "region_1_workshops": region_1_workshops,
                    "region_2_resources": region_2_resources,
                    "region_2_workshops": region_2_workshops,
                }

            settings["random_workshop"] = False

        elif goal in self.cookbook.primitives:
            settings = {
                "num_regions": 1,
                "resources": {goal: 1},
                "workshops": set(),
                "random_workshop": True,
            }

        elif goal in self.cookbook.recipes:
            resources, workshops = self.cookbook.primitives_all_for(goal)
            settings = {
                "num_regions": 1,
                "resources": resources,
                "workshops": workshops,
                "random_workshop": True,
            }

        else:
            assert False, "don't know how to build a scenario for %s" % goal

        settings['goal'] = goal
        return self._sample_scenario(settings)

    def _sample_scenario(self, settings):
        # generate grid
        grid = np.zeros(
            (self.MAP_WIDTH,
             self.MAP_HEIGHT,
             self.cookbook.n_kinds))
        i_bd = self.cookbook.index["boundary"]
        grid[0, :, i_bd] = 1
        grid[self.MAP_WIDTH - 1:, :, i_bd] = 1
        grid[:, 0, i_bd] = 1
        grid[:, self.MAP_HEIGHT - 1:, i_bd] = 1

        if settings['num_regions'] == 2:
            # Put river/wall
            if settings['boundary_type'] == "river":
                boundary_index = self.water_index
            else:
                boundary_index = self.stone_index

            for (x, y) in settings['boundary_line']:
                grid[x, y, boundary_index] = 1

            # Put necessary resources
            self._put_resources(
                grid,
                settings['region_1'],
                settings['region_1_resources'])
            self._put_resources(
                grid,
                settings['region_2'],
                settings['region_2_resources'])

            # Put necessary workshops
            self._put_workshops(
                grid,
                settings['region_1'],
                settings['region_1_workshops'])
            self._put_workshops(
                grid,
                settings['region_2'],
                settings['region_2_workshops'])

        elif settings['num_regions'] == 1:
            self._put_resources(grid, None, settings['resources'])
            self._put_workshops(grid, None, settings['workshops'])

        else:
            assert False, "don't know how to build a scenario for %s regions" % settings[
                'num_regions']

        # Put some random resources and workshops
        # 3 random resources
        primitives_no_goal = [i for i in self.primitive_indices if i != settings['goal']]
        for i in range(3):
            item = self.random.choice(primitives_no_goal)
            pos = self._random_free(grid, self.random)
            grid[pos[0], pos[1], item] = 1

        if settings['random_workshop']:
            # 1 random workshop
            for i in range(1):
                bench = self.random.choice(self.workshop_indices)
                pos = self._random_free(grid, self.random)
                grid[pos[0], pos[1], bench] = 1

        # generate init pos
        if settings['num_regions'] == 2:
            init_pos = self._random_free(
                grid, self.random, settings['region_1'])
        elif settings['num_regions'] == 1:
            init_pos = self._random_free(grid, self.random)
        else:
            assert False, "don't know how to build a scenario for %s regions" % settings[
                'num_regions']

        if self.use_ant:
            ant_args = {
                'ant_controller': self.ant_controller,
                'ant_start_states': self.ant_start_states,
            }
            return CraftScenario(
                grid,
                init_pos,
                self,
                settings['goal'],
                use_ant=self.use_ant,
                ant_args=ant_args)

        else:
            return CraftScenario(grid, init_pos, self, settings['goal'])

    def visualize(self, state, goal, plan_in_name):
        def _visualize(win):
            curses.start_color()
            for i in range(1, 12):
                curses.init_pair(i, i, curses.COLOR_BLACK)
                curses.init_pair(i + 11, curses.COLOR_BLACK, i % 3 + 1)
            win.clear()
            for y in range(self.MAP_HEIGHT):
                for x in range(self.MAP_WIDTH):
                    if state.mask[x, y] == 0:
                        ch1 = '.'
                        ch2 = '.'
                        color = curses.color_pair(0)
                    else:
                        if not (state.grid[x, y, :].any()
                                or (x, y) == state.pos):
                            continue
                        thing = state.grid[x, y, :].argmax()
                        if (x, y) == state.pos:
                            if state.dir == self.LEFT:
                                ch1 = "<"
                                ch2 = "@"
                            elif state.dir == self.RIGHT:
                                ch1 = "@"
                                ch2 = ">"
                            elif state.dir == self.UP:
                                ch1 = "^"
                                ch2 = "@"
                            elif state.dir == self.DOWN:
                                ch1 = "@"
                                ch2 = "v"
                            color = 0
                        elif thing == self.cookbook.index["boundary"]:
                            ch1 = ch2 = curses.ACS_BOARD
                            color = curses.color_pair(11 + thing)
                        else:
                            name = self.cookbook.index.get(thing)
                            ch1 = name[0]
                            ch2 = name[-1]
                            color = curses.color_pair(11 + thing)

                    win.addch(self.MAP_HEIGHT - y, x * 2, ch1, color)
                    win.addch(self.MAP_HEIGHT - y, x * 2 + 1, ch2, color)
                win.refresh()

            # Add goal
            goal_y = int(1.5 * self.MAP_HEIGHT)
            cur_x = 0
            color = curses.color_pair(0)
            goal_name = self.cookbook.index.get(goal)
            goal_display = ["Goal:", goal_name]
            for word in goal_display:
                for ch in word:
                    win.addch(goal_y, cur_x, ch, color)
                    cur_x += 1
                win.addch(goal_y, cur_x, ' ', color)
                cur_x += 1

            # Add plan
            plan_y = 2 * self.MAP_HEIGHT
            cur_x = 0
            for action in plan_in_name:
                for ch in action:
                    win.addch(plan_y, cur_x, ch, color)
                    cur_x += 1
                win.addch(plan_y, cur_x, ',', color)
                cur_x += 1
                win.addch(plan_y, cur_x, ' ', color)
                cur_x += 1
            win.refresh()

            time.sleep(60.0)
        curses.wrapper(_visualize)

    def setup_pygame(self, img_folder, object_param_list):
        self.object_image = {}
        for (objname, imgname) in object_param_list.items():
            image = pygame.image.load(os.path.join(img_folder, imgname))
            self.object_image[objname] = image

    def visualize_pretty(
            self,
            state,
            save_dir,
            num=1,
            no_agent=False,
            with_side=True,
            plan=None):
        # Init
        pygame.init()
        # pygame.freetype.init()

        size = self.setup_map(with_side)

        # Add things in
        self.add_things_in(state)

        # Render plan and inventory
        self.add_plan_and_inv(state, plan, size)

        # Save the plot
        save_path = os.path.join(save_dir, "pic{}.png".format(num))
        pygame.image.save(self.screen, save_path)

    def action_to_primitive(self, action):
        resources = {
            "wood",
            "iron",
            "grass",
            "gold",
            "gem",
        }
        workshops = {
            "workshop0": "factory",
            "workshop1": "workbench",
            "workshop2": "toolshed",
        }
        if action in resources:
            return "get " + action
        elif action in self.tools:
            return "use " + action
        else:
            return "use " + workshops[action]

    def pretty_rollout(self, rollout, save_dir, plans=None):
        for j, state in tqdm(enumerate(rollout)):
            self.visualize_pretty(state, save_dir, num=j, plan=plans[j])

        os.system('ffmpeg -y -r 2 -f image2 -s 1920x1080 -i {}/pic%d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}/test.mp4'.format(save_dir, save_dir))

    def setup_map(self, with_side):
        # Set up map
        pygame.event.pump()
        self.render_scale = 48
        if with_side:
            self.plan_font_size = 22
            self.title_font_size = 24
            count_font_size = 30
            self.TITLE_FONT = pygame.freetype.SysFont(
                'Comic Sans MS', self.title_font_size)
            self.PLAN_FONT = pygame.freetype.SysFont('Arial', self.plan_font_size)
            self.PLAN_FONT_CURRENT = pygame.freetype.SysFont(
                'Arial', self.plan_font_size, bold=True)
            self.COUNT_FONT = pygame.freetype.SysFont('Arial', count_font_size)
            side_width = 180

            size = [
                self.render_scale *
                self.MAP_WIDTH +
                side_width,
                self.render_scale *
                self.MAP_HEIGHT]

        else:
            size = [
                self.render_scale *
                self.MAP_HEIGHT,
                self.render_scale *
                self.MAP_WIDTH]

        self.screen = pygame.display.set_mode(size)
        self.screen.fill(WHITE)

        # Grid lines
        self.tbias = 0
        for x in range(self.MAP_WIDTH + 1):
            pygame.draw.line(self.screen,
                             DARK,
                             [x * self.render_scale,
                              self.tbias],
                             [x * self.render_scale,
                              self.tbias + self.MAP_HEIGHT * self.render_scale],
                             3)
        for y in range(self.MAP_WIDTH + 1):
            pygame.draw.line(self.screen,
                             DARK,
                             [0,
                              self.tbias + y * self.render_scale],
                             [self.MAP_WIDTH * self.render_scale,
                              self.tbias + y * self.render_scale],
                             3)

        return size

    def add_things_in(self, state):
        img_size = [self.render_scale, self.render_scale]
        for y in range(self.MAP_HEIGHT):
            for x in range(self.MAP_WIDTH):
                if state.grid[x, y, :].any() or (x, y) == state.pos:
                    thing = state.grid[x, y, :].argmax()
                    if (x, y) == state.pos:
                        obj_img = self.object_image['agent']

                    elif thing == self.cookbook.index["boundary"]:
                        obj_img = self.object_image['boundary']

                    else:
                        name = self.cookbook.index.get(thing)
                        obj_img = self.object_image[name]

                    if obj_img.get_width() != self.render_scale:
                        obj_img = pygame.transform.scale(obj_img, img_size)
                    self.screen.blit(
                        obj_img, (x * self.render_scale, self.tbias + y * self.render_scale))

                if state.mask[x, y] == 0:
                    s = pygame.Surface(img_size)
                    s.set_alpha(128)
                    s.fill(BLACK)
                    self.screen.blit(
                        s, (x * self.render_scale, self.tbias + y * self.render_scale))

    def traj_vis(self, rollout, save_dir, plans, vis_step, with_side=True, single_plan=False):
        # Init
        pygame.init()
        size = self.setup_map(with_side)

        # Base is the map at vis_step
        state = rollout[vis_step]
        self.add_things_in(state)

        # Add the things in initial map as transparent objects
        img_size = [self.render_scale, self.render_scale]
        init_state = rollout[0]
        for y in range(self.MAP_HEIGHT):
            for x in range(self.MAP_WIDTH):
                if init_state.grid[x, y, :].any() and (not state.grid[x, y, :].any()):
                    thing = init_state.grid[x, y, :].argmax()
                    name = self.cookbook.index.get(thing)
                    obj_img = self.object_image[name]
                    if obj_img.get_width() != self.render_scale:
                        obj_img = pygame.transform.scale(obj_img, img_size)
                    converted = obj_img.convert_alpha()
                    alpha_surface = pygame.Surface(converted.get_size(), pygame.SRCALPHA)
                    alpha_surface.fill((255, 255, 255, 90))
                    converted.blit(alpha_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                    self.screen.blit(
                        converted, (x * self.render_scale, self.tbias + y * self.render_scale))

        # Add agent trajectory
        if not single_plan:
            # Start point
            x, y = init_state.pos
            obj_img = self.object_image['start']
            if obj_img.get_width() != self.render_scale:
                obj_img = pygame.transform.scale(obj_img, img_size)
            self.screen.blit(
                obj_img, (x * self.render_scale, self.tbias + y * self.render_scale))

        # Trajectory
        arrow_size = [self.render_scale//2, self.render_scale//2]
        for i in range(vis_step):
            x_prev, y_prev = rollout[i].pos
            x_next, y_next = rollout[i+1].pos
            arrow_img = self.object_image['green_arrow']
            arrow_img = pygame.transform.scale(arrow_img, arrow_size)
            self.blit_arrow(arrow_img, x_prev, y_prev, x_next, y_next)

        if not single_plan:
            for i in range(vis_step, len(rollout)-1):
                x_prev, y_prev = rollout[i].pos
                x_next, y_next = rollout[i+1].pos
                arrow_img = self.object_image['red_arrow']
                arrow_img = pygame.transform.scale(arrow_img, arrow_size)
                self.blit_arrow(arrow_img, x_prev, y_prev, x_next, y_next)

        # Add plan
        if single_plan:
            self.add_plan_and_inv(state, plans[vis_step], size)
        else:
            self.add_plan_two_and_inv(state, plans[vis_step], plans[vis_step-1], size)

        # Save with step in name
        save_path = os.path.join(save_dir, "pwa{}.png".format(vis_step))
        pygame.image.save(self.screen, save_path)

    '''
    Arrow image should be left facing
    '''
    def blit_arrow(self, arrow_img, x_prev, y_prev, x_next, y_next):
        if x_next - x_prev == 1:
            # Right
            self.screen.blit(arrow_img, (x_prev * self.render_scale + self.render_scale/2 + self.render_scale/4, self.tbias + y_prev * self.render_scale + self.render_scale/4))

        elif x_next - x_prev == -1:
            # Left
            rotated = pygame.transform.rotate(arrow_img, 180)
            self.screen.blit(rotated, (x_next * self.render_scale + self.render_scale/2 + self.render_scale/4, self.tbias + y_next * self.render_scale + self.render_scale/4))

        elif y_next - y_prev == 1:
            # Down
            rotated = pygame.transform.rotate(arrow_img, 270)
            self.screen.blit(rotated, (x_prev * self.render_scale + self.render_scale/4, self.tbias + y_prev * self.render_scale + self.render_scale/2 + self.render_scale/4))

        elif y_next - y_prev == -1:
            # Up
            rotated = pygame.transform.rotate(arrow_img, 90)
            self.screen.blit(rotated, (x_next * self.render_scale + self.render_scale/4, self.tbias + y_next * self.render_scale + self.render_scale/2 + self.render_scale/4))

        else:
            # Did not move
            pass


    def add_plan_and_inv(self, state, plan, size):
        margin = 20
        side_x = self.render_scale * self.MAP_WIDTH + margin
        plan_y = margin
        self.TITLE_FONT.render_to(self.screen, (side_x, plan_y), "Program:")
        plan_y += self.title_font_size + 5
        if plan is not None:
            plan_in_name = plan['plan_in_name']
            for (i, action) in enumerate(plan_in_name):
                primitive = self.action_to_primitive(action)
                if i == 0:
                    self.PLAN_FONT_CURRENT.render_to(
                        self.screen, (side_x, plan_y), primitive)
                else:
                    self.PLAN_FONT.render_to(
                        self.screen, (side_x, plan_y), primitive)
                plan_y += self.plan_font_size + 5

        # Add inventory
        img_size = [self.render_scale, self.render_scale]
        inv_y = size[1] / 2
        inv_spacing = 15
        img_size_inv = [int(img_size[0] / 2), int(img_size[1] / 2)]
        count_x = side_x + img_size_inv[0] + margin
        self.TITLE_FONT.render_to(self.screen, (side_x, inv_y), "Inventory:")
        inv_y += self.title_font_size + 5
        for thing in self.grabbable_indices:
            count = int(state.inventory[thing])
            if count <= 0:
                continue
            name = self.cookbook.index.get(thing)

            obj_img = self.object_image[name]

            if obj_img.get_width() != self.render_scale:
                obj_img = pygame.transform.scale(obj_img, img_size_inv)
            self.screen.blit(
                obj_img, (side_x, inv_y))
            self.COUNT_FONT.render_to(
                self.screen, (count_x, inv_y), "* {}".format(count))
            inv_y += img_size_inv[1] + inv_spacing

    def add_plan_two_and_inv(self, state, new_plan, old_plan, size):
        margin = 20
        side_x = self.render_scale * self.MAP_WIDTH + margin
        plan_y = margin
        self.TITLE_FONT.render_to(self.screen, (side_x, plan_y), "New program:")
        plan_y += self.title_font_size + 5
        if new_plan is not None:
            plan_in_name = new_plan['plan_in_name']
            for (i, action) in enumerate(plan_in_name):
                primitive = self.action_to_primitive(action)
                if i == 0:
                    self.PLAN_FONT_CURRENT.render_to(
                        self.screen, (side_x, plan_y), primitive)
                else:
                    self.PLAN_FONT.render_to(
                        self.screen, (side_x, plan_y), primitive)
                plan_y += self.plan_font_size + 5

        plan_y += margin
        self.TITLE_FONT.render_to(self.screen, (side_x, plan_y), "Old program:")
        plan_y += self.title_font_size + 5
        if old_plan is not None:
            plan_in_name = old_plan['plan_in_name']
            for (i, action) in enumerate(plan_in_name):
                primitive = self.action_to_primitive(action)
                if i == 0:
                    self.PLAN_FONT_CURRENT.render_to(
                        self.screen, (side_x, plan_y), primitive)
                else:
                    self.PLAN_FONT.render_to(
                        self.screen, (side_x, plan_y), primitive)
                plan_y += self.plan_font_size + 5

        # Add inventory
        img_size = [self.render_scale, self.render_scale]
        inv_y = size[1] * 2 / 3
        inv_spacing = 15
        img_size_inv = [int(img_size[0] / 2), int(img_size[1] / 2)]
        count_x = side_x + img_size_inv[0] + margin
        self.TITLE_FONT.render_to(self.screen, (side_x, inv_y), "Inventory:")
        inv_y += self.title_font_size + 5
        for thing in self.grabbable_indices:
            count = int(state.inventory[thing])
            if count <= 0:
                continue
            name = self.cookbook.index.get(thing)

            obj_img = self.object_image[name]

            if obj_img.get_width() != self.render_scale:
                obj_img = pygame.transform.scale(obj_img, img_size_inv)
            self.screen.blit(
                obj_img, (side_x, inv_y))
            self.COUNT_FONT.render_to(
                self.screen, (count_x, inv_y), "* {}".format(count))
            inv_y += img_size_inv[1] + inv_spacing

    def visualize_rollout(self, rollout, plans=None):
        def _visualize(win):
            curses.start_color()
            for i in range(1, 12):
                curses.init_pair(i, i, curses.COLOR_BLACK)
                curses.init_pair(i + 11, curses.COLOR_BLACK, i % 3 + 1)
            for j, state in enumerate(rollout):
                win.clear()
                for y in range(self.MAP_HEIGHT):
                    for x in range(self.MAP_WIDTH):
                        if state.mask[x, y] == 0:
                            ch1 = '.'
                            ch2 = '.'
                            color = curses.color_pair(0)
                        else:
                            if not (state.grid[x, y, :].any()
                                    or (x, y) == state.pos):
                                continue
                            thing = state.grid[x, y, :].argmax()
                            if (x, y) == state.pos:
                                if state.dir == self.LEFT:
                                    ch1 = "<"
                                    ch2 = "@"
                                elif state.dir == self.RIGHT:
                                    ch1 = "@"
                                    ch2 = ">"
                                elif state.dir == self.UP:
                                    ch1 = "^"
                                    ch2 = "@"
                                elif state.dir == self.DOWN:
                                    ch1 = "@"
                                    ch2 = "v"
                                color = 0
                            elif thing == self.cookbook.index["boundary"]:
                                ch1 = ch2 = curses.ACS_BOARD
                                color = curses.color_pair(11 + thing)
                            else:
                                name = self.cookbook.index.get(thing)
                                ch1 = name[0]
                                ch2 = name[-1]
                                color = curses.color_pair(11 + thing)

                        win.addch(self.MAP_HEIGHT - y, x * 2, ch1, color)
                        win.addch(self.MAP_HEIGHT - y, x * 2 + 1, ch2, color)

                # Add inventory
                color = curses.color_pair(0)
                plan_y = 2 * self.MAP_HEIGHT
                cur_x = 0
                for thing in self.grabbable_indices:
                    count = int(state.inventory[thing])
                    if count <= 0:
                        continue
                    name = self.cookbook.index.get(thing)
                    for ch in name:
                        win.addch(plan_y, cur_x, ch, color)
                        cur_x += 1
                    win.addch(plan_y, cur_x, ':', color)
                    cur_x += 2
                    count = int(state.inventory[thing])
                    count_str = str(count)
                    for ch in count_str:
                        win.addch(plan_y, cur_x, ch, color)
                        cur_x += 1
                    plan_y += 1
                    cur_x = 0

                if (plans is not None) and (not plans[j] is None):
                    # Display plan
                    plan_y += 5
                    cur_x = 0
                    plan_in_name = plans[j]['plan_in_name']
                    for action in plan_in_name:
                        for ch in action:
                            win.addch(plan_y, cur_x, ch, color)
                            cur_x += 1
                        win.addch(plan_y, cur_x, ',', color)
                        cur_x += 1
                        win.addch(plan_y, cur_x, ' ', color)
                        cur_x += 1

                win.refresh()
                time.sleep(0.5)

            time.sleep(10.0)
        curses.wrapper(_visualize)

    '''
    From a concrete partially observed state, get partial abstract state for conditional input to CVAE
    '''

    def get_abstract_partial_state(self, state):
        # pair of two abstract states -- 1. assuming the unobserved squares are
        # unpassable and 2. assuming the unobserved squares are passable.
        abstract_map1 = self.get_abstract_state(state, masked=True, blocked=True)
        abstract_map2 = self.get_abstract_state(state, masked=True, blocked=False)

        return np.concatenate((abstract_map1, abstract_map2))

    def _get_zones_and_boundaries_masked(self, grid, mask, seed_pos, blocked=True):
        boundary_lines = []
        river_line = self._get_boundary_line_of_type(grid, self.water_index, mask=mask)
        if len(river_line) > 0:
            boundary_lines.append({'type': WATER, 'line': river_line})
        wall_line = self._get_boundary_line_of_type(grid, self.stone_index, mask=mask)
        if len(wall_line) > 0:
            boundary_lines.append({'type': STONE, 'line': wall_line})

        assert len(boundary_lines) < 2, "Currently only support up to two zones"

        result_dict = {}
        if len(boundary_lines) == 0:
            region = set()
            for x in range(1, self.MAP_WIDTH-1):
                for y in range(1, self.MAP_HEIGHT-1):
                    if mask[x,y] == 1:
                        region.add((x,y))
            boundaries = {}
            boundaries[(0, 0)] = CONNECTED
            result_dict['n_zones'] = 1
            result_dict['regions'] = [region]
            result_dict['boundaries'] = boundaries

        elif len(boundary_lines) == 1:
            mask_region = set()
            for x in range(1, self.MAP_WIDTH-1):
                for y in range(1, self.MAP_HEIGHT-1):
                    if mask[x,y] == 0:
                        mask_region.add((x,y))
            region_1, region_2 = self.get_regions(
                boundary_lines[0]['line'], seed_pos=seed_pos, mask_region=mask_region, blocked=blocked)

            n_zones = 2
            boundaries = {}
            for i in range(n_zones):
                boundaries[(i, i)] = CONNECTED

            boundaries[(0, 1)] = boundary_lines[0]['type']
            result_dict['n_zones'] = n_zones
            result_dict['regions'] = [region_1, region_2]
            result_dict['boundaries'] = boundaries

        return result_dict

    '''
    If masked=False, get the abstract state for a full map;
    If masked=True, get the abstract state for a partially observed map, blocked=True means treating unobserved squares as blocked.

    For masked=True:
        If no water/stone in observed parts, 1 region
        If there is, then we divide into two regions, one reachable from self pos, the other one contains other squares. blocked=True means treating unobserved parts as blocking.
        Regions always only contains cells that are observed
    '''

    def get_abstract_state(self, state, masked=False, blocked=False):
        # format: max two regions
        # region's existence [r1, r2] # {0,1}^2
        # boundary between r1 and r2 [b] # {0,1} 0 - water, 1 - stone
        # things in each region: 8 thing x 5 possible values x 2 regions =
        # {0,1}^80
        if masked:
            regions = self._get_zones_and_boundaries_masked(state.grid, state.mask, state.pos, blocked=blocked)
        else:
            regions = self._get_zones_and_boundaries(
                state.grid, seed_pos=state.pos)
        n_zones = regions['n_zones']
        assert(n_zones <= self.ABS_MAX_REGIONS)

        region_exists_vec = np.zeros(2)
        region_exists_vec[:n_zones] = 1

        boundary_vec = np.zeros(1)
        if n_zones == 2:
            boundary = regions['boundaries'][(0, 1)]
            assert(boundary == 1 or boundary == 2)
            boundary_vec[0] = boundary - 1

        things_vec = np.zeros(self.ABS_NUM_THINGS * self.ABS_MAX_VAL * self.ABS_MAX_REGIONS)
        cur_idx = 0
        for i in range(n_zones):
            region = regions['regions'][i]
            counts = self._count_things(
                state.grid, region)
            for k in sorted(counts.keys()):
                v = counts[k]
                if v >= self.ABS_MAX_VAL:
                    print(v)
                assert(v < self.ABS_MAX_VAL)
                things_vec[cur_idx:cur_idx + v + 1] = 1
                cur_idx += self.ABS_MAX_VAL

        return np.concatenate((region_exists_vec, boundary_vec, things_vec))

    '''
    Parse the output from C-VAE to completed abstract state
    '''

    def parse_abstract_state(self, state):
        region_exists_vec = state[:self.ABS_MAX_REGIONS]
        boundary_vec = state[self.ABS_MAX_REGIONS]
        things_vec = state[self.ABS_MAX_REGIONS + 1:].reshape(-1, self.ABS_MAX_VAL)

        abstract_state = {}

        # Number of zones
        n_zones = region_exists_vec.sum()
        if n_zones >= 1.5:
            n_zones = 2
        else:
            n_zones = 1
        abstract_state['n_zones'] = n_zones

        # Zone boundaries
        boundaries = {}
        for i in range(n_zones):
            boundaries[(i, i)] = CONNECTED
        if n_zones == 2:
            boundaries[(0, 1)] = WATER if boundary_vec < 0.5 else STONE
        abstract_state['boundaries'] = boundaries

        # Resource and workshops
        counts = {}
        things_set = set()
        for thing in self.cookbook.primitives:
            things_set.add(thing)
        for thing in self.workshop_set:
            things_set.add(thing)
        ctr = 0
        for k in range(n_zones):
            counts[k] = {}
            for thing in (sorted(things_set)):
                count = things_vec[ctr].sum() - 1
                counts[k][thing] = round(count.item())
                ctr += 1
        abstract_state['counts'] = counts

        return abstract_state

    def compare_abs_states(self, true_abs, completed):
        n_same = 0
        n_total = 0
        # n_zones
        n_total += 1
        if true_abs['n_zones'] == completed['n_zones']:
            n_same += 1
        # boundary
        if true_abs['n_zones'] == 2 and completed['n_zones'] == 2:
            n_total += 1
            n_same += (true_abs["boundaries"][(0,1)] == completed["boundaries"][(0,1)])
            # Counts
            for k in range(2):
                for thing in true_abs['counts'][k]:
                    n_total += 1
                    n_same += (true_abs['counts'][k][thing] == completed['counts'][k][thing])
        else:
            for thing in true_abs['counts'][0]:
                n_total += 1
                n_same += (true_abs['counts'][0][thing] == completed['counts'][0][thing])

        return n_same, n_total

    '''
    Return a dictionary of primitives + workshops counts in region:
    - item: count
    '''

    def _count_things(self, grid, region, obs_mask=None, count_unobs=False):
        counts = {}
        for thing in self.cookbook.primitives:
            counts[thing] = 0
        for thing in self.workshop_set:
            counts[thing] = 0

        if count_unobs:
            counts['unobserved'] = 0

        if region is None:
            return counts

        for (x, y) in region:
            if (obs_mask is not None) and (obs_mask[x, y] == 0):
                if count_unobs:
                    counts['unobserved'] += 1
                continue
            if not grid[x, y, :].any():
                continue
            thing = grid[x, y, :].argmax()
            if (thing in self.cookbook.primitives) or (
                    thing in self.workshop_set):
                counts[thing] += 1

        return counts

    def _get_boundary_line_of_type(self, grid, index, mask=None):
        map_width, map_height, _ = grid.shape
        line = set()
        for x in range(1, map_width - 1):
            for y in range(1, map_height - 1):
                if (not mask is None) and mask[x, y] == 0:
                    continue
                if grid[x, y, index] == 1:
                    line.add((x, y))

        return line

    '''
    Returns a dict with:
        'n_zones': number of zones
        'regions': [set(pos)]
        'boundaries': Dict((i,j): boundary_type)

    Currently supports up to two zones, extend to more later
    '''

    def _get_zones_and_boundaries(self, grid, seed_pos=None):
        boundary_lines = []
        river_line = self._get_boundary_line_of_type(grid, self.water_index)
        if len(river_line) > 0:
            boundary_lines.append({'type': WATER, 'line': river_line})
        wall_line = self._get_boundary_line_of_type(grid, self.stone_index)
        if len(wall_line) > 0:
            boundary_lines.append({'type': STONE, 'line': wall_line})

        assert len(boundary_lines) < 2, "Currently only support up to two zones"

        result_dict = {}
        if len(boundary_lines) == 0:
            region = set([(x, y) for x in range(1, self.MAP_WIDTH - 1)
                          for y in range(1, self.MAP_HEIGHT - 1)])
            boundaries = {}
            boundaries[(0, 0)] = CONNECTED
            result_dict['n_zones'] = 1
            result_dict['regions'] = [region]
            result_dict['boundaries'] = boundaries

        elif len(boundary_lines) == 1:
            region_1, region_2 = self.get_regions(
                boundary_lines[0]['line'], seed_pos=seed_pos)
            if region_2 is None:
                n_zones = 1
                boundaries = {}
                for i in range(n_zones):
                    boundaries[(i, i)] = CONNECTED
                result_dict['n_zones'] = 1
                result_dict['regions'] = [region_1]
                result_dict['boundaries'] = boundaries

            else:
                n_zones = 2
                boundaries = {}
                for i in range(n_zones):
                    boundaries[(i, i)] = CONNECTED

                boundaries[(0, 1)] = boundary_lines[0]['type']
                result_dict['n_zones'] = n_zones
                result_dict['regions'] = [region_1, region_2]
                result_dict['boundaries'] = boundaries

        return result_dict

    '''
    Returns: (region_1, region_2, line) where
        - region_1, region_2: set of positions in grid in each region. region_1 >= region_2
        - line: set of positions on the boundary line
    '''

    def _random_boundary_line(self, random):
        # Sample ending points
        end_sides = [self.LEFT, self.RIGHT, self.UP, self.DOWN]
        index_chosen = sorted(random.choice(
            [i for i in range(4)], size=2, replace=False))
        sides_chosen = [end_sides[i] for i in index_chosen]
        end_locations = [i for i in range(3, self.MAP_WIDTH - 3)]
        end_loc_1 = random.choice(end_locations)
        end_loc_2 = random.choice(end_locations)

        # Get boundary line
        line = self._get_boundary_line(sides_chosen, end_loc_1, end_loc_2)

        # Get regions separated by the line
        region_1, region_2 = self.get_regions(line)
        if len(region_1) > len(region_2):
            return region_1, region_2, line
        else:
            return region_2, region_1, line

    '''
    Returns:
        - line: set of positions on the boundary line
    '''

    def _get_boundary_line(
            self,
            sides_chosen,
            end_loc_1,
            end_loc_2,
            corner=False):
        assert corner == False
        line = defaultdict(lambda: 0)
        if sides_chosen[0] == self.LEFT:
            if sides_chosen[1] == self.RIGHT:
                mid_point = self.MAP_WIDTH // 2
                for x in range(1, mid_point):
                    line[(x, end_loc_1)] += 1
                for x in range(mid_point + 1, self.MAP_WIDTH - 1):
                    line[(x, end_loc_2)] += 1
                if (min(end_loc_1, end_loc_2) +
                        1) > (max(end_loc_1, end_loc_2) - 1):
                    line[(mid_point, end_loc_1)] += 1
                else:
                    for y in range(min(end_loc_1, end_loc_2) + 1,
                                   max(end_loc_1, end_loc_2)):
                        line[(mid_point, y)] += 1

            elif sides_chosen[1] == self.UP:
                for x in range(1, end_loc_2):
                    line[(x, end_loc_1)] += 1
                for y in range(end_loc_1 + 1, self.MAP_HEIGHT - 1):
                    line[(end_loc_2, y)] += 1

            elif sides_chosen[1] == self.DOWN:
                for x in range(1, end_loc_2):
                    line[(x, end_loc_1)] += 1
                for y in range(1, end_loc_1):
                    line[(end_loc_2, y)] += 1

            else:
                assert False, "Error in side sampling"

        elif sides_chosen[0] == self.RIGHT:
            if sides_chosen[1] == self.UP:
                for x in range(end_loc_2 + 1, self.MAP_WIDTH - 1):
                    line[(x, end_loc_1)] += 1
                for y in range(end_loc_1 + 1, self.MAP_HEIGHT - 1):
                    line[(end_loc_2, y)] += 1

            elif sides_chosen[1] == self.DOWN:
                for x in range(end_loc_2 + 1, self.MAP_WIDTH - 1):
                    line[(x, end_loc_1)] += 1
                for y in range(1, end_loc_1):
                    line[(end_loc_2, y)] += 1

            else:
                assert False, "Error in side sampling"

        elif sides_chosen[0] == self.UP:
            if sides_chosen[1] == self.DOWN:
                mid_point = self.MAP_HEIGHT // 2
                for y in range(mid_point + 1, self.MAP_HEIGHT - 1):
                    line[(end_loc_1, y)] += 1
                for y in range(1, mid_point):
                    line[(end_loc_2, y)] += 1
                if (min(end_loc_1, end_loc_2) +
                        1) > (max(end_loc_1, end_loc_2) - 1):
                    line[(end_loc_1, mid_point)] += 1
                else:
                    for x in range(min(end_loc_1, end_loc_2) + 1,
                                   max(end_loc_1, end_loc_2)):
                        line[(x, mid_point)] += 1

            else:
                assert False, "Error in side sampling"

        else:
            assert False, "Error in side sampling"

        boundary_line = set()
        for pos, count in line.items():
            if count >= 1:
                boundary_line.add(pos)

        return boundary_line

    '''
    Get the grid indices in the two regions separated by the boundary_line
    Result regions won't contain cells in mask_region
    if blocked=True, mask_region are treated as blocked
    '''

    def get_regions(self, boundary_line, seed_pos=None, mask_region=None, blocked=False):
        covered = set()
        for pos in boundary_line:
            covered.add(pos)
        if blocked and mask_region is not None:
            for pos in mask_region:
                covered.add(pos)

        # Start from one point, grow the first region
        if seed_pos is None:
            seed_pos = (1, 1)
        region_1 = self._grow_region(covered, seed_pos, mask_region=mask_region)

        # Grow the second region
        region_2 = set()
        for x in range(1, self.MAP_WIDTH - 1):
            for y in range(1, self.MAP_HEIGHT - 1):
                if not (x, y) in covered:
                    region_2.add((x,y))
        if len(region_2) == 0:
            region_2 = None

        return region_1, region_2

    def _random_free(self, grid, random, domain=None):
        pos = None
        if domain:
            domain_list = [pos for pos in domain]
        while pos is None:
            if domain:
                idx = random.randint(len(domain_list))
                (x, y) = domain_list[idx]
            else:
                (x, y) = (random.randint(self.MAP_WIDTH),
                          random.randint(self.MAP_HEIGHT))
            if grid[x, y, :].any():
                continue
            pos = (x, y)
        return pos

    def neighbors(self, pos, dir=None):
        x, y = pos
        neighbors = []
        if x > 1 and (dir is None or dir == self.LEFT):
            neighbors.append((x - 1, y))
        if y > 1 and (dir is None or dir == self.DOWN):
            neighbors.append((x, y - 1))
        if x < self.MAP_WIDTH - 2 and (dir is None or dir == self.RIGHT):
            neighbors.append((x + 1, y))
        if y < self.MAP_HEIGHT - 2 and (dir is None or dir == self.UP):
            neighbors.append((x, y + 1))
        return neighbors

    def _get_fresh_pos(self, covered):
        for x in range(1, self.MAP_WIDTH - 1):
            for y in range(1, self.MAP_HEIGHT - 1):
                if not (x, y) in covered:
                    return (x, y)

    def _grow_region(self, covered, seed_pos, mask_region=None):
        region_1 = set()
        frontier = []
        frontier.append(seed_pos)
        region_1.add(seed_pos)
        covered.add(seed_pos)
        while len(frontier) > 0:
            popped = frontier.pop(0)
            neighbor_pos = self.neighbors(popped)
            for pos in neighbor_pos:
                if pos not in covered:
                    covered.add(pos)
                    frontier.append(pos)
                    if mask_region and (pos in mask_region):
                        continue
                    region_1.add(pos)

        return region_1

    '''
    Sample resources as specified in resource_dict into the grid, region is the set
    of indices to put in
    '''

    def _put_resources(self, grid, region, resource_dict):
        for ingredient, count in resource_dict.items():
            for _ in range(count):
                pos = self._random_free(grid, self.random, region)
                grid[pos[0], pos[1], ingredient] = 1

    def _put_workshops(self, grid, region, workshop_set):
        for workshop in workshop_set:
            pos = self._random_free(grid, self.random, region)
            grid[pos[0], pos[1], self.cookbook.index[workshop]] = 1

    '''
    Return:
    - Dict{
        (x, y): (x_min, x_max, y_min, y_max)
    }
    '''

    def _get_ant_blocks(self, grid):
        blocks = {}
        for x in range(self.MAP_WIDTH):
            for y in range(self.MAP_HEIGHT):
                if grid[x, y, :].any():
                    ant_pos = self.convert_pos_craft_to_ant([x, y])
                    x_min = ant_pos[0] - self.ant_cell_size / 2
                    x_max = ant_pos[0] + self.ant_cell_size / 2
                    y_min = ant_pos[1] - self.ant_cell_size / 2
                    y_max = ant_pos[1] + self.ant_cell_size / 2
                    blocks[(x, y)] = (x_min, x_max, y_min, y_max)
        return blocks


class CraftScenario(object):
    def __init__(
            self,
            grid,
            init_pos,
            world,
            goal,
            init_dir=0,
            inventory=None,
            use_ant=False,
            ant_args=None):
        self.init_grid = grid
        self.init_pos = init_pos
        self.init_dir = init_dir
        self.world = world
        self.goal = goal
        if inventory is not None:
            self.init_inventory = inventory
        else:
            self.init_inventory = np.zeros(self.world.cookbook.n_kinds)

        # Mask for observed part, 1 means observed
        self.init_mask = self._initial_mask(init_pos)

        # For ANT
        self.use_ant = use_ant
        if use_ant:
            self.ant_controller = ant_args['ant_controller']
            ant_blocks = self.world._get_ant_blocks(self.init_grid)

            # Create ANT env
            pos_ant = self.world.convert_pos_craft_to_ant(self.init_pos)
            self.ant_env = AntCraftEnv(
                ant_args['ant_start_states'], ant_blocks, pos_ant)

    def _initial_mask(self, pos):
        mask = np.zeros((self.world.MAP_WIDTH, self.world.MAP_HEIGHT))
        x, y = pos
        x_min, x_max, y_min, y_max = self._get_current_view(x, y)

        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                mask[i, j] = 1

        return mask

    def _get_current_view(self, x, y):
        x_min = max(x - self.world.VIEW_RANGE, 0)
        x_max = min(x + self.world.VIEW_RANGE, self.world.MAP_WIDTH - 1)
        y_min = max(y - self.world.VIEW_RANGE, 0)
        y_max = min(y + self.world.VIEW_RANGE, self.world.MAP_HEIGHT - 1)
        return x_min, x_max, y_min, y_max

    '''
    Create initial state.

    - id: Managed by Trainer, only used for pre-sampled scenarios, unique id to identify this scenario.
    - For baseline that doesn't use synthesizer (world model), give synthesizer=None.
    '''

    def init(self, synthesizer, id=0, use_ground_truth=True):
        self.id = id
        state = CraftState(
            self.world,
            self.init_grid,
            self.init_pos,
            self.init_dir,
            self.init_inventory,
            self.init_mask)
        self.current_state = state
        self.commander = None

        self.in_final = False

        if self.use_ant:
            pos_ant = self.world.convert_pos_craft_to_ant(self.init_pos)
            self.ant_env.reset_rollout(pos_ant)

        if synthesizer is not None:
            # Synthesize ground truth plan
            max_max_steps = 10
            for max_steps in range(1, max_max_steps):
                solved_results = synthesizer.synthesize_plan(
                    self.current_state, self.goal, max_steps=max_steps)
                if solved_results['solved']:
                    break

            if solved_results['solved']:
                self.ground_truth_plan = solved_results['plan']
                if use_ground_truth:
                    self.commander = Commander()
                    self.commander.set_plan(
                        solved_results['plan'],
                        solved_results['plan_in_name'],
                        solved_results['transition_conditions'])
            else:
                assert False, "Unable to synthesize a plan for this scenario"

    def use_ground_truth(self, synthesizer):
        # Synthesize ground truth plan
        max_max_steps = 10
        for max_steps in range(1, max_max_steps):
            solved_results = synthesizer.synthesize_plan(
                self.current_state, self.goal, max_steps=max_steps)
            if solved_results['solved']:
                break

        if solved_results['solved']:
            self.ground_truth_plan = solved_results['plan']
            self.commander = Commander()
            self.commander.set_plan(
                solved_results['plan'],
                solved_results['plan_in_name'],
                solved_results['transition_conditions'])
        else:
            assert False, "Unable to synthesize a plan for this scenario"

    def check_random_explore(self, env):
        '''
        For the random explore then plan ablations.
        Check if the first zone are all observed, if so, we finish the random explore stage.
        Return True if random explore stage has finished
        '''
        if not hasattr(self, "regions"):
            self.regions = env._get_zones_and_boundaries(
                self.current_state.grid, seed_pos=self.current_state.pos)
            self.finish_expl = False
        if self.finish_expl:
            return True
        # Count unobserved in region 1
        has_unobs = False
        for (x, y) in self.regions['regions'][0]:
            if (self.current_state.mask[x, y] == 0):
                has_unobs = True
                break
        if not has_unobs:
            self.finish_expl = True
        return self.finish_expl

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
            n_completions=3):
        if print_things:
            print("Replan...")

        if True and self.in_final:
            return False, None, True

        success = False
        max_max_steps = 7
        if not optimistic:
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
                            self.goal))

                best_result = None
                best_n_sat = -1
                for max_steps in range(1, max_max_steps):
                    solved_results = synthesizer.plan_max_sat(
                        completed_states,
                        self.current_state,
                        self.world,
                        self.goal,
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

        else:
            # Optimistic synthesize
            for max_steps in range(1, max_max_steps):
                solved_results = synthesizer.synthesize_plan(
                    self.current_state, self.goal, max_steps=max_steps, optimistic=True)
                if solved_results['solved']:
                    break

            completed_states = None
            best_result = None
            if solved_results['solved']:
                best_result = solved_results

        plan_changed = False
        if best_result is not None:
            if self.commander is None:
                self.commander = Commander()
                self.commander.set_plan(
                    best_result['plan'],
                    best_result['plan_in_name'],
                    best_result['transition_conditions'])
            else:
                if best_result['plan'] != self.commander.plan[self.commander.current_stage:]:
                    self.commander.set_plan(
                        best_result['plan'],
                        best_result['plan_in_name'],
                        best_result['transition_conditions'])
                    plan_changed = True
            success = True

        if success:
            if print_things:
                print("Replan {}".format(plan_changed))

        else:
            if isinstance(generator, RandomHallucinator):
                if self.commander is None:
                    self.commander = Commander()
                    random_plan = [self.world.random.choice(self.world.action_indices)]
                    plan_in_name = [self.world.cookbook.index.get(action) for action in random_plan]
                    # Condition that will never be reached
                    conditions = [{
                        self.world.cookbook.index['axe']: 5,
                    }]
                    self.commander.set_plan(
                        random_plan,
                        plan_in_name,
                        conditions
                    )
                    print(
                        "Unable to synthesize a plan for this scenario, with hallucination or optimistic. Use a random plan")
                success = True
            else:
                print(
                    "Unable to synthesize a plan for this scenario, with hallucination or optimistic. Use the previous plan")

        return plan_changed, completed_states, success

    '''
    Export a copy of the current state and commander state
    '''

    def export_state(self):
        state_info = {}
        state_info['grid'] = self.current_state.grid.copy()
        state_info['pos'] = self.current_state.pos
        state_info['dir'] = self.current_state.dir
        state_info['inventory'] = self.current_state.inventory.copy()
        state_info['mask'] = self.current_state.mask.copy()

        plan_info = self.commander.export_remaining_plan()

        return state_info, plan_info

    def get_plan(self):
        return self.commander.export_remaining_plan()

    '''
    Alternative to init(). This function inits with a given plan
    '''

    def reset(self, keep_plan=True):
        state = CraftState(
            self.world,
            self.init_grid,
            self.init_pos,
            self.init_dir,
            self.init_inventory,
            self.init_mask)
        self.current_state = state
        if keep_plan:
            self.commander.reset_stage()
        else:
            self.commander = None

        self.in_final = False

        # For backward compatibility
        if not hasattr(self, "use_ant"):
            self.use_ant = False

        if self.use_ant:
            pos_ant = self.world.convert_pos_craft_to_ant(self.init_pos)
            self.ant_env.reset_rollout(pos_ant)

        if hasattr(self, "finish_expl"):
            self.finish_expl = False

    '''
    Get the current state
    '''

    def get_state(self):
        return self.current_state

    '''
    Take a step, update both the current state and the commader state
    '''

    def act_ant(self, action):
        # Compute goal pos
        state = self.current_state
        x, y = state.pos
        if action == self.world.DOWN:
            dx, dy = (0, -1)
            n_dir = self.world.DOWN
        elif action == self.world.UP:
            dx, dy = (0, 1)
            n_dir = self.world.UP
        elif action == self.world.LEFT:
            dx, dy = (-1, 0)
            n_dir = self.world.LEFT
        elif action == self.world.RIGHT:
            dx, dy = (1, 0)
            n_dir = self.world.RIGHT
        else:
            raise Exception("Unexpected action: %s" % action)

        # We won't let ANT go if there is something in that direction
        go = True
        n_x = x + dx
        n_y = y + dy
        if state.grid[n_x, n_y, :].any():
            n_x, n_y = x, y
            go = False

        if go:
            target_pos = self.world.convert_pos_craft_to_ant((n_x, n_y))
            current_ant_pos = self.ant_env.get_pos()
            relative_pos = target_pos - current_ant_pos

            # The goal for ANT is at most self.world.ant_cell_size far away
            # from current position
            if np.linalg.norm(relative_pos) <= self.world.ant_cell_size:
                self.ant_env.set_goal(target_pos)

            else:
                direction = relative_pos / np.linalg.norm(relative_pos)
                goal_pos = direction * self.world.ant_cell_size + current_ant_pos
                self.ant_env.set_goal(goal_pos)

            # Act ANT using controller
            obs = self.ant_env.get_obs()
            for _ in range(self.world.ant_n_steps):
                # Output from ant_controller.predict() is of format (action,
                # None)
                a = self.ant_controller.predict(
                    obs, deterministic=self.world.ant_deterministic)[0]
                obs, _, done, _ = self.ant_env.step(a)
                if done:
                    self.ant_env.reset_inplace()

            final_pos = self.ant_env.get_pos()
            actual_pos = self.world.convert_pos_ant_to_craft(final_pos)
            dx_actual = actual_pos[0] - x
            dy_actual = actual_pos[1] - y

        else:
            dx_actual, dy_actual = (0, 0)

        return dx_actual, dy_actual, n_dir

    def step(self, action):
        # Update current state
        state = self.current_state
        x, y = state.pos
        n_dir = state.dir
        n_inventory = state.inventory
        n_grid = state.grid

        reward = 0

        non_det = (hasattr(self.world, "non_det") and self.world.non_det)
        fail_prob = 0.2
        if non_det:
            failed = (self.world.random.uniform() < fail_prob)

        # use actions
        if action == self.world.USE:
            cookbook = self.world.cookbook
            dx, dy = (0, 0)
            success = False
            if not (non_det and failed):
                for nx, ny in self.world.neighbors(state.pos, state.dir):
                    here = state.grid[nx, ny, :]
                    if not state.grid[nx, ny, :].any():
                        continue

                    if here.sum() > 1:
                        print("impossible world configuration:")
                        print(here.sum())
                        print(state.grid.sum(axis=2))
                        print(state.grid.sum(axis=0).sum(axis=0))
                        print(cookbook.index.contents)
                    assert here.sum() == 1
                    thing = here.argmax()

                    if not(thing in self.world.grabbable_indices or
                            thing in self.world.workshop_indices or
                            thing == self.world.water_index or
                            thing == self.world.stone_index):
                        continue

                    n_inventory = state.inventory.copy()
                    n_grid = state.grid.copy()

                    if thing in self.world.grabbable_indices:
                        n_inventory[thing] += 1
                        n_grid[nx, ny, thing] = 0
                        if self.use_ant:
                            self.ant_env.remove_block((nx, ny))
                        success = True

                    elif thing in self.world.workshop_indices:
                        workshop = cookbook.index.get(thing)
                        made_something = True
                        while made_something:
                            made_something = False
                            for output, inputs in cookbook.recipes.items():
                                if inputs["_at"] != workshop:
                                    continue
                                yld = inputs["_yield"] if "_yield" in inputs else 1
                                ing = [i for i in inputs if isinstance(i, int)]
                                if any(n_inventory[i] < inputs[i] for i in ing):
                                    continue
                                n_inventory[output] += yld
                                for i in ing:
                                    n_inventory[i] -= inputs[i]
                                success = True
                                made_something = True

                    elif thing == self.world.water_index:
                        if n_inventory[cookbook.index["bridge"]] > 0:
                            n_grid[nx, ny, self.world.water_index] = 0
                            if self.use_ant:
                                self.ant_env.remove_block((nx, ny))
                            n_inventory[cookbook.index["bridge"]] -= 1
                            self.in_final = True
                        elif n_inventory[cookbook.index["ladder"]] > 0:
                            n_grid[nx, ny, self.world.water_index] = 0
                            if self.use_ant:
                                self.ant_env.remove_block((nx, ny))
                            n_inventory[cookbook.index["ladder"]] -= 1
                            self.in_final = True

                    elif thing == self.world.stone_index:
                        if n_inventory[cookbook.index["axe"]] > 0:
                            n_grid[nx, ny, self.world.stone_index] = 0
                            if self.use_ant:
                                self.ant_env.remove_block((nx, ny))
                            n_inventory[cookbook.index["axe"]] -= 1
                            self.in_final = True
                        elif n_inventory[cookbook.index["ladder"]] > 0:
                            n_grid[nx, ny, self.world.stone_index] = 0
                            if self.use_ant:
                                self.ant_env.remove_block((nx, ny))
                            n_inventory[cookbook.index["ladder"]] -= 1
                            self.in_final = True

                    break

        else:
            if self.use_ant:
                dx, dy, n_dir = self.act_ant(action)
            else:
                if non_det and failed:
                    real_action = self.world.random.randint(self.world.RIGHT+1)
                else:
                    real_action = action
                # move actions
                if real_action == self.world.DOWN:
                    dx, dy = (0, -1)
                    n_dir = self.world.DOWN
                elif real_action == self.world.UP:
                    dx, dy = (0, 1)
                    n_dir = self.world.UP
                elif real_action == self.world.LEFT:
                    dx, dy = (-1, 0)
                    n_dir = self.world.LEFT
                elif real_action == self.world.RIGHT:
                    dx, dy = (1, 0)
                    n_dir = self.world.RIGHT
                # other
                else:
                    raise Exception("Unexpected action: %s" % action)

        n_x = x + dx
        n_y = y + dy
        if state.grid[n_x, n_y, :].any():
            assert not self.use_ant, "Ant should not reach here"
            n_x, n_y = x, y

        n_mask = self._update_mask(n_x, n_y, state.mask)

        new_state = CraftState(
            self.world, n_grid, (n_x, n_y), n_dir, n_inventory, n_mask)
        self.current_state = new_state

        advanced, finished = False, False
        if self.commander is not None:
            # Update command
            advanced, finished = self.commander.advance_command(new_state)
            if advanced:
                reward += self.world.PARTIAL_REWARD

        goal_condition = {}
        goal_condition[self.goal] = 1
        if new_state.satisfy(goal_condition):
            finished = True
            if self.commander is not None:
                self.commander.finish()

        return reward, new_state, finished, advanced

    def _update_mask(self, n_x, n_y, old_mask):
        new_mask = old_mask.copy()
        x_min, x_max, y_min, y_max = self._get_current_view(n_x, n_y)

        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                new_mask[i, j] = 1

        return new_mask

    '''
    Get the current command
    '''

    def query_command(self):
        return self.commander.get_command()


class CraftState(object):
    '''
    mask: 1 means observable
    '''

    def __init__(self, world, grid, pos, dir, inventory, mask):
        self.world = world
        self.grid = grid
        self.pos = pos
        self.dir = dir
        self.inventory = inventory
        self.mask = mask
        self._cached_features = None

    def satisfy(self, condition):
        satisfies = True
        for item in condition:
            if item in self.world.usable_set:
                if self.inventory[item] != condition[item]:
                    satisfies = False
            else:
                if self.inventory[item] < condition[item]:
                    satisfies = False
        return satisfies

    def export_for_save(self):
        export_state = CraftState(None, self.grid, self.pos, self.dir, self.inventory, self.mask)
        return export_state

    '''
    Feature contains:
    - grid_feats: the grid around current position, within a window
    - grid_feats_big_red: aggregated grid info per (WINDOW_WIDTH, WINDOW_HEIGHT) region, WINDOW_WIDTH*WINDOW_HEIGHT total regions
    - inventory
    - direction
    '''

    def features(self):
        if self._cached_features is None:
            x, y = self.pos
            hw = self.world.WINDOW_WIDTH // 2
            hh = self.world.WINDOW_HEIGHT // 2
            bhw = (self.world.WINDOW_WIDTH * self.world.WINDOW_WIDTH) // 2
            bhh = (self.world.WINDOW_HEIGHT * self.world.WINDOW_HEIGHT) // 2

            # Mask grid
            prepared = self._apply_mask()

            grid_feats = array.pad_slice(prepared, (x - hw, x + hw + 1),
                                         (y - hh, y + hh + 1))
            grid_feats_big = array.pad_slice(prepared, (x - bhw, x + bhw + 1),
                                             (y - bhh, y + bhh + 1))
            grid_feats_big_red = block_reduce(
                grid_feats_big,
                (self.world.WINDOW_WIDTH,
                 self.world.WINDOW_HEIGHT,
                 1),
                func=np.max)

            dir_features = np.zeros(4)
            dir_features[self.dir] = 1

            features = np.concatenate(
                (grid_feats.ravel(),
                 grid_feats_big_red.ravel(),
                 self.inventory,
                 dir_features)).astype(
                np.float32)
            assert len(features) == self.world.n_features
            self._cached_features = features

        return self._cached_features

    '''
    Get features input to the V model in world models baseline. All features except task, which will be added when preparing training data.
    '''

    def features_world_model(self):
        # Add mask
        masked = self._apply_mask()

        # Add agent position
        pos_slice = np.zeros((self.world.MAP_WIDTH, self.world.MAP_HEIGHT, 1))
        x, y = self.pos
        pos_slice[x, y, :] = 1
        pos_added = np.concatenate((masked, pos_slice), axis=2)
        map_feature = np.transpose(pos_added, (2, 0, 1))

        # Direction feature
        dir_features = np.zeros(4)
        dir_features[self.dir] = 1

        # Add inventory features
        if not hasattr(self.world, 'max_val_inventory'):
            self.world.set_world_models_consts()
        max_val = self.world.max_val_inventory
        inventory_vec = np.zeros(self.world.inventory_feature_dim)
        cur_idx = 0
        for thing in self.world.grabbable_indices:
            count = int(self.inventory[thing])
            if count >= max_val:
                print(count)
            assert(count < max_val)
            inventory_vec[cur_idx:cur_idx + count + 1] = 1
            cur_idx += max_val

        flat_features = np.concatenate(
            (dir_features, inventory_vec)).astype(
            np.float32)

        return map_feature, flat_features

    '''
    Add mask to the one-hot grid representation
    '''

    def _apply_mask(self):
        expanded_mask = np.expand_dims(self.mask, 2)
        masked = self.grid * expanded_mask
        prepared = np.concatenate((masked, 1 - expanded_mask), axis=2)
        return prepared


'''
Manages the advancement of plan
'''


class Commander(object):
    def __init__(self):
        super(Commander, self).__init__()

    def set_plan(self, plan, plan_in_name, transition_conditions):
        self.plan = plan
        self.plan_in_name = plan_in_name
        self.transition_conditions = transition_conditions
        self.current_stage = 0

    '''
    Reset the current execution progress
    '''

    def reset_stage(self):
        self.current_stage = 0

    '''
    Advance command given new state. Return (advanced, finished)
    '''

    def advance_command(self, state):
        condition = self.transition_conditions[self.current_stage]
        advanced = False
        if state.satisfy(condition):
            self.current_stage += 1
            advanced = True

        finished = False
        if self.current_stage >= len(self.plan):
            finished = True

        return (advanced, finished)

    def get_command(self):
        if self.current_stage >= len(self.plan):
            return None
        else:
            return self.plan[self.current_stage]

    def finish(self):
        self.current_stage = len(self.plan)

    def export_remaining_plan(self):
        if self.current_stage >= len(self.plan):
            return None
        else:
            plan_info = {}
            plan_info['plan'] = copy.copy(self.plan[self.current_stage:])
            plan_info['plan_in_name'] = copy.copy(
                self.plan_in_name[self.current_stage:])
            plan_info['transition_conditions'] = copy.deepcopy(
                self.transition_conditions[self.current_stage:])
            return plan_info
