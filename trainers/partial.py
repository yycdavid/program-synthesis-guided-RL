from misc import util
import copy
import torch
import itertools
import numpy as np
import yaml
from collections import defaultdict, namedtuple
from tqdm import tqdm
import os
import time
from synthesizer.solver import CraftSynthesizer
from synthesizer.box_solver import BoxSynthesizer
import pickle
import dill
import random
from random_hallucinator import RandomHallucinator
# dill.detect.trace(True)

import re
import json
from models.modular_ac import ModularACModel, ModularACConvModel
from models.simple_nn import ConditionModel
from models.world_models import act_world_models


class Trainer(object):
    def __init__(self, config, world, manager):
        super(Trainer, self).__init__()
        self.config = config
        self.world = world
        self.world_name = self.config.world.name
        if self.world_name == "BoxWorld":
            if config.trainer.tasks == "all":
                self.tasks = [i + 1 for i in range(self.world.max_goal_length)]
            else:
                self.tasks = [int(config.trainer.tasks)]

        else:
            if config.trainer.tasks == "all":
                self.tasks = world.grabbable_indices
            else:
                chosen_index = world.cookbook.index[config.trainer.tasks]
                task_valid = False
                for task_id in world.grabbable_indices:
                    if chosen_index == task_id:
                        task_valid = True
                        break
                assert task_valid, "Invalid task speficied in config file"
                self.tasks = [chosen_index]

        self.manager = manager
        self.entropy_ratio = self.config.trainer.entropy_ratio

        self.n_iters = config.trainer.n_iters  # Max number of rollouts
        # Number of updates to recompute task probabilities
        self.n_update = config.trainer.n_update
        self.n_batch = config.trainer.n_batch  # Batch size for each update
        self.n_replan = config.trainer.n_replan  # Replan every n_replan steps
        self.n_batch_test = 1
        self.discount = 0.9
        self.id = 0

        if self.world_name == "BoxWorld":
            self.synthesizer = BoxSynthesizer()
        else:
            self.synthesizer = CraftSynthesizer()
        self.generator = None

        if config.model.name == "WorldModels":
            self.generations = config.c_model.generations
            self.population = config.c_model.population
            self.es_n_rollouts = config.c_model.n_rollouts

    def _pre_sample_scenarios(self, scenarios_per_task, set_plan=True):
        self.scenario_pool = defaultdict(lambda: [])
        # This needs to be larger than batch size
        self.manager.say("Pre-sample scenarios...")
        for task in self.tasks:
            if self.world_name == "BoxWorld":
                per_task = scenarios_per_task
            else:
                # scenarios_per_task is total number of scenarios in this case
                per_task = int(scenarios_per_task * self.world.task_probs[task])

            for _ in tqdm(range(per_task)):
                if self.world_name == "BoxWorld" and self.config.trainer.dangle:
                    scenario = self.world.sample_scenario_with_goal(
                        task, dangle_box=True)
                else:
                    scenario = self.world.sample_scenario_with_goal(task)
                if set_plan:
                    scenario.init(self.synthesizer, self.id)
                else:
                    scenario.init(
                        self.synthesizer,
                        self.id,
                        use_ground_truth=False)
                plan_len = len(scenario.ground_truth_plan)
                self.scenario_pool[(task, plan_len)].append(scenario)
                self.id += 1

    def _sample_scenarios_test(self, scenarios_per_task):
        self.scenario_pool = defaultdict(lambda: [])
        self.manager.say("Pre-sample scenarios...")

        if self.world_name == "CraftWorldHard":
            # For this case, scenarios_per_task is the total number of tasks
            for task in self.tasks:
                per_task = int(scenarios_per_task * self.world.task_probs[task])
                if task == self.world.cookbook.index['gold'] or task == self.world.cookbook.index['gem']:
                    for (path_type, path_prob) in zip(self.world.path_types, self.world.path_probs):
                        per_type = int(per_task * path_prob)
                        for _ in tqdm(range(per_type)):
                            scenario = self.world.sample_scenario_with_goal(task, path_type=path_type)
                            scenario.init(
                                self.synthesizer,
                                self.id,
                                use_ground_truth=False)
                            self.scenario_pool[(task, path_type)].append(scenario)
                            self.id += 1
                else:
                    for _ in tqdm(range(per_task)):
                        scenario = self.world.sample_scenario_with_goal(task)
                        scenario.init(
                            self.synthesizer,
                            self.id,
                            use_ground_truth=False)
                        plan_len = len(scenario.ground_truth_plan)
                        self.scenario_pool[(task, plan_len)].append(scenario)
                        self.id += 1

        else:
            for task in self.tasks:
                possible_len = set()

                for _ in tqdm(range(scenarios_per_task)):
                    scenario = self.world.sample_scenario_with_goal(task)
                    scenario.init(
                        self.synthesizer,
                        self.id,
                        use_ground_truth=False)
                    plan_len = len(scenario.ground_truth_plan)
                    self.scenario_pool[(task, plan_len)].append(scenario)
                    possible_len.add(plan_len)
                    self.id += 1

                # Fill in for tasks with multiple plan lengths
                for plan_len in possible_len:
                    while len(self.scenario_pool[(
                            task, plan_len)]) < scenarios_per_task:
                        scenario = self.world.sample_scenario_with_goal(task)
                        scenario.init(
                            self.synthesizer,
                            self.id,
                            use_ground_truth=False)
                        plan_len_this = len(scenario.ground_truth_plan)
                        if len(self.scenario_pool[(
                                task, plan_len_this)]) < scenarios_per_task:
                            self.scenario_pool[(task, plan_len_this)].append(
                                scenario)
                            self.id += 1

    def _load_or_sample_test_scenarios(self, scenarios_per_task):
        if os.path.exists(self.config.test_set):
            with open(self.config.test_set, 'rb') as f:
                self.scenario_pool = dill.load(f)

        else:
            # Sample test scenarios
            self._sample_scenarios_test(scenarios_per_task)
            self.scenario_pool = dict(self.scenario_pool)
            # Save test scenarios
            with open(self.config.test_set, 'wb') as f:
                dill.dump(self.scenario_pool, f)

    '''
    Argument:
    - specific_index: if not None, specify the specific index in scenario_pool to get the scenario
    - use_oracle: if true, will use ground truth plan during testing. For ablation.

    Returns:
    - log_prob_history: log probabilities of the chosen action
    - critic_score_history: scores output by the critics
    - neg_entropy_history: negative entropy of the action distribution
    - reward_history: per-step rewards
    - done_history: whether the rollout already finished at the start of each step. Used as mask later
    - tasks: task for each rollout (with plan length)
    - states_history: history of states for each rollout (test mode only)
    '''

    def do_rollout(
            self,
            model,
            possible_tasks,
            task_probs,
            test=False,
            use_pre_sampled=True,
            specific_index=None,
            print_things=False,
            use_oracle=False,
            get_completions=False,
            optimistic=False,
            n_completions=3,
            reuse_key=False,
            get_replan=False,
            rand_explore=False,
            get_exec_acc=False,
            get_hall_acc=False):
        do_replan = (
            test and (isinstance(
                model,
                ModularACModel) or isinstance(model, ModularACConvModel) or isinstance(model, ConditionModel)) and (
                not use_oracle))
        if test:
            n_batch = self.n_batch_test
            assert (not get_exec_acc) or use_oracle
        else:
            n_batch = self.n_batch
            assert (not get_exec_acc)

        # Sample scenarios for this batch
        if get_completions:
            completion_hist = {}
            scenarios, tasks, tasks_and_len, states_before, commands, plans, completed_states = self._sample_batch(
                n_batch, possible_tasks, task_probs, use_pre_sampled=use_pre_sampled, specific_index=specific_index, do_replan=do_replan, get_plan=test, print_things=print_things, get_completions=True, n_completions=n_completions)
            completion_hist[0] = completed_states
            if get_replan:
                replan_hist = []
                replan_hist.append(False)
        else:
            scenarios, tasks, tasks_and_len, states_before, commands, plans = self._sample_batch(
                n_batch, possible_tasks, task_probs, use_pre_sampled=use_pre_sampled, specific_index=specific_index, do_replan=do_replan, get_plan=test, print_things=print_things, optimistic=optimistic, n_completions=n_completions)

        if get_hall_acc:
            assert get_completions
            assert len(scenarios) == 1
            hall_acc_results = {}
            results = self._hall_acc(states_before[0], completed_states)
            hall_acc_results[0] = results

        if reuse_key and (not use_pre_sampled):
            for scenario in scenarios:
                scenario.set_reuse_key()

        if rand_explore:
            for scenario in scenarios:
                scenario.check_random_explore(self.world)

        model.init(tasks)

        # initialize timer
        timer = self.config.trainer.max_timesteps
        done = [False for _ in range(n_batch)]

        # Histories, each entry is shape (N,)
        log_prob_history = []  # log prob of the action at step t
        critic_score_history = []  # critic score at the state at the beginning of step t
        neg_entropy_history = []  # Negative entropy at step t
        reward_history = []  # reward[t]: the reward after the action of step t
        done_history = []  # done[t]: is it done at the start of step t
        states_history = []
        states_history.append(states_before)
        if test:
            plan_history = []
            plan_history.append(plans)

        # act!
        while not all(done) and timer > 0:
            # Record done
            done_history.append(copy.deepcopy(done))

            # Only compute action for commands != None
            actions, log_probs, critic_scores, neg_entropies = model.act(
                states_before, commands)

            if rand_explore:
                for (i, scenario) in enumerate(scenarios):
                    if not scenario.check_random_explore(self.world):
                        actions[i] = np.random.randint(self.world.n_actions-1)
                        # Any constant would be fine, just to remove gradient
                        log_probs[i] = 0.0
                        neg_entropies[i] = 1.0

            # Record log_probs, critic_scores
            log_prob_history.append(log_probs)
            critic_score_history.append(critic_scores)
            neg_entropy_history.append(neg_entropies)

            # Step, get reward, record
            if get_completions:
                if get_replan:
                    states_after, commands_next, rewards, plans, completed_states, plan_changed = self._batch_step(
                        scenarios, actions, done, states_before, tasks, timer, get_plan=test, do_replan=do_replan, get_completions=True, n_completions=n_completions, get_replan=get_replan)
                    replan_hist.append(plan_changed)
                else:
                    states_after, commands_next, rewards, plans, completed_states = self._batch_step(
                        scenarios, actions, done, states_before, tasks, timer, get_plan=test, do_replan=do_replan, get_completions=True, n_completions=n_completions)

                if completed_states is not None:
                    completion_hist[self.config.trainer.max_timesteps -
                                    timer] = completed_states

                    if get_hall_acc:
                        results = self._hall_acc(states_after[0], completed_states)
                        hall_acc_results[self.config.trainer.max_timesteps-timer] = results
            else:
                states_after, commands_next, rewards, plans = self._batch_step(
                    scenarios, actions, done, states_before, tasks, timer, get_plan=test, do_replan=do_replan, optimistic=optimistic, n_completions=n_completions)

            # Record reward, update
            reward_history.append(rewards)
            states_history.append(states_after)
            if test:
                plan_history.append(plans)
            states_before = states_after
            commands = commands_next
            timer -= 1

        results_dict = {}
        results_dict['log_prob_history'] = log_prob_history
        results_dict['critic_score_history'] = critic_score_history
        results_dict['neg_entropy_history'] = neg_entropy_history
        results_dict['done_history'] = done_history
        results_dict['reward_history'] = reward_history
        results_dict['tasks'] = tasks_and_len
        if test:
            results_dict['states_history'] = states_history
            results_dict['plan_history'] = plan_history

        if get_completions:
            results_dict['completion_hist'] = completion_hist
            if get_replan:
                results_dict['replan_hist'] = replan_hist

        if get_exec_acc:
            component_cnt = defaultdict(lambda: 0)
            success_cnt = defaultdict(lambda: 0)
            for scenario in scenarios:
                cmd = scenario.commander
                for c in cmd.plan[:cmd.current_stage]:
                    component_cnt[c] += 1
                    success_cnt[c] += 1
                if cmd.current_stage < len(cmd.plan):
                    component_cnt[cmd.plan[cmd.current_stage]] += 1
            results_dict['component_cnt'] = component_cnt
            results_dict['success_cnt'] = success_cnt

        if get_hall_acc:
            results_dict['hall_acc'] = hall_acc_results

        return results_dict

    def _hall_acc(self, concrete_state, completed_states):
        results = {
            "whole": [0, 0],
            "individual": [0, 0]
        }
        true_abs = self.world.get_abstract_state(concrete_state)
        true_state = self.world.parse_abstract_state(true_abs)
        for c in completed_states:
            n_same, n_total = self.world.compare_abs_states(true_state, c)
            if n_same == n_total:
                results["whole"][0] += 1
            results["whole"][1] += 1
            results["individual"][0] += n_same
            results["individual"][1] += n_total

        return results

    def _sample_batch(
            self,
            n_samples,
            possible_tasks,
            task_probs,
            use_pre_sampled=True,
            specific_index=None,
            do_replan=False,
            get_plan=False,
            print_things=False,
            get_completions=False,
            optimistic=False,
            n_completions=3):
        scenarios = []
        tasks = []
        tasks_and_len = []
        states_before = []
        commands = []
        plans = None
        if get_plan:
            plans = []

        id_sampled = set()

        for _ in range(n_samples):
            sampled = False
            while not sampled:
                task_and_len = possible_tasks[np.random.choice(
                    len(possible_tasks), p=task_probs)]
                if use_pre_sampled:
                    if specific_index is not None:
                        assert n_samples == 1, "Can only use batch size of 1 if you specify index to get scenario"
                        scenario = self.scenario_pool[task_and_len][specific_index]
                    else:
                        scenario = np.random.choice(
                            self.scenario_pool[task_and_len])

                    if scenario.id not in id_sampled:
                        scenario.reset(keep_plan=(not do_replan))
                    else:
                        continue
                else:
                    if isinstance(task_and_len[1], str):
                        scenario = self.world.sample_scenario_with_goal(
                            task_and_len[0], path_type=task_and_len[1])
                    else:
                        scenario = self.world.sample_scenario_with_goal(
                        task_and_len[0])

                    scenario.init(
                        self.synthesizer,
                        use_ground_truth=(
                            not do_replan))
                    if (not isinstance(task_and_len[1], str)) and (len(scenario.ground_truth_plan) != task_and_len[1]):
                        continue

                if do_replan:
                    if self.world_name == "BoxWorld":
                        plan_changed, completed_states, success = scenario.plan(
                            self.synthesizer,
                            self.generator,
                            print_things=print_things,
                            optimistic=optimistic,
                            n_completions=n_completions,
                            fail_with_simplest=True)
                    else:
                        plan_changed, completed_states, success = scenario.plan(
                            self.synthesizer,
                            self.generator,
                            print_things=print_things,
                            optimistic=optimistic,
                            n_completions=n_completions)
                    if not success:
                        continue
                    '''if True and len(scenario.commander.plan) > 1:
                        continue'''

                if use_pre_sampled:
                    id_sampled.add(scenario.id)

                sampled = True
                scenarios.append(scenario)
                tasks.append(task_and_len[0])
                tasks_and_len.append(task_and_len)
                states_before.append(scenario.get_state())
                commands.append(scenario.query_command())
                if get_plan:
                    plans.append(scenario.get_plan())

        if get_completions:
            return scenarios, tasks, tasks_and_len, states_before, commands, plans, completed_states
        else:
            return scenarios, tasks, tasks_and_len, states_before, commands, plans

    def _batch_step(
            self,
            scenarios,
            actions,
            done,
            states_before,
            tasks,
            timer,
            get_plan=False,
            do_replan=False,
            get_completions=False,
            optimistic=False,
            n_completions=3,
            get_replan=False):
        if get_completions:
            completed_states = None
        n_batch = len(actions)
        # Step, get reward, record
        states_after = [None for _ in range(n_batch)]
        commands_next = [None for _ in range(n_batch)]
        rewards = [0.0 for _ in range(n_batch)]
        plans = None
        if get_plan:
            plans = [None for _ in range(n_batch)]
        if get_replan:
            plan_changed = False
        # Only step for action != None
        for i in range(n_batch):
            if (actions[i] is None) or done[i]:
                # actions[i] is None == done[i]
                assert done[i]
                states_after[i] = states_before[i]
            else:
                reward, state_after, finished, _ = scenarios[i].step(
                    actions[i])
                # If finish, check goal, set done
                if finished:
                    done[i] = True
                    if self.world_name != "BoxWorld":
                        goal_condition = {}
                        goal_condition[tasks[i]] = 1
                        if state_after.satisfy(goal_condition):
                            reward = self.world.FINAL_REWARD
                else:
                    if do_replan:
                        # Replan if reaches n_replan
                        if (self.config.trainer.max_timesteps -
                                timer + 1) % self.n_replan == 0:
                            plan_changed, completed_states, _ = scenarios[i].plan(
                                self.synthesizer, self.generator, optimistic=optimistic, n_completions=n_completions)
                            if self.world_name != "BoxWorld":
                                if plan_changed:
                                    reward += self.world.REPLAN_REWARD

                commands_next[i] = scenarios[i].query_command()
                rewards[i] = reward
                states_after[i] = state_after

            if get_plan:
                plans[i] = scenarios[i].get_plan()

        if get_completions:
            if get_replan:
                return states_after, commands_next, rewards, plans, completed_states, plan_changed
            else:
                return states_after, commands_next, rewards, plans, completed_states
        else:
            return states_after, commands_next, rewards, plans

    def _get_fitlist(self, reward_histories):
        fitlist = []
        for reward_history in reward_histories:
            rewards_reshaped = np.transpose(
                np.array(reward_history))  # (N, T)
            total_reward = np.sum(rewards_reshaped)

            # Add bonus for early finish
            batch_size, max_steps = rewards_reshaped.shape
            for i in range(batch_size):
                reward_hist = rewards_reshaped[i, :]
                if max(reward_hist) > self.world.FINAL_REWARD - 1e-3:
                    # This rollout solves the task
                    total_reward += (max_steps -
                                     np.argmax(reward_hist)) / max_steps

            fitlist.append(total_reward)

        return np.array(fitlist)

    def train(self, model, number_of_saves=50, reuse_key=False):
        # Prepare model
        model.prepare(self.world, self)

        # Pre-sample scenarios
        self._pre_sample_scenarios(self.config.trainer.per_task, set_plan=True)

        if reuse_key and self.world_name == "BoxWorld":
            for task, scenarios in self.scenario_pool.items():
                for scenario in scenarios:
                    scenario.set_reuse_key()

        if hasattr(self.config.model, "load_saved") and self.config.model.load_saved != "None":
            assert os.path.exists(
                self.config.model.load_saved), "No trained model"
            model.load_state_dict(torch.load(self.config.model.load_saved))
            model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.trainer.lr)

        if self.config.trainer.use_curriculum:
            if hasattr(self.config.trainer, "starting_difficulty"):
                max_stages = self.config.trainer.starting_difficulty
            else:
                max_stages = 1
        else:
            max_stages = 100

        i_iter = 0
        task_probs = []
        task_running_score = {}
        for task_and_len in self.scenario_pool.keys():
            task_running_score[task_and_len] = 0.0
        score_discount = 0.9

        save_every = self.n_iters / number_of_saves
        # Save model
        torch.save(model.state_dict(), self._get_model_path(i_iter))
        next_save = save_every

        # Save total steps interacted with env
        total_steps = 0

        while i_iter < self.n_iters:
            possible_tasks = [
                t for t in self.scenario_pool.keys() if t[1] <= max_stages]

            # re-initialize task probs if necessary
            if len(task_probs) != len(possible_tasks):
                task_probs = np.ones(len(possible_tasks)) / len(possible_tasks)

            task_rewards = defaultdict(lambda: 0)
            task_counts = defaultdict(lambda: 0)
            task_finish_time = defaultdict(lambda: [])
            total_reward = 0.0
            cum_critic_loss = 0.0
            cum_actor_loss = 0.0
            for j in tqdm(range(self.n_update)):
                i_iter += self.n_batch
                # Get rollout
                rollout_results = self.do_rollout(
                    model, possible_tasks, task_probs)

                # Process rewards
                reward_history = rollout_results['reward_history']
                rewards_reshaped = np.transpose(
                    np.array(reward_history))  # (N, T)
                total_reward += np.sum(rewards_reshaped)
                self.discounted_future_reward = []
                running_reward = 0.0
                for reward_step in reward_history[::-1]:
                    running_reward = np.array(
                        reward_step) + running_reward * self.discount
                    self.discounted_future_reward.insert(0, running_reward)
                rewards_processed = np.transpose(
                    np.array(self.discounted_future_reward))  # (N, T)

                # Reshape things
                log_probs = torch.cat(
                    rollout_results['log_prob_history'],
                    dim=1)  # (N, T)
                critic_scores = torch.cat(
                    rollout_results['critic_score_history'], dim=1)  # (N, T)
                dones = torch.FloatTensor(rollout_results['done_history'])
                mask = torch.transpose(
                    1.0 - dones,
                    0,
                    1).to(
                    model.device)  # (N, T)
                neg_entropy = torch.cat(
                    rollout_results['neg_entropy_history'], dim=1)  # (N, T)
                rewards = torch.FloatTensor(
                    rewards_processed).to(model.device)  # (N, T)

                # Compute loss and update
                advantage = rewards - critic_scores
                actor_loss = torch.sum(log_probs * advantage * mask).mul(-1)
                entropy_regularization = torch.sum(
                    neg_entropy * mask) * self.entropy_ratio
                critic_loss = torch.sum(torch.square(advantage) * mask)
                total_loss = actor_loss + entropy_regularization + critic_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                cum_critic_loss += critic_loss.item()
                cum_actor_loss += actor_loss.item()

                # Update task statistics
                for (i, task) in enumerate(rollout_results['tasks']):
                    task_counts[task] += 1
                    reward_hist = rewards_reshaped[i, :]
                    if max(reward_hist) > self.world.FINAL_REWARD - 1e-3:
                        # This rollout solves the task
                        if self.world_name == "BoxWorld":
                            task_rewards[task] += 1.0
                        else:
                            task_rewards[task] += self.world.FINAL_REWARD
                        task_finish_time[task].append(
                            np.argmax(reward_hist) + 1)

                        total_steps += np.argmax(reward_hist) + 1
                    else:
                        task_finish_time[task].append(
                            self.config.trainer.max_timesteps + 1)

                        total_steps += 100

            # Update task running score
            for task in task_running_score:
                task_running_score[task] = task_running_score[task] * \
                    score_discount
            for task in possible_tasks:
                if task_counts[task] == 0:
                    score = task_running_score[task] * 0.9
                else:
                    score = 1. * task_rewards[task] / \
                        (task_counts[task] + 1e-3)
                task_running_score[task] += score * (1 - score_discount)

            # recompute task probs
            task_probs = np.zeros(len(possible_tasks))
            scores = []
            for i, task in enumerate(possible_tasks):
                task_probs[i] = task_running_score[task]
                scores.append(task_running_score[task])
            task_probs = 1 - task_probs
            task_probs += 0.01
            task_probs /= task_probs.sum()
            min_score = min(scores)

            # Move to longer plans if performance reaches threshold
            if min_score > self.config.trainer.improvement_threshold:
                max_stages += 1

            # Logging
            self.manager.say("[Training iter] {}".format(i_iter))
            self.manager.say("[Avg reward] {}".format(
                total_reward / (self.n_batch * self.n_update)))
            self.manager.say("[Avg critic loss] per rollout {}".format(
                cum_critic_loss / (self.n_batch * self.n_update)))
            self.manager.say("[Avg actor loss] per rollout {}".format(
                cum_actor_loss / (self.n_batch * self.n_update)))
            for i, task in enumerate(possible_tasks):
                if task_counts[task] == 0:
                    avg_finish_step = 0
                else:
                    avg_finish_step = np.mean(task_finish_time[task])

                if self.world_name == "BoxWorld":
                    self.manager.say(
                        "[task] {}-{}: {}/{} solved, avg finished at step {} ".format(
                            task[0], task[1], int(
                                task_rewards[task]), task_counts[task], avg_finish_step))
                else:
                    self.manager.say(
                        "[task] {}-{}: {}/{} solved, avg finished at step {} ".format(
                            self.world.cookbook.index.get(
                                task[0]), task[1], int(
                                task_rewards[task]), task_counts[task], avg_finish_step))

            # Save model
            if i_iter >= next_save:
                torch.save(model.state_dict(), self._get_model_path(i_iter))
                # Save total steps so far
                steps_path = os.path.join(
                    self.manager.result_folder,
                    "steps_{}.txt".format(i_iter))

                with open(steps_path, 'w') as fout:
                    fout.write("{}".format(total_steps))

                next_save += save_every

    def test(
            self,
            model,
            generative_model,
            model_to_test=None,
            viz_name='viz',
            optimistic=False,
            use_oracle=False,
            rand_explore=False,
            reuse_key=False,
            index_from_sampled=-1,
            seed=10,
            test_task='all',
            update_test_setting=False):
        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        if model_to_test is None or model_to_test == 'all':
            # Get the latest model
            latest_save_point = -1
            for file in os.listdir(self.manager.result_folder):
                if file.endswith(".pt"):
                    words = re.split(r'_|\.', file)
                    iter_number = int(words[1])
                    if iter_number > latest_save_point:
                        latest_save_point = iter_number
            assert latest_save_point >= 0, "No saved model"
            save_point_to_test = latest_save_point
        else:
            save_point_to_test = int(model_to_test)

        model_path = self._get_model_path(save_point_to_test)

        # Load model
        model.prepare(self.world, self)
        assert os.path.exists(model_path), "No trained model"
        model.load_state_dict(torch.load(model_path))
        self.generator = generative_model

        if test_task == "all":
            possible_tasks = list(self.scenario_pool.keys())
        else:
            task_spec = test_task.split('-')
            if self.world_name == "BoxWorld":
                possible_tasks = [
                    (int(task_spec[0]), int(task_spec[1]))]
            else:
                task_suffix = int(task_spec[1]) if task_spec[1].isnumeric() else task_spec[1]
                possible_tasks = [
                    (self.world.cookbook.index[task_spec[0]], task_suffix)]
        task_probs = []
        # re-initialize task probs if necessary
        if len(task_probs) != len(possible_tasks):
            task_probs = np.ones(len(possible_tasks)) / len(possible_tasks)

        get_completions = (not optimistic) and (not use_oracle) and (isinstance(
            model,
            ModularACModel) or isinstance(model, ModularACConvModel) or isinstance(model, ConditionModel))

        if index_from_sampled != -1:
            # Load or pre-sample scenarios
            if self.world_name == "CraftWorldHard":
                self._load_or_sample_test_scenarios(50)
            else:
                self._load_or_sample_test_scenarios(10)
            self.manager.say("Loaded test scenarios")
            if ((not isinstance(model, ModularACModel)) and (
                    not isinstance(model, ModularACConvModel)) and (
                        not isinstance(model, ConditionModel))) or use_oracle:
                # Let the scenarios use ground truth plan if we are not doing
                # replanning. The plans won't be used, just for unified
                # interface
                for task, scenarios in self.scenario_pool.items():
                    for scenario in tqdm(scenarios):
                        scenario.use_ground_truth(self.synthesizer)

            if reuse_key and self.world_name == "BoxWorld":
                for task, scenarios in self.scenario_pool.items():
                    for scenario in scenarios:
                        scenario.set_reuse_key()

            if update_test_setting:
                for task, scenarios in self.scenario_pool.items():
                    for scenario in scenarios:
                        scenario.world.VIEW_RANGE = self.world.VIEW_RANGE
                        scenario.world.non_det = self.world.non_det

            rollout_results = self.do_rollout(
                model,
                possible_tasks,
                task_probs,
                test=True,
                specific_index=index_from_sampled,
                use_oracle=use_oracle,
                optimistic=optimistic,
                rand_explore=rand_explore,
                get_completions=get_completions)

        else:

            rollout_results = self.do_rollout(
                model,
                possible_tasks,
                task_probs,
                test=True,
                use_pre_sampled=False,
                print_things=True,
                get_completions=get_completions,
                optimistic=optimistic,
                use_oracle=use_oracle,
                rand_explore=rand_explore,
                reuse_key=reuse_key)

        task = rollout_results['tasks'][0][0]
        if self.world_name == "BoxWorld":
            print("Goal: {}".format(task))
        else:
            print("Goal: {}".format(self.world.cookbook.index.get(task)))

        states_history = rollout_results['states_history']
        unpacked = [states[0] for states in states_history]
        plan_history = rollout_results['plan_history']
        unpacked_plans = [plan[0] for plan in plan_history]
        video_folder = os.path.join(self.manager.result_folder, viz_name)
        util.create_dir(video_folder)

        if get_completions:

            # Save completions
            com_save_file = os.path.join(video_folder, "completions.json")
            if self.world_name != "BoxWorld":
                processed = self._process_completion(
                    rollout_results['completion_hist'])
            else:
                processed = self._process_completion_box(
                    rollout_results['completion_hist'])
            with open(com_save_file, 'w') as f:
                f.write(json.dumps(processed, indent=2))

        self.world.pretty_rollout(unpacked, video_folder, plans=unpacked_plans)

        self.world.traj_vis(unpacked, video_folder, unpacked_plans, 0, single_plan=True)

        self.world.traj_vis(unpacked, video_folder, unpacked_plans, 20, single_plan=True)

        self.world.traj_vis(unpacked, video_folder, unpacked_plans, len(unpacked_plans)-1, single_plan=True)

    def inspect_replan(
            self,
            model,
            generative_model,
            model_to_test=None,
            viz_name='viz',
            optimistic=False,
            reuse_key=False,
            test_task='all',
            index_from_sampled=-1):
        assert self.world_name == "CraftWorldHard", "Currently only support craft world"
        if model_to_test is None or model_to_test == 'all':
            # Get the latest model
            latest_save_point = -1
            for file in os.listdir(self.manager.result_folder):
                if file.endswith(".pt"):
                    words = re.split(r'_|\.', file)
                    iter_number = int(words[1])
                    if iter_number > latest_save_point:
                        latest_save_point = iter_number
            assert latest_save_point >= 0, "No saved model"
            save_point_to_test = latest_save_point
        else:
            save_point_to_test = int(model_to_test)

        model_path = self._get_model_path(save_point_to_test)

        # Load model
        model.prepare(self.world, self)
        assert os.path.exists(model_path), "No trained model"
        model.load_state_dict(torch.load(model_path))
        self.generator = generative_model

        task_spec = test_task.split('-')
        if self.world_name == "BoxWorld":
            possible_tasks = [
                (int(task_spec[0]), int(task_spec[1]))]
        else:
            task_suffix = int(task_spec[1]) if task_spec[1].isnumeric() else task_spec[1]
            possible_tasks = [
                (self.world.cookbook.index[task_spec[0]], task_suffix)]
        task_probs = [1.0]

        get_completions = not optimistic
        get_desired = False
        while not get_desired:
            # Set random seed, print out
            cur_seed = random.randint(1, 5000)
            '''
            # For Figure 1
            cur_seed = 3444
            if cur_seed == 3219 or cur_seed == 122 or cur_seed == 3273:
                continue
            '''
            self.world.random = np.random.RandomState(cur_seed)
            np.random.seed(cur_seed)
            torch.manual_seed(cur_seed)

            rollout_results = self.do_rollout(
                model,
                possible_tasks,
                task_probs,
                test=True,
                use_pre_sampled=False,
                print_things=True,
                get_completions=get_completions,
                optimistic=optimistic,
                get_replan=True)

            # Check if replan happened
            plan_history = rollout_results['plan_history']
            unpacked_plans = [plan[0] for plan in plan_history]

            '''
            # For Figure 1
            if len(unpacked_plans[0]['plan']) == 1:
                get_desired = True
            '''

            # For optim
            #if len(unpacked_plans) < 99:
            #    get_desired = True

            # For direct with two zones
            '''init_state = rollout_results['states_history'][0][0]
            has_bound = (init_state.grid[:, :, self.world.cookbook.index['water']].any()) or (init_state.grid[:, :, self.world.cookbook.index['stone']].any())
            if len(unpacked_plans) < 99 and len(unpacked_plans[0]['plan']) > 1 and has_bound:
                get_desired = True'''

            # For three programs
            programs_appeared = []
            step_to_check = 0
            while step_to_check < len(unpacked_plans) - 1:
                if len(unpacked_plans[step_to_check]['plan_in_name']) == 1:
                    programs_appeared.append('gold')
                else:
                    programs_appeared.append(unpacked_plans[step_to_check]['plan_in_name'][-2])
                step_to_check += self.n_replan

            prog_set = set(programs_appeared)
            if (len(unpacked_plans) < 99) and (programs_appeared[0] == 'gold') and (len(prog_set) >= 3):
                get_desired = True

        print("The seed used to get desired: {}".format(cur_seed))

        # After getting a desired one, visualize
        states_history = rollout_results['states_history']
        unpacked = [states[0] for states in states_history]
        video_folder = os.path.join(self.manager.result_folder, viz_name)
        util.create_dir(video_folder)

        # Save traj data
        traj_dict = {
            'unpacked': [st.export_for_save() for st in unpacked],
            'unpacked_plans': unpacked_plans,
        }
        traj_dict_file = os.path.join(video_folder, 'traj.pkl')
        with open(traj_dict_file, 'wb') as f:
            dill.dump(traj_dict, f)

        if not optimistic:

            # Save completions
            com_save_file = os.path.join(video_folder, "completions.json")
            if self.world_name != "BoxWorld":
                processed = self._process_completion(
                    rollout_results['completion_hist'])
            else:
                processed = self._process_completion_box(
                    rollout_results['completion_hist'])
            with open(com_save_file, 'w') as f:
                f.write(json.dumps(processed, indent=2))

        self.world.pretty_rollout(unpacked, video_folder, plans=unpacked_plans)

        step_to_plot = 0
        while step_to_plot < len(unpacked_plans):
            self.world.traj_vis(unpacked, video_folder, unpacked_plans, step_to_plot, single_plan=True)
            step_to_plot += self.n_replan
        self.world.traj_vis(unpacked, video_folder, unpacked_plans, len(unpacked_plans)-1, single_plan=True)

    '''
    Process completion_hist to be ready to save as json
    '''

    def _process_completion(self, completion_hist):
        new_dict = {}
        for (iter, completions) in completion_hist.items():
            if completions is None:
                new_dict[iter] = None
            else:
                new_completions = []
                for completion in completions:
                    new_completion = {}
                    new_completion['n_zones'] = completion['n_zones']
                    new_completion['boundaries'] = {str(
                        key[0]) + str(key[1]): value for (key, value) in completion['boundaries'].items()}
                    new_counts = {}
                    for (zone, counts) in completion['counts'].items():
                        new_counts[zone] = {
                            self.world.cookbook.index.get(thing): count for (
                                thing, count) in counts.items()}
                    new_completion['counts'] = new_counts
                    new_completions.append(new_completion)

                new_dict[iter] = new_completions

        return new_dict

    def _process_completion_box(self, completion_hist):
        new_dict = {}
        for (iter, completions) in completion_hist.items():
            new_completions = []
            for completion in completions:
                new_completion = {}
                new_completion['loose'] = list(completion['loose'])
                new_completion['key'] = completion['key'].item()
                new_completion['box'] = {
                    str(key[0]) + str(key[1]): value for (key, value) in completion['box'].items()}
                new_completions.append(new_completion)
            new_dict[iter] = new_completions
        return new_dict

    '''
    If models_to_test='all', will test all saved models for this experiment.
    Otherwise, models_to_test should provide a certain iteration number, e.g. '1000', to specify which saved model to test

    Format of saved stats:
    [
        {
            "iter": 100,
            "result": [
                {
                    "task": 'gem-6',
                    "reward": [1.0, 0.0, 1.0],
                    "finish_time": [93, 95, 101],
                },
                ...
            ]
        },
        ...
    ]
    '''

    def test_statistics(
            self,
            model,
            generative_model,
            models_to_test="all",
            use_oracle=False,
            seed=10,
            optimistic=False,
            rand_explore=False,
            n_completions=3,
            reuse_key=False,
            save_separate=False,
            suffix='',
            save_time=False,
            get_exec_acc=False,
            update_test_setting=False,
            get_hall_acc=False):
        if save_time:
            start_time = time.time()
        # Set seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load or pre-sample scenarios
        if self.world_name == "CraftWorldHard":
            self._load_or_sample_test_scenarios(50)
        else:
            self._load_or_sample_test_scenarios(10)
        self.manager.say("Loaded test scenarios")
        if ((not isinstance(model, ModularACModel)) and (
                not isinstance(model, ModularACConvModel)) and (
                    not isinstance(model, ConditionModel))) or use_oracle:
            # Let the scenarios use ground truth plan if we are not doing
            # replanning. The plans won't be used, just for unified interface
            for task, scenarios in self.scenario_pool.items():
                for scenario in tqdm(scenarios):
                    scenario.use_ground_truth(self.synthesizer)

        if reuse_key and self.world_name == "BoxWorld":
            for task, scenarios in self.scenario_pool.items():
                for scenario in scenarios:
                    scenario.set_reuse_key()

        if update_test_setting:
            for task, scenarios in self.scenario_pool.items():
                for scenario in scenarios:
                    scenario.world.VIEW_RANGE = self.world.VIEW_RANGE
                    scenario.world.non_det = self.world.non_det

        # Get list of saved models
        save_points = []
        if models_to_test == 'all' or len(models_to_test.split("_")) == 2:
            for file in os.listdir(self.manager.result_folder):
                if file.endswith(".pt") and file.startswith("model"):
                    words = re.split(r'_|\.', file)
                    save_points.append(int(words[1]))
            if len(models_to_test.split("_")) == 2:
                test_range = models_to_test.split("_")
                min_iter = int(test_range[0])
                max_iter = int(test_range[1])
                save_points = [
                    i for i in save_points if (
                        (i >= min_iter) and (
                            i <= max_iter))]
        else:
            save_points.append(int(models_to_test))

        model.prepare(self.world, self)
        self.generator = generative_model

        all_stats = []
        if get_exec_acc:
            all_exec_acc = []
        if get_hall_acc:
            all_hall_acc = []
        save_points.sort()
        sep_dir = os.path.join(self.manager.result_folder, "stats")
        for iter_number in save_points:
            # Load model
            model_path = self._get_model_path(iter_number)
            assert os.path.exists(model_path), "No trained model"
            model.load_state_dict(torch.load(model_path))

            # Test 10 rollouts for each task
            possible_tasks = list(self.scenario_pool.keys())
            result_stats = []
            if get_exec_acc:
                component_cnt = defaultdict(lambda: 0)
                success_cnt = defaultdict(lambda: 0)
            if get_hall_acc:
                hall_acc = {}
            for i, task_and_len in enumerate(possible_tasks):
                task_probs = np.zeros(len(possible_tasks))
                task_probs[i] = 1

                task_rewards = []
                task_finish_time = []
                num_scenarios = len(self.scenario_pool[task_and_len])
                for scenario_i in tqdm(range(num_scenarios)):
                    rollout_results = self.do_rollout(
                        model,
                        possible_tasks,
                        task_probs,
                        test=True,
                        specific_index=scenario_i,
                        use_oracle=use_oracle,
                        optimistic=optimistic,
                        n_completions=n_completions,
                        rand_explore=rand_explore,
                        get_exec_acc=get_exec_acc,
                        get_hall_acc=get_hall_acc,
                        get_completions=get_hall_acc)
                    # Collect results
                    reward_history = rollout_results['reward_history']
                    rewards_reshaped = np.transpose(
                        np.array(reward_history))  # (N, T)
                    reward_hist = rewards_reshaped[0, :]
                    if max(reward_hist) > self.world.FINAL_REWARD - 1e-3:
                        # This rollout solves the task
                        if self.world_name == "BoxWorld":
                            task_rewards.append(1.0)
                        else:
                            task_rewards.append(self.world.FINAL_REWARD)
                        task_finish_time.append(
                            int(np.argmax(reward_hist)) + 1)
                    else:
                        task_rewards.append(0.0)
                        task_finish_time.append(
                            self.config.trainer.max_timesteps + 1)

                    if get_exec_acc:
                        for c in rollout_results['component_cnt']:
                            component_cnt[c] += rollout_results['component_cnt'][c]
                        for c in rollout_results['success_cnt']:
                            success_cnt[c] += rollout_results['success_cnt'][c]

                    if get_hall_acc:
                        r = rollout_results['hall_acc']
                        for step in r:
                            if not step in hall_acc:
                                hall_acc[step] = r[step]
                            else:
                                hall_acc[step]['whole'][0] += r[step]['whole'][0]
                                hall_acc[step]['whole'][1] += r[step]['whole'][1]
                                hall_acc[step]['individual'][0] += r[step]['individual'][0]
                                hall_acc[step]['individual'][1] += r[step]['individual'][1]

                # Print results
                avg_finish_step = np.mean(task_finish_time)
                if self.world_name == "BoxWorld":
                    self.manager.say(
                        "[task] {}-{}: {}/{} solved, avg finished at step {} ".format(
                            task_and_len[0], task_and_len[1], int(
                                sum(task_rewards)), num_scenarios, avg_finish_step))
                else:
                    self.manager.say(
                        "[task] {}-{}: {}/{} solved, avg finished at step {} ".format(
                            self.world.cookbook.index.get(
                                task_and_len[0]), task_and_len[1], int(
                                sum(task_rewards)), num_scenarios, avg_finish_step))

                # Save stats
                stats_this_task = {}
                if self.world_name == "BoxWorld":
                    task_name = "{}-{}".format(
                        task_and_len[0], task_and_len[1])
                else:
                    task_name = "{}-{}".format(
                        self.world.cookbook.index.get(
                            task_and_len[0]), task_and_len[1])
                stats_this_task['task'] = task_name
                stats_this_task['reward'] = task_rewards
                stats_this_task['finish_time'] = task_finish_time
                result_stats.append(stats_this_task)

            stats_this_model = {}
            stats_this_model['iter'] = iter_number
            stats_this_model['result'] = result_stats
            all_stats.append(stats_this_model)

            if get_exec_acc:
                exec_acc_this = {}
                exec_acc_this['iter'] = iter_number
                exec_acc = {}
                for c in self.world.action_indices:
                    exec_acc[c] = (success_cnt[c], component_cnt[c])
                exec_acc_this['result'] = exec_acc
                all_exec_acc.append(exec_acc_this)

            if get_hall_acc:
                hall_acc_this = {"iter": iter_number, "result": hall_acc}
                all_hall_acc.append(hall_acc_this)

            if save_separate:
                util.create_dir(sep_dir)
                if use_oracle:
                    save_file = os.path.join(
                        sep_dir,
                        "stats_{}_oracle_seed_{}{}.json".format(
                            iter_number,
                            seed,
                            suffix))
                elif optimistic:
                    save_file = os.path.join(
                        sep_dir,
                        "stats_{}_optim_seed_{}{}.json".format(
                            iter_number,
                            seed,
                            suffix))
                elif rand_explore:
                    save_file = os.path.join(
                        sep_dir,
                        "stats_{}_randexpl_seed_{}{}.json".format(
                            iter_number,
                            seed,
                            suffix))
                elif self.world.non_det:
                    save_file = os.path.join(
                        sep_dir,
                        "stats_{}_nd_seed_{}{}.json".format(
                            iter_number,
                            seed,
                            suffix))
                elif update_test_setting:
                    save_file = os.path.join(
                        sep_dir,
                        "stats_{}_full_seed_{}{}.json".format(
                            iter_number,
                            seed,
                            suffix))
                elif reuse_key and self.world_name == "BoxWorld":
                    save_file = os.path.join(
                        sep_dir,
                        "stats_{}_reuse_seed_{}{}.json".format(
                            iter_number,
                            seed,
                            suffix))
                else:
                    save_file = os.path.join(
                        sep_dir,
                        "stats_{}_N_{}_seed_{}{}.json".format(
                            iter_number,
                            n_completions,
                            seed,
                            suffix))
                with open(save_file, 'w') as f:
                    f.write(json.dumps(stats_this_model, indent=2))

                if get_exec_acc:
                    tgt_file = os.path.join(self.manager.result_folder, 'exec_acc_{}_seed_{}{}.txt'.format(iter_number, seed, suffix))
                    with open(tgt_file, 'w') as f:
                        f.write(json.dumps(exec_acc_this, indent=2))

                if get_hall_acc:
                    tgt_file = os.path.join(self.manager.result_folder, 'hall_acc_{}_seed_{}{}.txt'.format(iter_number, seed, suffix))
                    with open(tgt_file, 'w') as f:
                        f.write(json.dumps(hall_acc_this, indent=2))

        # Save all stats to a file
        if use_oracle:
            save_file = os.path.join(
                self.manager.result_folder,
                "stats_{}_oracle_seed_{}{}.json".format(
                    models_to_test,
                    seed,
                    suffix))
        elif optimistic:
            save_file = os.path.join(
                self.manager.result_folder,
                "stats_{}_optim_seed_{}{}.json".format(
                    models_to_test,
                    seed,
                    suffix))
        elif rand_explore:
            save_file = os.path.join(
                self.manager.result_folder,
                "stats_{}_randexpl_seed_{}{}.json".format(
                    models_to_test,
                    seed,
                    suffix))
        elif self.world.non_det:
            save_file = os.path.join(
                self.manager.result_folder,
                "stats_{}_nd_seed_{}{}.json".format(
                    models_to_test,
                    seed,
                    suffix))
        elif update_test_setting:
            save_file = os.path.join(
                self.manager.result_folder,
                "stats_{}_full_seed_{}{}.json".format(
                    models_to_test,
                    seed,
                    suffix))
        elif reuse_key and self.world_name == "BoxWorld":
            save_file = os.path.join(
                self.manager.result_folder,
                "stats_{}_reuse_N_{}_seed_{}{}.json".format(
                    models_to_test,
                    n_completions,
                    seed,
                    suffix))
        elif isinstance(generative_model, RandomHallucinator):
            save_file = os.path.join(
                self.manager.result_folder,
                "stats_{}_randhall_seed_{}{}.json".format(
                    models_to_test,
                    seed,
                    suffix))
        else:
            save_file = os.path.join(
                self.manager.result_folder,
                "stats_{}_N_{}_seed_{}{}.json".format(
                    models_to_test,
                    n_completions,
                    seed,
                    suffix))
        with open(save_file, 'w') as f:
            f.write(json.dumps(all_stats, indent=2))

        if save_time:
            end_time = time.time()
            time_file = os.path.join(self.manager.result_folder, 'time_N_{}_seed_{}.txt'.format(n_completions, seed))
            with open(time_file, 'w') as f:
                f.write("{}".format(end_time - start_time))

        if get_exec_acc:
            tgt_file = os.path.join(self.manager.result_folder, 'exec_acc_seed_{}{}.txt'.format(seed, suffix))
            with open(tgt_file, 'w') as f:
                f.write(json.dumps(all_exec_acc, indent=2))

        if get_hall_acc:
            tgt_file = os.path.join(self.manager.result_folder, 'hall_acc_seed_{}{}.txt'.format(seed, suffix))
            with open(tgt_file, 'w') as f:
                f.write(json.dumps(all_hall_acc, indent=2))

    def _get_model_path(self, iter_number):
        model_path = os.path.join(
            self.manager.result_folder,
            "model_{}.pt".format(iter_number))
        return model_path
