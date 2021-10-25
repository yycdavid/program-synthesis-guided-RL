import environment
import models
import trainers

import yaml
import os
import sys
import argparse
from misc.util import Struct, OutputManager, create_dir
import numpy as np
from synthesizer.solver import CraftSynthesizer
from synthesizer.box_solver import BoxSynthesizer
from models.cvae import CVAE
from random_hallucinator import RandomHallucinator
import torch
import json


def get_args():
    parser = argparse.ArgumentParser(description='Main experiment script')
    parser.add_argument('--seed', type=int, default=1995, help='random seed, default is 1995, do not change default')
    parser.add_argument('--train_seed', type=int, default=1995, help='random seed used in training. This is just to get the training folder during test time')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Config file')
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='Running mode')
    parser.add_argument(
        '--cvae_path',
        type=str,
        default='net_cvae_abstract.pth',
        help='Path for storing the trained CVAE network')
    parser.add_argument(
        '--test_iter',
        type=str,
        default='all',
        help='Test the model saved at which iteration')
    parser.add_argument(
        '--number_of_saves',
        type=int,
        default=50,
        help='Number of save points for training')
    parser.add_argument(
        '--oracle',
        action='store_true',
        default=False,
        help='Use ground truth plan for testing')
    parser.add_argument(
        '--optim',
        action='store_true',
        default=False,
        help='Use optimistic synthesis')
    parser.add_argument(
        '--randexpl',
        action='store_true',
        default=False,
        help='Use random explore then plan')
    parser.add_argument(
        '--viz_name',
        type=str,
        default='viz',
        help='Folder name to store the visualizations, for test mode')
    parser.add_argument(
        '--test_task',
        type=str,
        default='all',
        help='Task for test mode')
    parser.add_argument(
        '--n_completions',
        type=int,
        default=3,
        help='Number of completions used in MAXSAT')
    parser.add_argument(
        '--no_reuse_key',
        action='store_true',
        default=False,
        help='Allow reusing keys in box environment')
    parser.add_argument(
        '--train_reuse_key',
        action='store_true',
        default=False,
        help='Allow reusing keys in box environment during training')
    parser.add_argument(
        '--index_from_sampled',
        type=int,
        default=-1,
        help='Index of pre-sampled scenario to test on')
    parser.add_argument(
        '--save_separate',
        action='store_true',
        default=False,
        help='Save stats separately when test stats')
    parser.add_argument(
        '--save_time',
        action='store_true',
        default=False)
    parser.add_argument(
        '--get_exec_acc',
        action='store_true',
        default=False)
    parser.add_argument(
        '--get_hall_acc',
        action='store_true',
        default=False)
    parser.add_argument('--dhid', type=int, default=200,
                        help='Hidden dim of CVAE')
    parser.add_argument('--drep', type=int, default=100,
                        help='Rep dim of CVAE')
    parser.add_argument(
        '--suffix',
        type=str,
        default='',
        help='Suffix for stats file')
    parser.add_argument(
        '--use_random_hallucinator',
        action='store_true',
        default=False,
        help='Use random hallucinator')
    parser.add_argument(
        '--n_replan',
        type=int,
        default=-1,
        help='Number of steps between each replan')
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


def get_model(is_cuda, dinp=83, dcond=181, dhid=200, drep=100):
    model = CVAE(dinp, dhid, drep, dinp, dcond, is_cuda)
    return model


def inspect(args):
    # Load config
    config = configure(args.config)

    if args.n_replan != -1:
        config.trainer.n_replan = args.n_replan

    # Create results directory and logging
    if args.train_seed != 1995:
        results_path = config.results_dir + config.name + '_seed_{}'.format(args.train_seed)
    else:
        results_path = config.results_dir + config.name
    create_dir(results_path)
    manager = OutputManager(results_path,
    filename='log_inspect.txt')
    manager.say("Log starts, exp: {}".format(config.name))

    # Get environment, model
    exp_env = environment.load(config, args.seed)
    model = models.load(config)
    model.eval()
    trainer = trainers.load(config, exp_env, manager)
    # Load generative model
    if config.world.name == "BoxWorld":
        dinp = 462
        dcond = 577
        if args.use_random_hallucinator:
            generative_model = RandomHallucinator(config.world.name, exp_env)
        else:
            generative_model = get_model(
                False,
                dinp=dinp,
                dcond=dcond,
                dhid=args.dhid,
                drep=args.drep)
    else:
        if args.use_random_hallucinator:
            generative_model = RandomHallucinator(config.world.name, exp_env)
        else:
            generative_model = get_model(
                False,
                dinp=exp_env.ABS_SIZE,
                dcond=exp_env.ABS_PARTIAL_SIZE)

    if not args.use_random_hallucinator:
        generative_model.load_state_dict(
            torch.load(args.cvae_path, map_location="cuda"))
        generative_model.eval()

    # Setup pygame
    spec_file = os.path.join(config.icons, 'spec.json')
    with open(spec_file) as json_file:
        object_param_list = json.load(json_file)
    exp_env.setup_pygame(config.icons, object_param_list)

    trainer.inspect_replan(
        model,
        generative_model,
        model_to_test=args.test_iter,
        viz_name=args.viz_name,
        optimistic=args.optim,
        test_task=args.test_task)


def test(args, stats=False):
    # Load config
    config = configure(args.config)

    # Create results directory and logging
    if args.train_seed != 1995:
        results_path = config.results_dir + config.name + '_seed_{}'.format(args.train_seed)
    else:
        results_path = config.results_dir + config.name
    create_dir(results_path)
    manager = OutputManager(results_path,
    filename='log_test.txt')
    manager.say("Log starts, exp: {}".format(config.name))

    # Get environment, model
    exp_env = environment.load(config, args.seed)
    model = models.load(config)
    model.eval()
    trainer = trainers.load(config, exp_env, manager)
    # Load generative model
    if config.world.name == "BoxWorld":
        dinp = 462
        dcond = 577
        if args.use_random_hallucinator:
            generative_model = RandomHallucinator(config.world.name, exp_env)
        else:
            generative_model = get_model(
                False,
                dinp=dinp,
                dcond=dcond,
                dhid=args.dhid,
                drep=args.drep)
    else:
        if args.use_random_hallucinator:
            generative_model = RandomHallucinator(config.world.name, exp_env)
        else:
            generative_model = get_model(
                False,
                dinp=exp_env.ABS_SIZE,
                dcond=exp_env.ABS_PARTIAL_SIZE)

    if not args.use_random_hallucinator:
        generative_model.load_state_dict(
            torch.load(args.cvae_path, map_location="cuda"))
        generative_model.eval()
    if stats:
        trainer.test_statistics(
            model,
            generative_model,
            models_to_test=args.test_iter,
            use_oracle=args.oracle,
            seed=args.seed,
            optimistic=args.optim,
            rand_explore=args.randexpl,
            n_completions=args.n_completions,
            reuse_key=args.reuse_key,
            save_separate=args.save_separate,
            suffix=args.suffix,
            save_time=args.save_time,
            get_exec_acc=args.get_exec_acc,
            get_hall_acc=args.get_hall_acc,
            update_test_setting=args.update_test_setting)
    else:
        if config.world.name != "BoxWorld":
            # Setup pygame
            spec_file = os.path.join(config.icons, 'spec.json')
            with open(spec_file) as json_file:
                object_param_list = json.load(json_file)
            exp_env.setup_pygame(config.icons, object_param_list)

            trainer.test(
                model,
                generative_model,
                viz_name=args.viz_name,
                use_oracle=args.oracle,
                optimistic=args.optim,
                rand_explore=args.randexpl,
                index_from_sampled=args.index_from_sampled,
                seed=args.seed,
                test_task=args.test_task,
                update_test_setting=args.update_test_setting)
        else:
            trainer.test(
                model,
                generative_model,
                viz_name=args.viz_name,
                use_oracle=args.oracle,
                reuse_key=args.reuse_key,
                index_from_sampled=args.index_from_sampled,
                seed=args.seed,
                test_task=args.test_task,
                update_test_setting=args.update_test_setting)


def train(args):
    # Load config
    config = configure(args.config)

    # Create results directory and logging
    if args.seed != 1995:
        results_path = config.results_dir + config.name + '_seed_{}'.format(args.seed)
    else:
        results_path = config.results_dir + config.name
    create_dir(results_path)
    manager = OutputManager(results_path)
    manager.say("Log starts, exp: {}".format(config.name))

    # Get environment, model
    exp_env = environment.load(config, args.seed)
    model = models.load(config)
    trainer = trainers.load(config, exp_env, manager)
    trainer.train(model, number_of_saves=args.number_of_saves, reuse_key=args.train_reuse_key)


if __name__ == '__main__':
    try:
        args = get_args()
        if args.mode == "train":
            train(args)
        elif args.mode == "test":
            test(args)
        elif args.mode == "test_stats":
            test(args, stats=True)
        elif args.mode == "inspect":
            inspect(args)
        else:
            print("Running mode not supported")

    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
