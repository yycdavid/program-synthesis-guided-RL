import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import scipy
import scipy.stats
import re
import math


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        print("Creating {}".format(dir_path))
        os.makedirs(dir_path)
    else:
        #raise Exception('Result folder for this experiment already exists')
        pass


def get_args():
    parser = argparse.ArgumentParser(
        description='Analysis script, get the statistics we want')
    parser.add_argument('--mode', type=str, default='training_curve',
                        help='Mode of analysis')
    parser.add_argument(
        '--spec',
        type=str,
        default='spec.json',
        help='Spec file specifying which experiments to analyze')
    parser.add_argument(
        '--exp',
        type=str,
        default='p_ours',
        help='Which experiment to analyze')
    parser.add_argument(
        '--file',
        type=str,
        default='stats_all.json',
        help='Which stats file')
    parser.add_argument(
        '--task',
        type=str,
        default='all',
        help='Which task to inspect')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='box')
    parser.add_argument(
        '--max_iter',
        type=int,
        default=400000,
        help='Max iter to plot training curves')
    parser.add_argument(
        '--use_steps',
        action='store_true',
        default=False,
        help='For training curves. If set, will plot training curve over number of steps of interaction with environment.')
    parser.add_argument(
        '--iter',
        type=int,
        default=-1,
        help='Iter for avg stats')
    parser.add_argument(
        '--merge_seed',
        type=int,
        default=1995,
        help='Seed for merge stats')
    parser.add_argument(
        '--merge_middle',
        type=str,
        default='reuse')
    parser.add_argument(
        '--box',
        action='store_true',
        default=False)
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        help='Number of episodes to offset the training curves')

    return parser.parse_args()


def avg_stats_against_iter(
        stats,
        target_stats,
        target_task='all',
        max_iter=400000):
    if target_task != 'all':
        if not isinstance(target_task, list):
            target_tasks = set([target_task])
        else:
            target_tasks = set(target_task)

    avg_stats = {}
    for iter_stats in stats:
        iter = iter_stats['iter']
        if iter > max_iter:
            continue
        n_rollouts = 0
        total_target = 0
        for task_result in iter_stats['result']:
            if target_task != 'all':
                if not task_result['task'] in target_tasks:
                    continue
            n_rollouts += len(task_result[target_stats])
            total_target += sum(task_result[target_stats])
        avg_target = total_target / n_rollouts
        avg_stats[iter] = avg_target

    iters = sorted(avg_stats.keys())
    avgs = []
    for iter in iters:
        avgs.append(avg_stats[iter])

    return iters, avgs


def plot_training_curve(data, metric_name, save_dir='', use_steps=False, with_err=False):
    fig, ax1 = plt.subplots()
    lines = []
    model_names = []
    for model_name in data.keys():
        if use_steps:
            x_vals = data[model_name]['steps']
            x_vals = [ep / 1000000 for ep in x_vals]
        else:
            x_vals = data[model_name]['iters']
            x_vals = [ep / 1000 for ep in x_vals]
        avg_rewards = data[model_name][metric_name]
        if with_err:
            err_name = 'sterr' + metric_name[3:]
            errs = data[model_name][err_name]
            line = ax1.errorbar(x_vals, avg_rewards, yerr=errs, label=model_name, capsize=2.0)
        else:
            line = ax1.plot(x_vals, avg_rewards, label=model_name)
        lines.append(line[0])
        model_names.append(model_name)

    # ax1.legend()
    ax1.set_ylabel(metric_name)
    if metric_name == 'avg_rewards':
        ax1.set_ylim(0.0, 1.0)
    if use_steps:
        ax1.set_xlabel('steps (*10^6)')
    else:
        ax1.set_xlabel('episodes (*1000)')
    ax1.ticklabel_format(axis='x', style='plain')
    save_file = os.path.join(save_dir, "{}.pdf".format(metric_name))
    plt.savefig(save_file, bbox_inches='tight')

    # Save legend separately
    figlegend = plt.figure(figsize=(3.0, 3.0))
    figlegend.legend(
        lines,
        model_names,
        'center',
        ncol=1,
        fancybox=True,
        shadow=True,
        prop={
            'size': 14})
    save_file = os.path.join(save_dir, "legend.pdf")
    figlegend.savefig(save_file, bbox_inches='tight')

    plt.close()


def print_best_stats(args):
    # Read stats
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    spec_file = os.path.join(base_dir, 'analysis', args.spec)
    with open(spec_file, 'r') as f:
        spec = json.load(f)

    data = {}
    for model_name, exp_name in spec.items():
        # Load stats for exp
        stats_path = os.path.join(results_dir, exp_name)
        with open(stats_path, 'r') as f:
            stats = json.load(f)

        # Get stats
        iters, avg_rewards = avg_stats_against_iter(stats, 'reward')
        _, avg_finish_times = avg_stats_against_iter(stats, 'finish_time')

        best_iter = np.argmax(avg_rewards)
        print(model_name)
        print("Best reward: {}, avg finish time: {}, at iter {}".format(
            avg_rewards[best_iter], avg_finish_times[best_iter], iters[best_iter]))


def training_curve_with_error(args):
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'figure.autolayout': True})

    # Read stats
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    spec_file = os.path.join(base_dir, 'analysis', args.spec)
    with open(spec_file, 'r') as f:
        spec = json.load(f)

    data = {}
    for model_name, setting in spec.items():
        stats_by_seeds = []
        for (i, seed) in enumerate(setting['seeds']):
            exp_folder = os.path.join(results_dir, "{}_seed_{}".format(setting['name'], seed))
            if isinstance(setting['test_seeds'], list):
                test_seed = setting['test_seeds'][i]
            else:
                test_seed = setting['test_seeds']

            stats_path = os.path.join(exp_folder, "stats_all_{}_seed_{}.json".format(setting['middle'], test_seed))
            with open(stats_path, 'r') as f:
                stats = json.load(f)

            # Get stats
            iters, avg_rewards = avg_stats_against_iter(
                stats, 'reward', max_iter=args.max_iter)
            _, avg_finish_times = avg_stats_against_iter(
                stats, 'finish_time', max_iter=args.max_iter)

            stats_one_seed = {}
            for (i, iter) in enumerate(iters):
                stats_one_seed[iter] = {
                    'reward': avg_rewards[i],
                    'finish_time': avg_finish_times[i],
                }
            stats_by_seeds.append(stats_one_seed)

        # Get common iters and filter
        iters_by_seeds = [set(s.keys()) for s in stats_by_seeds]
        common_iters = set.intersection(*iters_by_seeds)

        common_iters = list(common_iters)
        common_iters.sort()
        mean_rewards = []
        sterr_rewards = []
        mean_finish_times = []
        sterr_finish_times = []
        for iter in common_iters:
            rewards = []
            finish_times = []
            for stats_one_seed in stats_by_seeds:
                rewards.append(stats_one_seed[iter]['reward'])
                finish_times.append(stats_one_seed[iter]['finish_time'])
            mean_rewards.append(np.mean(rewards))
            sterr_rewards.append(scipy.stats.sem(rewards))
            mean_finish_times.append(np.mean(finish_times))
            sterr_finish_times.append(scipy.stats.sem(finish_times))

        result_stats = {
            'iters': common_iters,
            'avg_rewards': mean_rewards,
            'sterr_rewards': sterr_rewards,
            'avg_finish_times': mean_finish_times,
            'sterr_finish_times': sterr_finish_times,
        }

        if model_name == "Ours" or model_name == "WM":
            if args.offset > 0:
                result_stats['iters'] = [k + args.offset for k in result_stats['iters']]

        data[model_name] = result_stats
        print(model_name)
        print("Mean reward: {} +- {}".format(mean_rewards[-1], sterr_rewards[-1]))
        print("Mean finish time: {} +- {}".format(mean_finish_times[-1], sterr_finish_times[-1]))

    save_dir = os.path.join(base_dir, "analysis", args.save_dir)
    create_dir(save_dir)

    # Plot reward against episodes
    plot_training_curve(data, 'avg_rewards', save_dir=save_dir, with_err=True)

    # Plot finish time against episodes
    plot_training_curve(data, 'avg_finish_times', save_dir=save_dir, with_err=True)


def ablations(args):
    # Read stats
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    spec_file = os.path.join(base_dir, 'analysis', args.spec)
    with open(spec_file, 'r') as f:
        spec = json.load(f)

    data = {}
    for model_name, setting in spec.items():
        stats_by_seeds = []
        if model_name == 'Optim' or model_name == 'Random' or model_name == "RandExpl":
            iter_name = '400000'
        else:
            iter_name = 'all'
        for (i, seed) in enumerate(setting['seeds']):
            exp_folder = os.path.join(results_dir, "{}_seed_{}".format(setting['name'], seed))
            if isinstance(setting['test_seeds'], list):
                test_seed = setting['test_seeds'][i]
            else:
                test_seed = setting['test_seeds']

            stats_path = os.path.join(exp_folder, "stats_{}_{}_seed_{}.json".format(iter_name, setting['middle'], test_seed))
            with open(stats_path, 'r') as f:
                stats = json.load(f)

            # Get stats
            iters, avg_rewards = avg_stats_against_iter(
                stats, 'reward', max_iter=args.max_iter)
            _, avg_finish_times = avg_stats_against_iter(
                stats, 'finish_time', max_iter=args.max_iter)

            stats_one_seed = {}
            for (i, iter) in enumerate(iters):
                stats_one_seed[iter] = {
                    'reward': avg_rewards[i],
                    'finish_time': avg_finish_times[i],
                }
            stats_by_seeds.append(stats_one_seed)

        # Get common iters and filter
        iters_by_seeds = [set(s.keys()) for s in stats_by_seeds]
        common_iters = set.intersection(*iters_by_seeds)

        common_iters = list(common_iters)
        common_iters.sort()
        mean_rewards = []
        sterr_rewards = []
        mean_finish_times = []
        sterr_finish_times = []
        for iter in common_iters:
            rewards = []
            finish_times = []
            for stats_one_seed in stats_by_seeds:
                rewards.append(stats_one_seed[iter]['reward'])
                finish_times.append(stats_one_seed[iter]['finish_time'])
            mean_rewards.append(np.mean(rewards))
            sterr_rewards.append(scipy.stats.sem(rewards))
            mean_finish_times.append(np.mean(finish_times))
            sterr_finish_times.append(scipy.stats.sem(finish_times))
        result_stats = {
            'iters': common_iters,
            'avg_rewards': mean_rewards,
            'sterr_rewards': sterr_rewards,
            'avg_finish_times': mean_finish_times,
            'sterr_finish_times': sterr_finish_times,
        }

        print(model_name)
        print("Mean reward: {} +- {}".format(mean_rewards[-1], sterr_rewards[-1]))
        print("Mean finish time: {} +- {}".format(mean_finish_times[-1], sterr_finish_times[-1]))


def final_iter_avg_stats(stats_path, args, target_task='all', iter='final'):
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    # Get stats to plot
    iters, avg_rewards = avg_stats_against_iter(
        stats, 'reward', target_task=target_task, max_iter=args.max_iter)
    _, avg_finish_times = avg_stats_against_iter(
        stats, 'finish_time', target_task=target_task, max_iter=args.max_iter)

    if iter == 'final':
        return avg_rewards[-1], avg_finish_times[-1]
    else:
        for (i, it) in enumerate(iters):
            if it == iter:
                return avg_rewards[i], avg_finish_times[i]


def avg_stats(args):
    # Read stats
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    stats_path = os.path.join(results_dir, args.exp, args.file)
    if args.iter == -1:
        iter = 'final'
    else:
        iter = args.iter
    avg_reward, avg_finish_time = final_iter_avg_stats(stats_path, args, args.task, iter)
    print("avg reward: {}".format(avg_reward))
    print("avg finish time: {}".format(avg_finish_time))


def get_target_task_avg(stats_path, target_tasks):
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    iters, avg_rewards = avg_stats_against_iter(
        stats, 'reward', target_task=target_tasks)
    _, avg_finish_times = avg_stats_against_iter(
        stats, 'finish_time', target_task=target_tasks)
    return avg_rewards[0], avg_finish_times[0]


def compare_n_comletions(args):
    # Read stats
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")

    Ns = [1, 2, 3, 4, 5]
    test_seeds = [1, 2, 3, 4, 5]
    time_seeds = [6, 7]
    box = args.box
    mean_rewards = []
    var_rewards = []
    mean_finish_times = []
    var_finish_times = []
    runtimes = []
    for n_completions in Ns:
        reward = []
        finish_t = []
        for seed in test_seeds:
            if box:
                file_name = 'stats_200000_reuse_N_{}_seed_{}.json'.format(n_completions, seed)
            else:
                file_name = 'stats_400000_N_{}_seed_{}.json'.format(n_completions, seed)
            stats_path = os.path.join(results_dir, args.exp, file_name)
            avg_reward, avg_finish_time = get_target_task_avg(
                stats_path, 'all')
            reward.append(avg_reward)
            finish_t.append(avg_finish_time)

        mean_rewards.append(np.mean(reward))
        var_rewards.append(np.var(reward))
        mean_finish_times.append(np.mean(finish_t))
        var_finish_times.append(np.var(finish_t))

        print(n_completions)
        print("Mean reward: {} +- {}".format(np.mean(reward), scipy.stats.sem(reward)))
        print("Mean finish time: {} +- {}".format(np.mean(finish_t),scipy.stats.sem(finish_t)))

        run_t = []
        for seed in time_seeds:
            time_file = os.path.join(results_dir, args.exp, 'time_N_{}_seed_{}.txt'.format(n_completions, seed))
            with open(time_file, 'r') as f:
                tt = f.read()
                run_t.append(float(tt))
        runtimes.append(np.mean(run_t))
        print("Mean runtime: {}".format(np.mean(run_t)))

    # Plot
    plt.rcParams.update({'font.size': 26})
    plt.rcParams.update({'figure.autolayout': True})

    '''
    Mean and var in separate lines
    '''
    def plot_mean_var(n_samples, mean_v, var_v, name):
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Number of samples')
        ax1.set_xticks(n_samples)
        ax1.set_ylabel('Mean', color=color)
        ax1.plot(Ns, mean_v, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()

        color = 'tab:red'
        if name == "reward":
            ax2.set_ylabel('Variance (1e-3)', color=color)
            var_v = [v * 1000 for v in var_v]
            ax2.plot(Ns, var_v, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        else:
            ax2.set_ylabel('Variance', color=color)
            ax2.plot(Ns, var_v, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        save_file = os.path.join(base_dir, "analysis", "{}_N.pdf".format(name))
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()

    '''
    Var as error bar
    '''
    def plot_mean_var_2(n_samples, mean_v, var_v, name):
        fig, ax1 = plt.subplots()
        std_v = [math.sqrt(v) for v in var_v]
        line = ax1.errorbar(n_samples, mean_v, yerr=std_v, ecolor='r', capsize=3.0)
        ax1.set_xlabel('Number of samples')
        ax1.set_xticks(n_samples)
        if name == "rewards":
            ylabel = "avg reward"
        else:
            ylabel = "avg finish time"
        ax1.set_ylabel(ylabel)
        fig.tight_layout()
        save_file = os.path.join(base_dir, "analysis", "{}_N_bar.pdf".format(name))
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()

    plot_mean_var(Ns, mean_rewards, var_rewards, "reward")
    plot_mean_var(Ns, mean_finish_times, var_finish_times, "finish_t")

    plt.figure()
    plt.plot(Ns, runtimes)
    plt.xticks(Ns)
    plt.xlabel("Number of samples")
    plt.ylabel("Runtime (s)")
    save_file = os.path.join(base_dir, "analysis", "runtime_N.pdf")
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()


def compare_oracle(args):
    # Read stats
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")

    reward = []
    reward_oracle = []
    finish_t = []
    finish_t_oracle = []
    for seed in range(5):
        stats_path = os.path.join(
            results_dir,
            args.exp,
            'stats_400000_seed_{}.json'.format(seed))
        avg_reward, avg_finish_time = final_iter_avg_stats(
            stats_path, args.task)
        reward.append(avg_reward)
        finish_t.append(avg_finish_time)

        stats_path = os.path.join(
            results_dir,
            args.exp,
            'stats_400000_oracle_seed_{}.json'.format(seed))
        avg_reward, avg_finish_time = final_iter_avg_stats(
            stats_path, args.task)
        reward_oracle.append(avg_reward)
        finish_t_oracle.append(avg_finish_time)

    print("Normal")
    print("Mean reward: {} +- {}".format(np.mean(reward), np.std(reward)))
    print("Mean finish time: {} +- {}".format(np.mean(finish_t), np.std(finish_t)))
    print("Oracle")
    print("Mean reward: {} +- {}".format(np.mean(reward_oracle), np.std(reward_oracle)))
    print("Mean finish time: {} +- {}".format(np.mean(finish_t_oracle),
                                              np.std(finish_t_oracle)))


def merge_stats(args):
    # Read stats
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results", args.exp)
    stats_dir = os.path.join(results_dir, "stats")

    save_points = set()
    for file in os.listdir(stats_dir):
        if file.endswith(".json") and file.startswith("stats"):
            words = re.split(r'_|\.', file)
            save_points.add(int(words[1]))

    results_together = []
    save_points = list(save_points)
    save_points.sort()
    for iter_number in save_points:
        stats_path = os.path.join(
            stats_dir, "stats_{}_{}_seed_{}.json".format(iter_number, args.merge_middle, args.merge_seed))
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        results_together.append(stats)

    #next_file = os.path.join(results_dir, "stats_120000_208000_reuse_N_3_seed_10.json")
    # with open(next_file, 'r') as f:
    #    stats = json.load(f)
    # results_together.extend(stats)

    save_file = os.path.join(results_dir, "stats_all_{}_seed_{}.json".format(args.merge_middle, args.merge_seed))
    with open(save_file, 'w') as f:
        f.write(json.dumps(results_together, indent=2))


def random_seed_results(args):
    # Read locations
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    spec_file = os.path.join(base_dir, 'analysis', args.spec)
    with open(spec_file, 'r') as f:
        spec = json.load(f)

    # For each, get stats and print
    for model_name, settings in spec.items():
        reward = []
        finish_t = []
        for seed in settings['test_seeds']:
            exp_name = settings['name']
            stats_file_name = "stats_{}_{}_seed_{}.json".format(settings['iter'], settings['middle'], seed)
            # Load stats for exp
            stats_path = os.path.join(results_dir, exp_name, stats_file_name)
            #avg_reward, avg_finish_time = final_iter_avg_stats(stats_path, args.task, iter=settings['iter'])
            avg_reward, avg_finish_time = final_iter_avg_stats(stats_path, args.task)
            reward.append(avg_reward)
            finish_t.append(avg_finish_time)

        print(model_name)
        print("Mean reward: {} +- {}".format(np.mean(reward), scipy.stats.sem(reward)))
        print("Mean finish time: {} +- {}".format(np.mean(finish_t), scipy.stats.sem(finish_t)))

def exec_acc(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    stats_file = os.path.join(results_dir, args.file)
    with open(stats_file, 'r') as f:
        results = json.load(f)
    accs = results[0]['result']
    cnts = 0
    succ = 0
    for c in accs:
        cnts += accs[c][1]
        succ += accs[c][0]

    print("Success/total: {}/{}, acc {}".format(succ, cnts, succ/cnts))

def hall_acc(args):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    stats_file = os.path.join(results_dir, args.file)
    with open(stats_file, 'r') as f:
        results = json.load(f)
    accs = results[0]['result']
    def get_numbers(cnts):
        return cnts[0], cnts[1], cnts[0]/cnts[1]
    for step in accs:
        print("Step: {}".format(step))
        n_same, n_total, rate = get_numbers(accs[step]['whole'])
        print("Whole: {}/{}, acc {}".format(n_same, n_total, rate))
        n_same, n_total, rate = get_numbers(accs[step]['individual'])
        print("Individual: {}/{}, acc {}".format(n_same, n_total, rate))

def main():
    # Parse arguments
    args = get_args()
    if args.mode == 'training_curve':
        # Line plot of training curves (reward, finish time) between ours and
        # baselines
        training_curve_with_error(args)

    elif args.mode == 'compare_oracle':
        compare_oracle(args)

    elif args.mode == 'avg_stats':
        avg_stats(args)

    elif args.mode == 'best_stats':
        print_best_stats(args)

    elif args.mode == 'abla':
        ablations(args)

    elif args.mode == 'n_comp':
        compare_n_comletions(args)

    elif args.mode == 'merge_stats':
        merge_stats(args)

    elif args.mode == 'seeds':
        random_seed_results(args)

    elif args.mode == 'exec':
        exec_acc(args)

    elif args.mode == 'hall':
        hall_acc(args)

    else:
        assert False, "Mode not supported"

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)
        import pdb
        pdb.post_mortem()
