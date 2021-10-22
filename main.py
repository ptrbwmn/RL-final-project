import numpy as np

import yaml
import sys
import time
import copy

from run_setup import run_setup
from utils import make_result_directory, seed_everything, SaveResults

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def multi_seed_run(config):
    #    config['seeds']=[i for i in range(10)]
    num_seeds = config['seeds']
    config['seeds'] = np.arange(num_seeds)
    num_iter = config['num_iter']
    num_metrics = len(config['metric_names'])

    # maak een zeros array van num_iter,num_metrics,num_seeds
    # vul die elke seed op de juiste seed index
    # neem dan mean over de seed dimensie voor de (num_iter,num_metrics)
    # en in een andere de std over seed dimensie ook resulterende in (num_iter,num_metrics)
    #
    metrics_vanilla_all = np.zeros((num_seeds, num_metrics, num_iter))
    for i, seed in enumerate(config['seeds']):
        seed_everything(seed=seed)
        Q, metrics_vanilla, policy_vanilla, Q_tables_vanilla, env = run_setup(
            config, "vanilla")
        metrics_vanilla_all[i] = metrics_vanilla

    metrics_double_all = np.zeros((num_seeds, num_metrics, num_iter))
    for i, seed in enumerate(config['seeds']):
        seed_everything(seed=seed)
        Q1, Q2, metrics_double, policy_double, Q_tables_double, env = run_setup(
            config, "double")
        metrics_double_all[i] = metrics_double

    metrics_vanilla_mean = np.mean(metrics_vanilla_all, axis=0)
    metrics_vanilla_std = np.std(metrics_vanilla_all, axis=0)

    metrics_vanilla_episode_returns = metrics_vanilla_all[:, 0, :]
    metrics_vanilla_episode_lengths = metrics_vanilla_all[:, 1, :]
    metrics_vanilla_avgperstep = np.sum(
        metrics_vanilla_episode_returns, axis=0)/np.sum(metrics_vanilla_episode_lengths, axis=0)

    metrics_double_mean = np.mean(metrics_double_all, axis=0)
    metrics_double_std = np.std(metrics_double_all, axis=0)

    metrics_double_episode_returns = metrics_double_all[:, 0, :]
    metrics_double_episode_lengths = metrics_double_all[:, 1, :]
    metrics_double_avgperstep = np.sum(
        metrics_double_episode_returns, axis=0)/np.sum(metrics_double_episode_lengths, axis=0)

    metrics_vanilla = [metrics_vanilla_mean, metrics_vanilla_std]
    metrics_double = [metrics_double_mean, metrics_double_std]

    last_Q = Q
    last_policy_vanilla = policy_vanilla
    last_Q1 = Q1
    last_Q2 = Q2
    last_policy_double = policy_double
    last_Q_tables_vanilla = Q_tables_vanilla
    last_Q_tables_double = Q_tables_double
    return (last_Q, metrics_vanilla, last_policy_vanilla, last_Q_tables_vanilla, metrics_vanilla_avgperstep), (last_Q1, last_Q2, metrics_double, last_policy_double, last_Q_tables_double, metrics_double_avgperstep), env


if __name__ == '__main__':

    config_filename = sys.argv[1]
    print(config_filename)

    start = time.time()

    for filename in [config_filename]:
        with open("configs/"+filename) as f:
            config = yaml.load(f, yaml.SafeLoader)

        dirname = make_result_directory(config, filename)

        print("RUNNING SETUP:")
        print(config)
        config_plotting = copy.deepcopy(config)

        vanilla_Q_learning, double_Q_learning, env = \
            multi_seed_run(config)

        print("SAVING RESULTS")
        SaveResults(vanilla_Q_learning, double_Q_learning,
                    config_plotting['metric_names'], dirname, config_plotting, env)

    end = time.time()
    print("FULL TIME SPENT:")
    print(end - start)
