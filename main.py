import numpy as np
from collections import defaultdict

import sys
import time

import yaml
import os
import sys
import time
from datetime import datetime
import shutil
from pathlib import Path
import copy

from run_setup import run_setup
from utils import make_result_directory, seed_everything, SaveResults

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def multi_seed_run(config):

    all_episode_returns_vanilla = []

    for i, seed in enumerate(config['seeds']):
        seed_everything(seed=seed)
        Q, episode_returns, policy_vanilla = run_setup(config,"vanilla")
        all_episode_returns_vanilla.append(episode_returns)

    all_episode_returns_vanilla = np.array(all_episode_returns_vanilla)
    all_episode_returns_vanilla_agg = list(zip(np.mean(all_episode_returns_vanilla, axis=0), np.std(all_episode_returns_vanilla, axis=0)))

    all_episode_returns_double = []

    for i, seed in enumerate(config['seeds']):
        seed_everything(seed=seed)
        Q1, Q2, episode_returns, policy_double = run_setup(config,"double")
        all_episode_returns_double.append(episode_returns)

    all_episode_returns_double = np.array(all_episode_returns_double)
    all_episode_returns_double_agg = list(zip(np.mean(all_episode_returns_double, axis=0), np.std(all_episode_returns_double, axis=0)))


    return (Q, [all_episode_returns_vanilla_agg], policy_vanilla), (Q1, Q2, [all_episode_returns_double_agg], policy_double)


if __name__ == '__main__':

    config_filename = sys.argv[1]
    print(config_filename)

    start = time.time()

    for filename in [config_filename]:
        with open ("configs/"+filename) as f:
            config = yaml.load(f, yaml.SafeLoader)

        dirname = make_result_directory(config, filename)

        print("RUNNING SETUP:")
        print(config)
        config_plotting = copy.deepcopy(config)

        vanilla_Q_learning, double_Q_learning = \
            multi_seed_run(config)
        
        # print(f'AREA UNDER episode_lengths CURVE average: {np.mean(np.array(all_AUC_episode_lengths))}')
        # print(f'AREA UNDER episode_returns CURVE average: {np.mean(np.array(all_AUC_episode_returns))}')

        SaveResults(vanilla_Q_learning, double_Q_learning, ["episode returns"], dirname, config_plotting)

    end = time.time()
    print("FULL TIME SPENT:")
    print(end - start)