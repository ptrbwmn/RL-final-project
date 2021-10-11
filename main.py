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

def multi_seed_run(config):

    # all_episode_lengths = []
    all_episode_returns = []

    # all_AUC_episode_lengths = []
    # all_AUC_episode_returns = []

    for i, seed in enumerate(config['seeds']):
        seed_everything(seed=seed)
        
        Q_table_list, episode_returns, policy = run_setup(config)

        # all_episode_lengths.append(episode_lengths)
        all_episode_returns.append(episode_returns)
        
        # all_AUC_episode_lengths.append(AreaUnderCurve(episode_lengths,config).item())
        # all_AUC_episode_returns.append(AreaUnderCurve(episode_returns,config).item())


    # all_episode_lengths = np.array(all_episode_lengths)
    all_episode_returns = np.array(all_episode_returns)

    # all_episode_lengths_agg = list(zip(np.mean(all_episode_lengths, axis=0), np.std(all_episode_lengths, axis=0)))
    all_episode_returns_agg = list(zip(np.mean(all_episode_returns, axis=0), np.std(all_episode_returns, axis=0)))


    return Q_table_list, all_episode_returns_agg, policy


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

        Q_table_list, episode_returns_agg, policy = \
            multi_seed_run(config)
        
        # print(f'AREA UNDER episode_lengths CURVE average: {np.mean(np.array(all_AUC_episode_lengths))}')
        # print(f'AREA UNDER episode_returns CURVE average: {np.mean(np.array(all_AUC_episode_returns))}')

        SaveResults(Q_table_list, [episode_returns_agg], ["episode returns"], policy, dirname, config_plotting)

    end = time.time()
    print("FULL TIME SPENT:")
    print(end - start)