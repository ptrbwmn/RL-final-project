import os
from datetime import datetime
import torch
import shutil
from pathlib import Path
import random
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import torch

import os
import pickle
import yaml

from plotting import SavePlot


def make_result_directory(config, filename):
    ######## Create folder for results ########
    dirname = make_dirname(config)
    # dirname = "results/" + str(config['name'])
    Path(dirname).mkdir(parents=True, exist_ok=True)

    # save config file right away, results are added to the folder later
    original = "configs/" + filename
    target = "./" + dirname + "/" + filename
    shutil.copyfile(original, target)
    print(dirname)

    return dirname


def make_dirname(config, pickle_file_name=False):
    policy = config['policy']
    epsilon = config['epsilon_0']
    epsilon_decay = config['epsilon_decay']
    alpha_decay = config['alpha_decay']
    gamma = config['gamma']
    num_iter = config['num_iter']
    env = config['env']

    timestamp = datetime.now()
    dirname = "results/" \
    + str(policy) + "/" \
    + str(env) + "/" \
    + str(timestamp).replace(" ", "_").replace(".", "_").replace(":", "-")

    if pickle_file_name:
        print("IN PICKLE FILE NAME")
        #add functionality to abbreviate values, e.g. Random Acquisition -> RA
        dirname = str(policy) + "_" + str(env) \
        + "_" + "numiter" + str(num_iter) \
        + "_" + "epsilon0_" + str(epsilon) \
        + "_" + "epsilondecay_" + str(epsilon_decay) \
        + "_" + "alphadecay_" + str(alpha_decay) \
        + "_" + "gamma_" + str(gamma) \
        + "." + str(timestamp).replace(" ", "_").replace(".", "_").replace(":", "-")

    return dirname


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # if you are using multi-GPU.
    np.random.seed(seed)
    # Numpy module.
    random.seed(seed)
    # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def SaveResults(vanilla_Q_learning, double_Q_learning, metric_names, dirname, config, env):
    last_Q, metrics_vanilla, last_policy_vanilla, last_Q_tables_vanilla = vanilla_Q_learning
    last_Q1, last_Q2, metrics_double, last_policy_double, last_Q_tables_double = double_Q_learning
    results = \
        {"vanilla_Q_learning": vanilla_Q_learning,
         "double_Q_learning": double_Q_learning,
         "last_Q": last_Q,
         "metrics_vanilla": metrics_vanilla,
         "last_policy_vanilla": last_policy_vanilla,
         "last_Q1": last_Q1,
         "last_Q2": last_Q2,
         "metrics_double": metrics_double,
         "last_policy_double": last_policy_double,
         "metric_names": metric_names,
        #  "last_Q_tables_vanilla": last_Q_tables_vanilla,
        #  "last_Q_tables_double": last_Q_tables_double,
         "env": env
         }

    # print("Saving results:")
    # print(results)
    a_file = open(dirname + "/" + "results.pkl", "wb")
    pickle.dump(results, a_file)
    a_file.close()

    name = config['name']

    folder = "result_pickles/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    result_pickle_name = make_dirname(config,True)
    b_file = open(folder + result_pickle_name + ".pkl", "wb")
    pickle.dump(results, b_file)
    b_file.close()

   #save plots
    SavePlot(vanilla_Q_learning, double_Q_learning, metric_names,name,dirname,config,last_Q_tables_vanilla[len(last_Q_tables_vanilla)-1],env,smooth=False)

    # Save YAML file
 #   results = {key: str(val) for key, val in results.items()}
 #   with open(dirname + "/" + "results.yaml", 'w') as file:
 #       yaml.dump(results, file)

     





