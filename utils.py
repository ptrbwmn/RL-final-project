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
    dirname = "results/" + str(config['name'])
    Path(dirname).mkdir(parents=True, exist_ok=True)

    # save config file right away, results are added to the folder later
    original = "configs/" + filename
    target = "./" + dirname + "/" + filename
    shutil.copyfile(original, target)

    return dirname


def make_dirname(config, pickle_file_name=False):
    q_learning_variant = config['q_learning_variant']
    policy = config['policy']
    epsilon = config['epsilon']
    num_iter = config['num_iter']
    env = config['env']

    timestamp = datetime.now()
    dirname = "results/" \
    + str(q_learning_variant) \
    + "/" + str(policy) + "/" \
    + str(env) + "/" \
    + str(timestamp).replace(" ", "_").replace(".", "_").replace(":", "-")

    if pickle_file_name:
        print("IN PICKLE FILE NAME")
        #add functionality to abbreviate values, e.g. Random Acquisition -> RA
        dirname = str(q_learning_variant) + "_" + str(policy) + "_" + str(env) \
        + "_" + "numiter" + str(num_iter) \
        + "_" + "epsilon" + str(epsilon) \
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

def SaveResults(Q_table_list, metrics, metric_names, policy, dirname, config):

    results = \
        {"Q_table_list": Q_table_list,
         "metrics": metrics,
         "policy": policy}

    # print("Saving results:")
    # print(results)
    a_file = open(dirname + "/" + "results.pkl", "wb")
    pickle.dump(results, a_file)
    a_file.close()

    name = config['name']

    folder = "result_pickles/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    b_file = open(folder + name + ".pkl", "wb")
    pickle.dump(results, b_file)
    b_file.close()

    # Save YAML file
    results = {key: str(val) for key, val in results.items()}
    with open(dirname + "/" + "results.yaml", 'w') as file:
        yaml.dump(results, file)

    #save plots
    SavePlot(metrics,metric_names,name,dirname,smooth=False)
    





