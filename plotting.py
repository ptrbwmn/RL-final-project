import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Rectangle
import torch

from q_learning import double_q_learning

def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n

def SavePlot(vanilla_Q_learning, double_Q_learning, metric_names, name, dirname, config, smooth=False):
    _, metrics_vanilla, _, _ = vanilla_Q_learning
    _, _, metrics_double, _, _ = double_Q_learning

    metrics_vanilla_mean, metrics_vanilla_std = metrics_vanilla
    metrics_double_mean, metrics_double_std = metrics_double

    for metric_idx, metric_name in enumerate(metric_names):
        #vanilla
        num_iter = config['num_iter']
        xs=np.arange(num_iter)
        ys=metrics_vanilla_mean[metric_idx]
        stds=metrics_vanilla_std[metric_idx]
        lower_std = [val - std for val, std in zip(ys, stds)]
        upper_std = [val + std for val, std in zip(ys, stds)]
        if smooth:
            raise NotImplementedError
        else:
            plt.plot(xs,ys,label=metric_names[metric_idx]+" vanilla")
            plt.fill_between(xs, lower_std, upper_std, alpha=0.4)

        #double
        num_iter = config['num_iter']
        xs=np.arange(num_iter)
        ys=metrics_double_mean[metric_idx]
        stds=metrics_double_std[metric_idx]
        lower_std = [val - std for val, std in zip(ys, stds)]
        upper_std = [val + std for val, std in zip(ys, stds)]
        if smooth:
            raise NotImplementedError
        else:
            plt.plot(xs,ys,label=metric_names[metric_idx]+" double")
            plt.fill_between(xs, lower_std, upper_std, alpha=0.4)

        #save&wipe
        Path(dirname).mkdir(parents=True, exist_ok=True)
        plt.xlabel("steps")
        plt.ylabel(metric_name)
        plt.legend()
        plt.savefig(dirname + "/" + name + "_" + metric_names[metric_idx] + ".png")
        plt.clf()


def PlotMap(Q_table):
    V_table = np.max(Q_table,axis=1).reshape(4,12)
    min_Q_value = np.min(Q_table)
    print(Q_table)
    plt.imshow(V_table[::-1][:])
    plt.show()
    # V_table = np.zeros((4,4))
    # blank_background = np.ones((5,13))*0.5
    # plt.imshow(blank_background,'winter')
    plt.xlim((-4,13))
    plt.ylim((-4,13))
    # plt.imshow(V_table[::-1][:])
    for i in range(12):
        for j in range(4):
            plt.gca().add_patch(Rectangle((i,j),1,1,linewidth=1,edgecolor='r',facecolor='none'))
            max_action = np.argmax(Q_table[j*12+i])
            #Q_table[j*12+i,0]/min_Q_value
            if max_action == 0:
                plt.arrow(i+0.5, j+0.5, 0.45, 0, width=0.04,length_includes_head=True, color=(0.1,0.1,0.1,1))
            elif max_action == 1:
                plt.arrow(i+0.5, j+0.5, 0, -0.45, width=0.04,length_includes_head=True, color=(0.1,0.1,0.1,1))
            elif max_action == 2:
                plt.arrow(i+0.5, j+0.5, -0.45, 0, width=0.04,length_includes_head=True, color=(0.1,0.1,0.1,1))
            elif max_action == 3:
                plt.arrow(i+0.5, j+0.5, 0, 0.45, width=0.04,length_includes_head=True, color=(0.1,0.1,0.1,1))
            else:
                print(max_action)
    # plt.colorbar()
    plt.show()