import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from q_learning import double_q_learning

def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n

def SavePlot(vanilla_Q_learning, double_Q_learning, metric_names, name, dirname, smooth=False):
    _, metrics_vanilla, _, = vanilla_Q_learning
    _, _, metrics_double, _, = double_Q_learning
    print(metrics_vanilla)
    for i, metric in enumerate(metrics_vanilla):
        x=0
        xs = []
        ys = []
        stds = []
        for coordinate in metric:
            print(coordinate)
            xs.append(x)
            ys.append(coordinate[0])
            stds.append(coordinate[1])
            x+=1

        lower_std = [val - std for val, std in zip(ys, stds)]
        upper_std = [val + std for val, std in zip(ys, stds)]

        if smooth:
            raise NotImplementedError
        else:
            plt.plot(xs,ys,label=metric_names[i]+" vanilla")
            plt.fill_between(xs, lower_std, upper_std, alpha=0.4)
    
    for i, metric in enumerate(metrics_double):
        x=0
        xs = []
        ys = []
        stds = []
        for coordinate in metric:
            xs.append(x)
            ys.append(coordinate[0])
            stds.append(coordinate[1])
            x+=1

        lower_std = [val - std for val, std in zip(ys, stds)]
        upper_std = [val + std for val, std in zip(ys, stds)]

        if smooth:
            raise NotImplementedError
        else:
            plt.plot(xs,ys,label=metric_names[i]+" double")
            plt.fill_between(xs, lower_std, upper_std, alpha=0.4)
    
    Path(dirname).mkdir(parents=True, exist_ok=True)
    plt.legend()
    plt.savefig(dirname + "/" + name + ".png")