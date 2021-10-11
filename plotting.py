import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n

def SavePlot(metrics, metric_names, name, dirname, smooth=False):
    for i, metric in enumerate(metrics):
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
            plt.plot(xs,ys,label=metric_names[i])
            plt.fill_between(xs, lower_std, upper_std, alpha=0.4)
    
    Path(dirname).mkdir(parents=True, exist_ok=True)
    plt.legend()
    plt.savefig(dirname + "/" + name + ".png")