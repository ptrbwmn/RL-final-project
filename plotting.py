import matplotlib.pyplot as plt
import numpy as np

def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n

def plot(episode_lengths, episode_returns, name):
    n = 50
    # We will help you with plotting this time
    plt.plot(running_mean(episode_lengths,n))
    plt.title('Episode lengths ' + name)
    plt.show()
    plt.plot(running_mean(episode_returns,n))
    plt.title('Episode returns ' + name)
    plt.show()