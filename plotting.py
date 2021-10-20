import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Rectangle
import torch

from q_learning import double_q_learning


def running_mean(vals, n=1):
    cumvals = np.array(vals).cumsum()
    return (cumvals[n:] - cumvals[:-n]) / n


def SavePlot(vanilla_Q_learning, double_Q_learning, metric_names, name, dirname, config, Q_table, Q_tables_double, env, smooth=False):
    _, metrics_vanilla, _, _, metrics_vanilla_avgperstep = vanilla_Q_learning
    _, _, metrics_double, _, _, metrics_double_avgperstep = double_Q_learning

    metrics_vanilla_mean, metrics_vanilla_std = metrics_vanilla
    metrics_double_mean, metrics_double_std = metrics_double

    for metric_idx, metric_name in enumerate(metric_names):
        # vanilla
        num_iter = config['num_iter']
        xs = np.arange(num_iter)
        ys = metrics_vanilla_mean[metric_idx]
        stds = metrics_vanilla_std[metric_idx]
        lower_std = [val - std for val, std in zip(ys, stds)]
        upper_std = [val + std for val, std in zip(ys, stds)]
        if smooth:
            raise NotImplementedError
        else:
            plt.plot(xs, ys, label="vanilla")
            plt.fill_between(xs, lower_std, upper_std, alpha=0.4)

        # double
        num_iter = config['num_iter']
        xs = np.arange(num_iter)
        ys = metrics_double_mean[metric_idx]
        stds = metrics_double_std[metric_idx]
        lower_std = [val - std for val, std in zip(ys, stds)]
        upper_std = [val + std for val, std in zip(ys, stds)]
        if smooth:
            raise NotImplementedError
        else:
            plt.plot(xs, ys, label="double")
            plt.fill_between(xs, lower_std, upper_std, alpha=0.4)

        # save&wipe
        Path(dirname).mkdir(parents=True, exist_ok=True)
        plt.xlabel("Episode")
        plt.ylabel(metric_name)
        plt.legend()
        plt.savefig(dirname + "/" + name + "_" +
                    metric_names[metric_idx] + ".png")
        plt.clf()

    cols = env.cols
    rows = env.rows
    V_table = np.max(Q_table, axis=1).reshape(rows, cols)
    # print(Q_table)
    vmin = env.final_reward * -1.2
    vmax = env.final_reward
    plt.imshow(V_table[::-1][:], vmin=vmin, vmax=vmax)  # -30)
    plt.title("State values after training Vanilla Q-learning")
    plt.colorbar()
    plt.savefig(dirname + "/" + name + "_" +
                "V_table_heatmap_vanilla" + ".png")
    plt.clf()

    # Construct V-table for Double Q-learning
    Q1 = Q_tables_double["Q1"]
    Q2 = Q_tables_double["Q2"]
    Q_double = (Q1 + Q2) / 2
    V_table = np.max(Q_double, axis=1).reshape(rows, cols)
    vmin = env.final_reward * -1.2
    vmax = env.final_reward
    plt.imshow(V_table[::-1][:], vmin=vmin, vmax=vmax)  # -30)
    plt.title("State values after training Double Q-learning")
    plt.colorbar()
    plt.savefig(dirname + "/" + name + "_" + "V_table_heatmap_double" + ".png")
    plt.clf()

    plt.xlim((0, cols))
    plt.ylim((0, rows))
    plt.title("Greedy policy after training Vanilla Q-learning")
    for i in range(cols):
        for j in range(rows):
            state_number = j*cols+i
            state_color = env.get_state_color(state_number)
            plt.gca().add_patch(Rectangle((i, j), 1, 1, linewidth=1,
                                          edgecolor='r', facecolor=state_color))
            #max_action = np.argmax(Q_table[state_number])
            max_actions,  = np.where(
                Q_table[state_number] == np.max(Q_table[state_number]))
            if state_color in ["white", "blue"]:
                if 1 in max_actions:
                    plt.arrow(i+0.5, j+0.5, 0.45, 0, width=0.04,
                              length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
                if 2 in max_actions:
                    plt.arrow(i+0.5, j+0.5, 0, -0.45, width=0.04,
                              length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
                if 3 in max_actions:
                    plt.arrow(i+0.5, j+0.5, -0.45, 0, width=0.04,
                              length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
                if 0 in max_actions:
                    plt.arrow(i+0.5, j+0.5, 0, 0.45, width=0.04,
                              length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
    # plt.colorbar()

#    mgr = plt.get_current_fig_manager()
#    mgr.window.setGeometry(0,0,800,50)
#    plt.rcParams["figure.figsize"] = (20,3)
    plt.savefig(dirname + "/" + name + "_" +
                "Q_max_action_arrows_vanilla" + ".png")
    plt.clf()

    # Now make policy chart for Double-Q learning
    plt.xlim((0, cols))
    plt.ylim((0, rows))
    plt.title("Greedy policy after training Double Q-learning")
    for i in range(cols):
        for j in range(rows):
            state_number = j*cols+i
            state_color = env.get_state_color(state_number)
            plt.gca().add_patch(Rectangle((i, j), 1, 1, linewidth=1,
                                          edgecolor='r', facecolor=state_color))
            #max_action = np.argmax(Q_table[state_number])
            max_actions,  = np.where(
                Q_double[state_number] == np.max(Q_double[state_number]))
            if state_color in ["white", "blue"]:
                if 1 in max_actions:
                    plt.arrow(i+0.5, j+0.5, 0.45, 0, width=0.04,
                              length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
                if 2 in max_actions:
                    plt.arrow(i+0.5, j+0.5, 0, -0.45, width=0.04,
                              length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
                if 3 in max_actions:
                    plt.arrow(i+0.5, j+0.5, -0.45, 0, width=0.04,
                              length_includes_head=True, color=(0.1, 0.1, 0.1, 1))
                if 0 in max_actions:
                    plt.arrow(i+0.5, j+0.5, 0, 0.45, width=0.04,
                              length_includes_head=True, color=(0.1, 0.1, 0.1, 1))

    plt.savefig(dirname + "/" + name + "_" +
                "Q_max_action_arrows_double" + ".png")
    plt.clf()

    plt.plot(metrics_vanilla_avgperstep, label="vanilla")
    plt.title("Average return per step over the episodes")
    plt.plot(metrics_double_avgperstep, label="double")
    plt.ylabel("Average return per step")
    plt.xlabel("Episode")
    plt.legend()
    plt.savefig(dirname + '/' + name + '_' + 'avgperstep.png')
    # plt.show()
    plt.clf()

# def PlotMap(Q_table,env):
#     V_table = np.max(Q_table,axis=1).reshape(4,12)
#     print(Q_table)
#     plt.imshow(V_table[::-1][:])
#     plt.colorbar()
#     plt.show()
#     plt.savefig(dirname + "/" + name + "_" + "V_table_heatmap" + ".png")
#     plt.clf()

#     plt.xlim((0,12))
#     plt.ylim((0,4))
#     cols = env.cols
#     rows = env.rows
#     for i in range(cols):
#         for j in range(rows):
#             state_number = j*cols+i
#             plt.gca().add_patch(Rectangle((i,j),1,1,linewidth=1,edgecolor='r',facecolor=env.get_state_color(state_number)))
#             max_action = np.argmax(Q_table[state_number])
#             #Q_table[j*12+i,0]/min_Q_value
#             if max_action == 1:
#                 plt.arrow(i+0.5, j+0.5, 0.45, 0, width=0.04,length_includes_head=True, color=(0.1,0.1,0.1,1))
#             elif max_action == 2:
#                 plt.arrow(i+0.5, j+0.5, 0, -0.45, width=0.04,length_includes_head=True, color=(0.1,0.1,0.1,1))
#             elif max_action == 3:
#                 plt.arrow(i+0.5, j+0.5, -0.45, 0, width=0.04,length_includes_head=True, color=(0.1,0.1,0.1,1))
#             elif max_action == 0:
#                 plt.arrow(i+0.5, j+0.5, 0, 0.45, width=0.04,length_includes_head=True, color=(0.1,0.1,0.1,1))
#             else:
#                 print(max_action)
#     # plt.colorbar()
#     plt.show()
#     plt.savefig(dirname + "/" + name + "_" + "V_table_heatmap" + ".png")
#     plt.clf()
