import matplotlib.pyplot as plt
import numpy as np
from consts import *
import stats


def plot_rank_saturation_correspondence(satur_layer_rank_corr_dict, k=5):
    colors = ['forestgreen', 'cornflowerblue', 'mediumpurple', 'indianred', 'darkorange']
    fig, ax = plt.subplots()
    ax.set_ylabel('saturation layer rank', fontsize=16)
    ax.set_xlabel('token rank', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)

    means = [np.mean(satur_layer_rank_corr_dict[j]) + 1 for j in range(k)]
    stds = [np.std(satur_layer_rank_corr_dict[j]) / np.sqrt(len(satur_layer_rank_corr_dict[j])) for j in range(k)]
    ax.bar([i + 1 for i in range(k)], height=means, yerr=stds, align='center',
           color=colors, alpha=0.5, ecolor='black', capsize=10)
    ax.set_xticks([i + 1 for i in range(k)])
    ax.set_yticks([i + 1 for i in range(k - 1)])

    max_mean = max([means[i] + stds[i] for i in range(len(means))])

    # t-tests
    for j in range(k - 1):
        data1 = np.array(satur_layer_rank_corr_dict[str(j)])
        data2 = np.array(satur_layer_rank_corr_dict[str(j + 1)])
        res = stats.ttest_ind(data1, data2)
        stars = ""
        if res.pvalue < 0.001:
            stars = "***"
        elif res.pvalue < 0.01:
            stars = "**"
        if len(stars) > 0:
            x1, x2 = j + 1 + 0.1, j + 2 - 0.1  # Positions of the boxes
            ax.text((x1 + x2) * .5, max_mean + 0.2, stars, ha='center', va='bottom', color=col)

    ax = plt.gca()
    ax.set_ylim([0, math.ceil(max_mean + 0.2)])
    plt.savefig("satur_rank_corr.png")


def plot_comparison_between_early_exit_methods(all_acc_speedup_dicts):
    baseline = []
    oracle_acc = []
    oracle_speedup = []
    clf_2_acc = []
    clf_2_speedup = []
    softmax_accs = [[] for _ in range(len(SOFTMAX_THRS_FOR_EE))]
    softmax_speedups = [[] for _ in range(len(SOFTMAX_THRS_FOR_EE))]
    cos_sim_accs = [[] for _ in range(len(COS_SIM_THR_FOR_EE))]
    cos_sim_speedups = [[] for _ in range(len(COS_SIM_THR_FOR_EE))]

    for acc_speedup_dict in all_acc_speedup_dicts:
        baseline.append(acc_speedup_dict['baseline'][0])
        oracle_acc.append(acc_speedup_dict['oracle'][0])
        oracle_speedup.append(acc_speedup_dict['oracle'][1])
        clf_2_acc.append(acc_speedup_dict['clf_2'][0])
        clf_2_speedup.append(acc_speedup_dict['clf_2'][1])

        for i, thr in enumerate(SOFTMAX_THRS_FOR_EE):
            softmax_accs[i].append(acc_speedup_dict[f'softmax_{thr}'][0])
            softmax_speedups[i].append(acc_speedup_dict[f'softmax_{thr}'][1])

        for j, thr in enumerate(COS_SIM_THR_FOR_EE):
            cos_sim_accs[j].append(acc_speedup_dict[f'cos_sim_{thr}'][0])
            cos_sim_speedups[j].append(acc_speedup_dict[f'cos_sim_{thr}'][1])

    fig, ax = plt.subplots()
    ax.scatter(1, np.mean(baseline), label="baseline", marker="+")
    ax.scatter(np.mean(oracle_speedup), np.mean(oracle_acc), label="oracle", marker="^")
    ax.scatter(np.mean(clf_2_speedup), np.mean(clf_2_acc), label="ours", color="green", marker="*")
    ax.scatter([np.mean(y) for y in softmax_speedups], [np.mean(x) for x in softmax_accs], label="softmax",
               color="purple")
    ax.plot([np.mean(y) for y in softmax_speedups], [np.mean(x) for x in softmax_accs], color="purple",
            linestyle='dashed',
            alpha=0.5)
    ax.scatter([np.mean(y) for y in cos_sim_speedups], [np.mean(x) for x in cos_sim_accs], label="state", color="red")
    ax.plot([np.mean(y) for y in cos_sim_speedups], [np.mean(x) for x in cos_sim_accs], color="red", linestyle='dashed',
            alpha=0.5)

    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.set_xlabel('Speedup ratio', fontsize=16)
    ax.legend(fontsize=12)
    fig.savefig("early_exit_methods_compar.png")


def plot_interv_results(model_name, layers_df):
    num_adj_layers = NUM_ADJACENT_LAYERS_FOR_INTERV_MAPPING[model_name]
    layers_df['distance_from_satur_layer'] = layers_df['inter_layer'] - layers_df['early_satur_layer'] - 1
    layers_df['new_satur_layer_match_inter_layer'] = np.logical_and(
        layers_df['inter_layer'] == layers_df['new_satur_layer'], layers_df['pred_unchanged']).astype(int)

    grouped_layers_df = layers_df.groupby('distance_from_satur_layer')
    values = grouped_layers_df['new_satur_layer_match_inter_layer'].sum().tolist()
    sizes = grouped_layers_df['new_satur_layer_match_inter_layer'].size().tolist()
    values = np.array([values[i] / sizes[i] * 100 for i in range(2 * num_adj_layers + 1)])

    categories = list(range(-num_adj_layers, num_adj_layers + 1))
    fig, ax = plt.subplots()
    ax.bar(categories, values, color='tab:blue', alpha=0.5)
    step_values = [max(values[:num_adj_layers]) + 3] * num_adj_layers + [max(values[num_adj_layers:]) + 3] * (
            num_adj_layers + 2)
    ax.step(np.array(categories + [num_adj_layers + 1]) - 0.6, step_values, linestyle='--', where='post',
            color='orange')

    # Add title and labels
    ax.set_xlabel('Distance of implanted layer from original saturation layer', fontsize=12)
    ax.set_ylabel('% of examples', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_xticks(categories)
    fig.savefig("interv_results_plot.png")
