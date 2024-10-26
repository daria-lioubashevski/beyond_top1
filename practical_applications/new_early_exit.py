import argparse
import pickle
from plots import plot_comparison_between_early_exit_methods
from utils import *
from sklearn.metrics import pairwise_distances


def calculate_accuracy(preds, true_labels):
    return np.mean(preds == true_labels)


def calculate_speedup(total_layers, layer_counts):
    return total_layers / np.sum(layer_counts)


def evalute_various_early_exit_methods(indxs_per_layer, embds_per_layer, probs_per_layer,
                                       num_layers, input_ids, clf):
    input_ids = input_ids[0].numpy()
    labels = input_ids[1:]
    num_examples = len(labels)

    # baseline
    model_preds = indxs_per_layer[-1, :-1, 0]
    accuracy = len(np.where(model_preds == labels)[0]) / num_examples
    total_layers = num_layers * num_examples

    # oracle
    num_layer_where_decision_finalizes = calc_top1_satur_layer(indxs_per_layer, num_layers)
    oracle_total_layers = np.sum(num_layer_where_decision_finalizes)
    oracle_speedup = total_layers / oracle_total_layers

    # clf
    clf_preds = model_preds
    clf_layers = [num_layers] * num_examples
    for i in range(num_layers):
        cur_embds = embds_per_layer[i, :-1, :]
        pred_k = clf.predict(cur_embds)
        satur_inds = np.where(pred_k == 1)[0]
        for ind in satur_inds:
            if clf_layers[ind] == num_layers:
                clf_layers[ind] = i
                clf_preds[ind] = indxs_per_layer[i, ind, 0]

    clf_acc = len(np.where(clf_preds == labels)[0]) / num_examples
    clf_speedup = total_layers / np.sum(clf_layers)

    # softmax - different thresholds
    softmax_accs = []
    softmax_speedups = []
    for thr in SOFTMAX_THRS_FOR_EE:
        softmax_preds = model_preds
        softmax_layers = [num_layers] * num_examples
        for i in range(num_layers):
            cur_probs = probs_per_layer[i, :-1, :2]
            satur_inds = np.where(cur_probs[:, 0] - cur_probs[:, 1] >= thr)[0]
            for ind in satur_inds:
                if softmax_layers[ind] == num_layers:
                    softmax_layers[ind] = i
                    softmax_preds[ind] = indxs_per_layer[i, ind, 0]

        softmax_accs.append(calculate_accuracy(softmax_preds, labels))
        softmax_speedups.append(calculate_speedup(total_layers, softmax_layers))

    # embedding cosine similarity
    cos_sim_accs = []
    cos_sim_speedups = []
    for thr in COS_SIM_THR_FOR_EE:
        cos_sim_preds = model_preds
        cos_sim_layers = [num_layers] * num_examples
        for i in range(1, num_layers):
            cur_embds = embds_per_layer[i, :-1, :]
            prev_embds = embds_per_layer[i - 1, :-1, :]
            dist_matrix = pairwise_distances(cur_embds, prev_embds, metric="cosine")
            cos_sim_arr = 1 - np.diagonal(dist_matrix)
            satur_inds = np.where(cos_sim_arr >= thr)[0]
            for ind in satur_inds:
                if cos_sim_layers[ind] == num_layers:
                    cos_sim_layers[ind] = i
                    cos_sim_preds[ind] = indxs_per_layer[i, ind, 0]
        cos_sim_accs.append(calculate_accuracy(cos_sim_preds, labels))
        cos_sim_speedups.append(calculate_speedup(total_layers, cos_sim_layers))

    acc_and_speedup_dict = {"baseline": [accuracy, 1], "oracle": [accuracy, oracle_speedup],
                            "clf_2": [clf_acc, clf_speedup]}
    for i, thr in enumerate(SOFTMAX_THRS_FOR_EE):
        acc_and_speedup_dict[f"softmax_{thr}"] = [softmax_accs[i], softmax_speedups[i]]
    for j, thr in enumerate(COS_SIM_THR_FOR_EE):
        acc_and_speedup_dict[f"cos_sim_{thr}"] = [cos_sim_accs[j], cos_sim_speedups[j]]
    return acc_and_speedup_dict


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_samples", type=int, help="number of samples")
    parser.add_argument("-c", "--clf_path", type=str, help="path to trained task classifier pickle")
    return parser.parse_args()


def main(args):
    model_name = "gpt2"
    model, tokenizer, processor = load_model(model_name)
    num_layers = model.config.num_hidden_layers
    samples = load_samples(model_name, args.num_samples)
    with open(args.clf_path, 'rb') as fp:
        clf = pickle.load(fp)
    all_acc_and_speedup_dicts = []
    for sample in samples:
        input_ids = tokenize_text(tokenizer, sample)
        indxs_per_layer, embds_per_layer, probs_per_layer = extract_hidden_layers_reps(model_name, model,
                                                                                       tokenizer, processor,
                                                                                       [sample], num_layers,
                                                                                       extract_embds=True,
                                                                                       extract_probs=True)
        if indxs_per_layer is not None:
            acc_and_speedup_dict = evalute_various_early_exit_methods(indxs_per_layer, embds_per_layer,
                                                                      probs_per_layer, num_layers,
                                                                      input_ids, clf)
            all_acc_and_speedup_dicts.append(acc_and_speedup_dict)
    plot_comparison_between_early_exit_methods(all_acc_and_speedup_dicts)


if __name__ == '__main__':
    args = args_parse()
    main(args)
