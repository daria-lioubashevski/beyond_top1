import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from consts import KENDALLS_TAU_NUM_PERMUTATIONS
from plots import plot_rank_saturation_correspondence
from utils import load_model, load_samples, extract_hidden_layers_reps, calc_top_k_saturation_layers, write_results


def calc_rank_from_satur_layers_arr(saturation_layer_arr, num_layers, k=5):
    satur_layer_rank_corr_dict = defaultdict(list)
    for ind in range(saturation_layer_arr.shape[1]):
        sorted_satur_layers = np.sort(saturation_layer_arr[:, ind]).tolist()
        for j in range(k):
            if saturation_layer_arr[j, ind] < num_layers - 1:
                cur_ind = sorted_satur_layers.index(saturation_layer_arr[j, ind])
                satur_layer_rank_corr_dict[j].append(cur_ind)
    return satur_layer_rank_corr_dict


def is_concordant_strict(pair, ls1, ls2):
    """
    Given a pair of indices, check if ls1 and ls2 agree on their ranking.
    """
    i1, i2 = pair
    return ((ls1[i1] > ls1[i2]) and (ls2[i1] > ls2[i2])) or \
           ((ls1[i1] < ls1[i2]) and (ls2[i1] < ls2[i2])) or \
           ((ls1[i1] == ls1[i2]) and (ls2[i1] == ls2[i2]))


def ktau_strict(ls1, ls2):
    """
    A version of Kendall's tau where ties can be counted as disagreements.
    """
    assert (len(ls1) == len(ls2))
    num_of_items = len(ls1)
    all_pairs = [(i, j)
                 for i in range(num_of_items)
                 for j in range(i + 1, num_of_items)]
    num_of_pairs = len(all_pairs)

    conc = sum([is_concordant_strict(pair, ls1, ls2) for pair in all_pairs])
    disc = num_of_pairs - conc
    tau = (conc - disc) / num_of_pairs
    return tau


def create_matching_monot_seq(sat_list, num_layers):
    last_layer_ind = np.where(sat_list != num_layers - 1)[0][-1]
    return list(range(last_layer_ind + 1))


def calc_mean_kendalls_tau(saturation_layer_arr, num_layers, perm=False):
    kt_distances = []
    num_examples = np.shape(saturation_layer_arr)[1]
    for ind in range(num_examples):
        sat_list = saturation_layer_arr[:, ind]
        monotone_seq = create_matching_monot_seq(sat_list, num_layers)
        if perm:
            sat_list = np.random.permutation(sat_list)
        cur_k = len(monotone_seq)
        if cur_k >= 3:
            tau = ktau_strict(monotone_seq, sat_list[:cur_k])
            if np.isnan(tau):
                kt_distances.append(0)
            else:
                kt_distances.append(tau)
    kt_distances = np.array(kt_distances)
    return np.mean(kt_distances)


def run_permutation_test(saturation_layer_arr, num_layers, actual_tau):
    mean_taus = []
    for j in range(KENDALLS_TAU_NUM_PERMUTATIONS):
        mean_taus.append(calc_mean_kendalls_tau(saturation_layer_arr, num_layers, perm=True))
    mean_taus = np.array(mean_taus)
    if max(mean_taus) < actual_tau:
        return "p < 0.001"
    else:
        return f"p = {len(np.where(mean_taus >= actual_tau)[0]) / len(mean_taus)}"


def calc_strict_kendalls_tau_w_permutation_test(saturation_layer_arr, num_layers, output_path):
    actual_tau = calc_mean_kendalls_tau(saturation_layer_arr, num_layers)
    kendall_tau_result = f"stricter kendall's tau value: {round(actual_tau, 3)}"
    perm_test_result = run_permutation_test(saturation_layer_arr, num_layers, actual_tau)
    results = [kendall_tau_result, perm_test_result]
    write_results(results, output_path)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model_name", type=str, choices=["gpt2", "vit", "whisper", "random_gpt2"], required=True)
    parser.add_argument("-a", "--analysis", type=str, choices=["rank_corr", "kendalls_tau"], required=True)
    parser.add_argument("-n", "--num_samples", type=int, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)
    return parser.parse_args()


def main(args):
    model, tokenizer, processor = load_model(args.model_name)
    num_layers = model.config.num_hidden_layers
    samples = load_samples(args.model_name, args.num_samples)
    if "gpt2" in args.model_name:
        saturation_layer_arr_list = []
        for sample in tqdm(samples):
            indxs_per_layer, _, _ = extract_hidden_layers_reps(args.model_name, model, tokenizer,
                                                            processor, [sample], num_layers)
            if indxs_per_layer is not None:
                saturation_layer_arr_list.append(calc_top_k_saturation_layers(indxs_per_layer, num_layers))
        saturation_layer_arr = np.hstack(saturation_layer_arr_list)
    else:
        indxs_per_layer, _, _ = extract_hidden_layers_reps(args.model_name, model, tokenizer,
                                                        processor, samples, num_layers)
        saturation_layer_arr = calc_top_k_saturation_layers(indxs_per_layer, num_layers)
    if args.analysis == "kendalls_tau":
        calc_strict_kendalls_tau_w_permutation_test(saturation_layer_arr, num_layers, args.output_path)
    else:  # rank_corr
        satur_layer_rank_corr_dict = calc_rank_from_satur_layers_arr(saturation_layer_arr, num_layers)
        plot_rank_saturation_correspondence(satur_layer_rank_corr_dict, args.output_path)
    print(f"Reminder: your results are at {args.output_path}")


if __name__ == '__main__':
    args = args_parse()
    main(args)
