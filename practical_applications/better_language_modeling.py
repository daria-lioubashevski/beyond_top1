import argparse
from collections import defaultdict
import math
import numpy as np
from scipy.stats import norm
from utils import load_model, load_samples, tokenize_text, extract_hidden_layers_reps, \
    calc_top_k_saturation_layers, write_results


def calc_acc_of_late_vs_early_top_2_saturation(indxs_per_layer, num_layers, chunk_input_ids):
    chunk_input_ids = chunk_input_ids[0][1:].numpy()
    saturation_layer_arr = calc_top_k_saturation_layers(indxs_per_layer, num_layers, top_k=2, filter_rel_indxs=False)
    not_correct_top_1_indxs = np.where(np.array(indxs_per_layer[-1, :-1, 0]) != chunk_input_ids)[0]

    all_second_token_det_only_last_layer_indxs = np.where(saturation_layer_arr[1] == num_layers - 1)[0]
    second_token_det_only_last_layer_indxs = list(
        set(not_correct_top_1_indxs).intersection(all_second_token_det_only_last_layer_indxs))
    second_token_det_only_last_layer_preds = indxs_per_layer[-1, second_token_det_only_last_layer_indxs, 1]
    second_token_det_only_last_layer_preds_corr = np.where(
        np.array(second_token_det_only_last_layer_preds) == chunk_input_ids[second_token_det_only_last_layer_indxs])[0]

    all_second_token_det_early_indxs = np.where(saturation_layer_arr[1] <= num_layers - 7)[0]
    second_token_det_early_indxs = list(set(not_correct_top_1_indxs).intersection(all_second_token_det_early_indxs))
    second_token_det_early_preds = indxs_per_layer[-1, second_token_det_early_indxs, 1]
    second_token_det_early_preds_corr = \
        np.where(np.array(second_token_det_early_preds) == chunk_input_ids[second_token_det_early_indxs])[0]

    results = {"second_finalizes_in_last_count": len(second_token_det_only_last_layer_indxs),
               "second_finalizes_in_last_pred_correct_count": len(second_token_det_only_last_layer_preds_corr),
               "second_finalizes_early_count": len(second_token_det_early_indxs),
               "second_finalizes_early_pred_correct_count": len(second_token_det_early_preds_corr)
               }
    return results


def calculate_proportion_test(n1, x1, n2, x2):
    p1 = x1 / n1
    p2 = x2 / n2
    pooled_p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return p_value


def compare_acc_of_late_vs_early_top_2_saturation(layer_number_results, output_path):
    results = []
    late_top2_acc = round(
        100 * layer_number_results["second_finalizes_in_last_pred_correct_count"] / layer_number_results[
            "second_finalizes_in_last_count"], 3)
    results.append(f"{late_top2_acc}% of correct predictions for top-2 saturation in final layer")

    early_top2_acc = round(
        100 * layer_number_results["second_finalizes_early_pred_correct_count"] / layer_number_results[
            "second_finalizes_early_count"], 3)
    results.append(f"{early_top2_acc}% of correct predictions for early top-2 saturaiton (at least 7 layers before last)")

    count = [layer_number_results["second_finalizes_in_last_count"],
             layer_number_results["second_finalizes_early_count"]]
    nobs = [layer_number_results["second_finalizes_in_last_pred_correct_count"],
            layer_number_results["second_finalizes_early_pred_correct_count"]]

    pval = calculate_proportion_test(count[0], nobs[0], count[1], nobs[1])
    pval = "< 0.001" if pval < 0.001 else round(pval, 3)
    results.append(f"t-test p-value {pval}")
    write_results(results, output_path)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_samples", type=int, help="number of samples")
    parser.add_argument("-o", "--output_path", type=str)
    return parser.parse_args()


def main(args):
    model_name = "gpt2"
    model, tokenizer, processor = load_model(model_name)
    num_layers = model.config.num_hidden_layers
    samples = load_samples(model_name, args.num_samples)
    all_results = defaultdict(int)
    for sample in samples:
        input_ids = tokenize_text(tokenizer, sample)
        indxs_per_layer, embds_per_layer, probs_per_layer = extract_hidden_layers_reps(model_name, model,
                                                                                       tokenizer, processor,
                                                                                       [sample], num_layers)
        if indxs_per_layer is not None:
            cur_results = calc_acc_of_late_vs_early_top_2_saturation(indxs_per_layer, num_layers, input_ids)
            for k, val in cur_results.items():
                all_results[k] += val
    compare_acc_of_late_vs_early_top_2_saturation(all_results, args.output_path)


if __name__ == '__main__':
    args = args_parse()
    main(args)
