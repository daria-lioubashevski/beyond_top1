import argparse
import torch
import pandas as pd
import numpy as np

from consts import MIN_DIFF_IN_SATUR_LAYERS_FOR_INTERV_MAPPING, NUM_ADJACENT_LAYERS_FOR_INTERV_MAPPING, \
    INTERV_RESULTS_CSV_NAME
from plots import plot_interv_results
from utils import load_model, load_samples, extract_hidden_layers_reps, calc_top1_satur_layer, \
    get_indxs_from_hidden_layers_vit, calc_top1_satur_layer_single_img, get_indxs_from_hidden_layers_whisper, \
    tokenize_text, calc_num_tokens_per_sample_whisper, get_indxs_from_hidden_layers_gpt


def is_pair_fit_for_interv(model_name, num_layers, model_preds, num_layer_where_decision_finalizes, i, j):
    min_diff_in_satur_layers = MIN_DIFF_IN_SATUR_LAYERS_FOR_INTERV_MAPPING[model_name]
    diff_in_satur_layers = np.abs(num_layer_where_decision_finalizes[i] - num_layer_where_decision_finalizes[j])
    same_pred = model_preds[i] == model_preds[j]
    if model_name == "vit":
        return (same_pred and diff_in_satur_layers >= min_diff_in_satur_layers)
    elif model_name == "whisper":
        return (same_pred and (model_preds[i] != 50258) and diff_in_satur_layers >= min_diff_in_satur_layers
                and max(num_layer_where_decision_finalizes[i], num_layer_where_decision_finalizes[j]) <= num_layers - 5)
    else:  # gpt2
        return (same_pred and diff_in_satur_layers >= min_diff_in_satur_layers
                and j - i <= 30 and min(num_layer_where_decision_finalizes[i],
                                        num_layer_where_decision_finalizes[j]) >= 10)


def find_candidates_for_intervention(model_name, indxs_per_layer, embds_per_layer, num_layers, num_adjacent_layers):
    num_layer_where_decision_finalizes = calc_top1_satur_layer(indxs_per_layer, num_layers)
    model_preds = indxs_per_layer[-1, :, 0]

    pairs = []
    for i in range(len(model_preds)):
        for j in range(i + 1, len(model_preds)):
            if is_pair_fit_for_interv(model_name, num_layers, model_preds, num_layer_where_decision_finalizes, i, j):
                # taking embeddings from x layers before and x layers after saturation to implant
                if num_layer_where_decision_finalizes[i] < num_layer_where_decision_finalizes[j]:
                    pair_dict = {'class_indx': model_preds[i], 'early_layer': num_layer_where_decision_finalizes[i],
                                 'late_layer': num_layer_where_decision_finalizes[j], 'late_indx': j,
                                 'embds': embds_per_layer[
                                          num_layer_where_decision_finalizes[i] - num_adjacent_layers:
                                          num_layer_where_decision_finalizes[i] + num_adjacent_layers + 1, i, :]}
                else:
                    pair_dict = {'class_indx': model_preds[i], 'early_layer': num_layer_where_decision_finalizes[j],
                                 'late_layer': num_layer_where_decision_finalizes[i], 'late_indx': i,
                                 'embds': embds_per_layer[
                                          num_layer_where_decision_finalizes[j] - num_adjacent_layers:
                                          num_layer_where_decision_finalizes[j] + num_adjacent_layers + 1, j, :]}
                pairs.append(pair_dict)
    return pairs


def hook_fn_vit(module, inputs, outputs, new_vector):
    input_tensor = inputs[0].clone()
    # Replace the [CLS] token at position 0 with the new vector
    input_tensor[:, 0, :] = new_vector
    # Return the modified inputs, preserving the rest of the original inputs
    return (input_tensor, *inputs[1:])


def hook_fn_lang(module, inputs, outputs, target_index, new_vector):
    input_tensor = inputs[0].clone()
    input_tensor[:, target_index, :] = new_vector
    return (input_tensor, *inputs[1:])


def run_mode_forward_w_inter_multiple_layers_vit(model, num_layers, img_processor, chunk_imgs, chunk_interv_pairs):
    early_satur_layers, inter_layers, late_satur_layers, new_satur_layers, preds_unchanged = [], [], [], [], []
    num_adj_layers = NUM_ADJACENT_LAYERS_FOR_INTERV_MAPPING["vit"]
    for pair in chunk_interv_pairs:
        embds = pair['embds']
        img = chunk_imgs[pair["late_indx"]]
        inputs = img_processor(img, return_tensors="pt")

        for i, embd in enumerate(embds):
            new_vector = torch.tensor(embd)
            insert_layer = pair["early_layer"] - num_adj_layers + i + 1
            layer = model.vit.encoder.layer[
                insert_layer]  # this is output of pair["early_layer"], so should serve as input to next layer
            hook = layer.register_forward_hook(
                lambda module, inputs, outputs: hook_fn_vit(module, inputs, outputs, new_vector))

            torch.cuda.empty_cache()
            output = model(**inputs, output_hidden_states=True)
            hook.remove()

            hidden_states = output.hidden_states
            all_layer_indxs = get_indxs_from_hidden_layers_vit(model, hidden_states, num_layers)
            del output, hidden_states

            num_satur_layer = calc_top1_satur_layer_single_img(all_layer_indxs)
            new_satur_layers.append(num_satur_layer)
            late_satur_layers.append(pair['late_layer'])
            early_satur_layers.append(pair["early_layer"])
            inter_layers.append(insert_layer)

            orig_pred = pair['class_indx']
            pred_unchanged = all_layer_indxs[-1, 0] == orig_pred
            preds_unchanged.append(pred_unchanged)
            torch.cuda.empty_cache()

    layers_df = pd.DataFrame({'early_satur_layer': early_satur_layers,
                              'inter_layer': inter_layers,
                              'late_satur_layer': late_satur_layers,
                              'new_satur_layer': new_satur_layers,
                              'pred_unchanged': preds_unchanged})
    layers_df.to_csv(INTERV_RESULTS_CSV_NAME)


def run_mode_forward_w_inter_multiple_layers_whisper(model, num_layers, processor, samples, pairs):
    early_satur_layers, inter_layers, late_satur_layers, new_satur_layers, preds_unchanged = [], [], [], [], []
    num_adj_layers = NUM_ADJACENT_LAYERS_FOR_INTERV_MAPPING["whisper"]
    for pair in pairs:
        target_index = pair['late_indx']
        embds = pair['embds']
        audio_indx = int(pair['sample_ind'])

        audio_sample = samples[audio_indx]["audio"]
        waveform = audio_sample["array"]
        sampling_rate = audio_sample["sampling_rate"]
        input_features = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features

        target_seq = samples[audio_indx]["text"]
        decoder_input_ids = processor.tokenizer(target_seq, return_tensors="pt").input_ids

        for i, embd in enumerate(embds):
            new_vector = torch.tensor(embd)
            insert_layer = pair["early_layer"] - num_adj_layers + i + 1

            layer = model.model.decoder.layers[
                insert_layer]  # this is output of pair["early_layer"], so should serve as input to next layer
            hook = layer.register_forward_hook(
                lambda module, inputs, outputs: hook_fn_lang(module, inputs, outputs, target_index, new_vector))
            output = model(input_features, decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                           return_dict=True)
            hook.remove()

            hidden_states = output.decoder_hidden_states
            all_layer_indxs = get_indxs_from_hidden_layers_whisper(hidden_states, model, num_layers)
            del output, hidden_states

            num_layer_where_decision_finalizes = calc_top1_satur_layer(all_layer_indxs, num_layers)

            new_satur_layers.append(num_layer_where_decision_finalizes[target_index])
            late_satur_layers.append(pair['late_layer'])
            early_satur_layers.append(pair["early_layer"])
            inter_layers.append(insert_layer)

            orig_pred = pair['class_indx']
            pred_unchanged = all_layer_indxs[-1, target_index, 0] == orig_pred
            preds_unchanged.append(pred_unchanged)
            torch.cuda.empty_cache()

    layers_df = pd.DataFrame({'early_satur_layer': early_satur_layers,
                              'inter_layer': inter_layers,
                              'late_satur_layer': late_satur_layers,
                              'new_satur_layer': new_satur_layers,
                              'pred_unchanged': preds_unchanged})
    layers_df.to_csv(INTERV_RESULTS_CSV_NAME)


def run_mode_forward_w_inter_multiple_layers_gpt(model, num_layers, tokenizer, samples, pairs):
    early_satur_layers, inter_layers, late_satur_layers, new_satur_layers, preds_unchanged = [], [], [], [], []
    num_adj_layers = NUM_ADJACENT_LAYERS_FOR_INTERV_MAPPING["gpt2"]
    for pair in pairs:
        target_index = pair['late_indx']
        embds = pair['embds']
        text = samples[pair["sample_ind"]]
        input_ids = tokenize_text(tokenizer, text)
        for i, embd in enumerate(embds):
            new_vector = torch.tensor(embd)
            insert_layer = pair["early_layer"] - num_adj_layers + i + 1
            layer = model.transformer.h[insert_layer]
            # this is output of pair["early_layer"], so should serve as input to next layer
            hook = layer.register_forward_hook(
                lambda module, inputs, outputs: hook_fn_lang(module, inputs, outputs, target_index, new_vector))

            outputs = model(input_ids[:target_index + 1], return_dict=True, output_hidden_states=True,
                            use_cache=False)
            hook.remove()

            hidden_states = outputs.hidden_states
            all_layer_indxs = get_indxs_from_hidden_layers_gpt(model, tokenizer, hidden_states, num_layers)
            del outputs, hidden_states

            num_layer_where_decision_finalizes = calc_top1_satur_layer(all_layer_indxs, num_layers)

            new_satur_layers.append(num_layer_where_decision_finalizes[target_index])
            late_satur_layers.append(pair['late_layer'])
            early_satur_layers.append(pair["early_layer"])
            inter_layers.append(insert_layer)

            orig_pred = pair['class_indx']
            pred_unchanged = all_layer_indxs[-1, target_index, 0] == orig_pred
            preds_unchanged.append(pred_unchanged)

    layers_df = pd.DataFrame({'early_satur_layer': early_satur_layers,
                              'inter_layer': inter_layers,
                              'late_satur_layer': late_satur_layers,
                              'new_satur_layer': new_satur_layers,
                              'pred_unchanged': preds_unchanged})
    layers_df.to_csv(INTERV_RESULTS_CSV_NAME)


def run_mode_forward_w_inter_multiple_layers_wrapper(model_name, model, num_layers, tokenizer, processor,
                                                     samples, pairs):
    if model_name == "vit":
        run_mode_forward_w_inter_multiple_layers_vit(model, num_layers, processor, samples, pairs)
    elif model_name == "whisper":
        run_mode_forward_w_inter_multiple_layers_whisper(model, num_layers, processor, samples, pairs)
    else:  # gpt2
        run_mode_forward_w_inter_multiple_layers_gpt(model, num_layers, tokenizer, samples, pairs)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model_name", type=str, choices=["gpt2", "vit", "whisper"])
    parser.add_argument("-n", "--num_pairs", type=int, default=100)
    parser.add_argument("-o", "--output_path", type=str)
    return parser.parse_args()


def get_samples_for_interv(model_name, num_pairs):
    if model_name == "gpt2":
        samples = load_samples(args.model_name, num_samples=num_pairs // 10)
    elif model_name == "vit":
        samples = load_samples(args.model_name, num_samples=num_pairs * 5)
    else:  # whisper
        samples = load_samples(args.model_name, num_samples=num_pairs * 3)
    return samples


def fix_pair_indxs_for_interv_whisper(pairs, num_tokens_per_sample):
    tokens_per_sample_cumsum = np.cumsum(num_tokens_per_sample)
    for pair in pairs:
        late_ind = pair["late_indx"]
        if late_ind < tokens_per_sample_cumsum[0]:
            sample_ind = 0
            true_late_ind = late_ind
        else:
            sample_ind = np.where(tokens_per_sample_cumsum < late_ind)[0][-1] + 1
            true_late_ind = late_ind - tokens_per_sample_cumsum[sample_ind - 1]

        pair["late_indx"] = true_late_ind
        pair["sample_ind"] = sample_ind
    return pairs


def main(args):
    model, tokenizer, processor = load_model(args.model_name)
    num_layers = model.config.num_hidden_layers
    samples = get_samples_for_interv(args.model_name, args.num_pairs)
    num_adjacent_layers = NUM_ADJACENT_LAYERS_FOR_INTERV_MAPPING[args.model_name]
    if args.model_name == "vit":
        indxs_per_layer, embds_per_layer = extract_hidden_layers_reps(args.model_name, model, tokenizer, processor,
                                                                      samples, num_layers, extract_embds=True,
                                                                      embds_before_ln=True)
        pairs = find_candidates_for_intervention(args.model_name, indxs_per_layer, embds_per_layer,
                                                 num_layers, num_adjacent_layers)
    elif args.model_name == "whisper":
        num_tokens_per_sample = calc_num_tokens_per_sample_whisper(processor, samples)
        indxs_per_layer, embds_per_layer = extract_hidden_layers_reps(args.model_name, model, tokenizer, processor,
                                                                      samples, num_layers, extract_embds=True,
                                                                      embds_before_ln=True)
        pairs = find_candidates_for_intervention(args.model_name, indxs_per_layer, embds_per_layer,
                                                     num_layers, num_adjacent_layers)
        pairs = fix_pair_indxs_for_interv_whisper(pairs, num_tokens_per_sample)

    else:  # gpt2
        pairs = []
        for ind, sample in enumerate(samples):
            if len(pairs) < args.num_pairs:
                indxs_per_layer, embds_per_layer = extract_hidden_layers_reps(args.model_name, model, tokenizer,
                                                                              processor, [sample], num_layers,
                                                                              extract_embds=True,
                                                                              embds_before_ln=True)
                cur_pairs = find_candidates_for_intervention(args.model_name, indxs_per_layer, embds_per_layer,
                                                             num_layers, num_adjacent_layers)
                for pair in cur_pairs:
                    pair["sample_ind"] = ind
                pairs += cur_pairs

    pairs = pairs[:args.num_pairs]
    run_mode_forward_w_inter_multiple_layers_wrapper(args.model_name, model, num_layers, tokenizer,
                                                     processor, samples, pairs)
    interv_results_df = pd.read_csv(INTERV_RESULTS_CSV_NAME)
    plot_interv_results(args.model_name, interv_results_df, args.output_path)


if __name__ == '__main__':
    args = args_parse()
    main(args)
