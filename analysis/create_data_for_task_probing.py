import argparse

from consts import MIN_LAYER_FOR_PROBING
from utils import load_model, load_samples, extract_hidden_layers_reps, calc_top_k_saturation_layers
import pickle
from collections import defaultdict


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", "--model_name", type=str, choices=["gpt2", "vit", "whisper", "random_gpt2"])
    parser.add_argument("-n", "--num_samples", type=int, help="number of samples")
    parser.add_argument("-k", "--num_tasks", type=int, help="number of tasks (should probably be between 3 to 5)")
    parser.add_argument("-o", "--output_path", type=str, help="path to pkl containing training data")
    return parser.parse_args()


def get_layer_embds_for_task_transition(indxs_per_layer, embds_per_layer, num_layers, k=5):
    saturation_layer_arr = calc_top_k_saturation_layers(indxs_per_layer, num_layers)
    min_layer, max_layer = MIN_LAYER_FOR_PROBING, num_layers

    layer_embd_dict = {layer: defaultdict(list) for layer in range(min_layer, max_layer)}
    for ind in range(saturation_layer_arr.shape[1]):
        for i in range(k):
            if saturation_layer_arr[i][ind] > min_layer:
                if i == 0:
                    start_layer = min_layer
                else:
                    start_layer = max(min_layer, saturation_layer_arr[i - 1][ind] + 1)
            else:
                continue
            end_layer = min(saturation_layer_arr[i][ind] + 1, max_layer)
            for layer in range(start_layer, end_layer):
                layer_embd_dict[layer][i].append(embds_per_layer[layer, ind, :])
    return layer_embd_dict


def main(args):
    model, tokenizer, processor = load_model(args.model_name)
    num_layers = model.config.num_hidden_layers
    samples = load_samples(args.model_name, args.num_samples)
    if "gpt2" in args.model_name:
        total_layer_embd_dict = {layer: defaultdict(list) for layer in range(MIN_LAYER_FOR_PROBING, num_layers)}
        for j, sample in enumerate(samples):
            indxs_per_layer, embds_per_layer, _ = extract_hidden_layers_reps(args.model_name, model,
                                                                             tokenizer, processor,
                                                                             [sample], num_layers,
                                                                             extract_embds=True)
            if indxs_per_layer is not None:
                layer_embd_dict = get_layer_embds_for_task_transition(indxs_per_layer, embds_per_layer, num_layers,
                                                                      args.num_tasks)
                for layer in layer_embd_dict:
                    for j in range(args.num_tasks):
                        total_layer_embd_dict[layer][j] += layer_embd_dict[layer][j]

    else:
        indxs_per_layer, embds_per_layer, _ = extract_hidden_layers_reps(args.model_name, model, tokenizer,
                                                                         processor, samples, num_layers,
                                                                         extract_embds=True)
        total_layer_embd_dict = get_layer_embds_for_task_transition(indxs_per_layer, embds_per_layer, num_layers,
                                                              args.num_tasks)
    with open(args.output_path, "wb") as f:
        pickle.dump(total_layer_embd_dict, f)


if __name__ == '__main__':
    args = args_parse()
    main(args)
