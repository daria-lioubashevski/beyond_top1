import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, \
    WhisperProcessor, WhisperForConditionalGeneration, AutoImageProcessor, ViTForImageClassification
from datasets import load_dataset

from consts import MODEL_NAME_MAPPING, TEXT_INPUT_LENGTH, MAX_TOP_1_SATUR_LAYER_RATIO


def load_model(model_name):
    if "gpt2" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_MAPPING[model_name])
        if "random" in model_name:
            config = AutoConfig.from_pretrained(MODEL_NAME_MAPPING[model_name])
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_MAPPING[model_name])
        return model, tokenizer, None
    elif model_name == "vit":
        model = ViTForImageClassification.from_pretrained(MODEL_NAME_MAPPING[model_name][0])
        image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME_MAPPING[model_name][1])
        return model, None, image_processor
    elif model_name == "whisper":
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME_MAPPING[model_name])
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_MAPPING[model_name], add_prefix_space=True)
        processor = WhisperProcessor.from_pretrained(MODEL_NAME_MAPPING[model_name])
        return model, tokenizer, processor
    return None


def load_samples(model_name, num_samples):
    if "gpt2" in model_name:
        dataset = load_dataset("cnn_dailymail", '3.0.0')["train"]
        dataset = dataset["article"]
    elif model_name == "vit":
        dataset = load_dataset("uoft-cs/cifar10", split="train", trust_remote_code=True)
        dataset = dataset["img"]
    else:  # model_name = whisper
        dataset = load_dataset("Raziullah/librispeech_small_asr_fine-tune", split="train[:40%]")
    samples_indxs = np.random.permutation(range(len(dataset)))[:num_samples]
    samples = [dataset[int(ind)] for ind in samples_indxs]
    return samples


def calc_num_tokens_per_sample_whisper(processor, samples):
    num_tokens_per_sample = []
    for sample in samples:
        target_seq = sample["text"]
        decoder_input_ids = processor.tokenizer(target_seq, return_tensors="pt").input_ids
        num_tokens_per_sample.append(len(decoder_input_ids[0]))
    return num_tokens_per_sample


def tokenize_text(tokenizer, text):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    input_ids = input_ids[:TEXT_INPUT_LENGTH]
    return input_ids


def calc_top1_satur_layer(indxs_per_layer, num_layers):
    final_token_indxs = list(indxs_per_layer[-1][:, 0])
    all_layers_top_token_indxs = np.array([list(layer_results[:, 0]) for layer_results in indxs_per_layer])

    token_not_matching_final_deicison_mask = np.not_equal(all_layers_top_token_indxs, final_token_indxs)
    # we want to find last layer where it's not equal <-> reverse order, first layer where it's not equal
    reverse_token_not_matching_final_deicison_mask = token_not_matching_final_deicison_mask[::-1, :]
    num_layers_not_matching = np.argmax(reverse_token_not_matching_final_deicison_mask, axis=0)
    num_layers_not_matching[num_layers_not_matching < 1] = num_layers
    num_layer_where_decision_finalizes = num_layers - num_layers_not_matching
    return num_layer_where_decision_finalizes


def get_indxs_from_hidden_layers_vit(model, hidden_states, num_layers):
    all_layer_indxs = []
    for i in range(1, num_layers + 1):
        h = hidden_states[i]
        if i != num_layers:
            h_norm = model.vit.layernorm(h)
        else:
            h_norm = h
        layer_logits = model.classifier(h_norm[:, 0, :])[0].float().cpu().detach()
        _, indxs = torch.sort(layer_logits, dim=-1, descending=True)
        all_layer_indxs.append(indxs)
    all_layer_indxs = np.array(all_layer_indxs)
    return all_layer_indxs


def get_indxs_from_hidden_layers_gpt(model, tokenizer, hidden_states, num_layers):
    all_layer_indxs = []
    for i in range(1, num_layers + 1):
        h = hidden_states[i].cpu().detach()
        if i != num_layers:
            h_norm = model.transformer.ln_f(h).cpu().detach()
        else:
            h_norm = h
        batch_size, sequence_length, hidden_size = h.shape
        layer_logits = model.lm_head(h_norm).float().cpu().detach()
        layer_logits = torch.reshape(layer_logits,
                                     [batch_size, sequence_length, tokenizer.vocab_size]).squeeze()
        _, indxs = torch.sort(layer_logits, dim=-1, descending=True)
        all_layer_indxs.append(indxs)
    all_layer_indxs = np.array(all_layer_indxs)
    return all_layer_indxs


def get_indxs_from_hidden_layers_whisper(hidden_states, model, num_layers):
    all_layer_indxs = []
    for i in range(1, num_layers + 1):
        h = hidden_states[i]
        if i != num_layers:
            h_norm = model.model.decoder.layer_norm(h)
        else:
            h_norm = h
        lm_logits = model.proj_out(h_norm).float().cpu().detach().squeeze()
        _, indxs = torch.sort(lm_logits, dim=-1, descending=True)
        all_layer_indxs.append(indxs.numpy())
    all_layer_indxs = np.array(all_layer_indxs)
    return all_layer_indxs


def calc_top1_satur_layer_single_img(all_layer_indxs):
    top_1_all_layers = all_layer_indxs[:, 0]
    final_pred = top_1_all_layers[-1]
    not_match_final_pred = np.where(top_1_all_layers != final_pred)[0][-1]
    satur_layer = not_match_final_pred + 1
    return satur_layer


def extract_hidden_layers_reps(model_name, model, tokenizer, processor, samples,
                               num_layers, extract_embds=False, embds_before_ln=False,
                               extract_probs=False):
    embds_per_layer, probs_per_layer = None, None
    indxs_per_layer = [[] for _ in range(num_layers)]
    if extract_embds:
        embds_per_layer = [[] for _ in range(num_layers)]
    if extract_probs:
        probs_per_layer = [[] for _ in range(num_layers)]

    if model_name == "vit":
        for ind, image in tqdm(enumerate(samples)):
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                torch.cuda.empty_cache()
                output = model(**inputs, output_hidden_states=True)
                hidden_states = output.hidden_states

                for i in range(1, num_layers + 1):
                    h = hidden_states[i]
                    if i != num_layers:
                        h_norm = model.vit.layernorm(h)
                    else:
                        h_norm = h
                    layer_logits = model.classifier(h_norm[:, 0, :])[0].float().cpu().detach()
                    _, indxs = torch.sort(layer_logits, dim=-1, descending=True)
                    indxs_per_layer[i - 1].append(indxs.numpy())
                    if extract_embds:
                        if embds_before_ln:
                            embds_per_layer[i - 1].append(h[:, 0, :].numpy())
                        else:
                            embds_per_layer[i - 1].append(h_norm[:, 0, :].numpy())
                del hidden_states

    elif model_name == "whisper":
        for ind in tqdm(range(len(samples))):
            torch.cuda.empty_cache()
            audio_sample = samples[ind]["audio"]
            waveform = audio_sample["array"]
            sampling_rate = audio_sample["sampling_rate"]
            input_features = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features
            target_seq = samples[ind]["text"]
            decoder_input_ids = processor.tokenizer(target_seq, return_tensors="pt").input_ids

            with torch.no_grad():
                output = model(input_features, decoder_input_ids=decoder_input_ids, output_hidden_states=True,
                               return_dict=True)

            hidden_states = output.decoder_hidden_states
            for i in range(1, num_layers + 1):
                h = hidden_states[i]
                if i != num_layers:
                    h_norm = model.model.decoder.layer_norm(h)
                else:
                    h_norm = h
                lm_logits = model.proj_out(h_norm).float().cpu().detach().squeeze()
                _, indxs = torch.sort(lm_logits, dim=-1, descending=True)
                indxs_per_layer[i - 1].append(indxs.numpy())
                if extract_embds:
                    if embds_before_ln:
                        embds_per_layer[i - 1].append(h.float().cpu().detach().squeeze().numpy())
                    else:
                        embds_per_layer[i - 1].append(h_norm.float().cpu().detach().squeeze().numpy())
            del hidden_states

    else:  # gpt-2
        assert (len(samples) == 1)
        text = samples[0]
        cur_input_ids = tokenize_text(tokenizer, text)
        with torch.no_grad():
            try:
                outputs = model(cur_input_ids, return_dict=True, output_hidden_states=True, use_cache=False)
                hidden_states = outputs.hidden_states
                for i in range(1, len(hidden_states)):
                    h = hidden_states[i].cpu().detach()
                    if i != num_layers:
                        h_norm = model.transformer.ln_f(h).cpu().detach()
                    else:
                        h_norm = h

                    batch_size, sequence_length, hidden_size = h.shape
                    layer_logits = model.lm_head(h_norm).float().cpu().detach()
                    layer_logits = torch.reshape(layer_logits,
                                                 [batch_size, sequence_length, tokenizer.vocab_size]).squeeze()

                    sorted_logits, indxs = torch.sort(layer_logits, dim=-1, descending=True)
                    if extract_probs:
                        sorted_probs = torch.softmax(sorted_logits, axis=-1)
                        probs_per_layer[i - 1].append(sorted_probs.numpy())
                    indxs_per_layer[i - 1].append(indxs.numpy())
                    if extract_embds:
                        if embds_before_ln:
                            h_flat = torch.reshape(h_norm, [batch_size * sequence_length, hidden_size])
                        else:
                            h_flat = torch.reshape(h, [batch_size * sequence_length, hidden_size])
                        embds_per_layer[i - 1].append(h_flat.numpy())
                del outputs, hidden_states
                indxs_per_layer = np.array(indxs_per_layer).squeeze()
                embds_per_layer = np.array(embds_per_layer).squeeze()
                return indxs_per_layer, embds_per_layer, np.array(probs_per_layer).squeeze()

            except Exception as e:
                return None, None, None

    indxs_stacked = [np.vstack(layer_indxs) for layer_indxs in indxs_per_layer]
    indxs_per_layer = np.array(indxs_stacked)

    if extract_embds:
        embds_per_layer = [np.vstack(layer_embds) for layer_embds in embds_per_layer]
        embds_per_layer = np.array(embds_per_layer)

    return indxs_per_layer, embds_per_layer, probs_per_layer


def calc_top_k_saturation_layers(indxs_per_layer, num_layers, top_k=5, filter_rel_indxs=True):
    saturation_layer_arr = []
    for i in range(top_k):
        final_layer_top_k_indx = list(indxs_per_layer[-1][:, i])
        all_layers_top_k_indx = np.array([list(layer_results[:, i]) for layer_results in indxs_per_layer])

        token_not_matching_final_deicison_mask = np.not_equal(all_layers_top_k_indx, final_layer_top_k_indx)
        reverse_token_not_matching_final_deicison_mask = token_not_matching_final_deicison_mask[::-1, :]
        top_k_saturation_layer = num_layers - np.argmax(reverse_token_not_matching_final_deicison_mask, axis=0)
        saturation_layer_arr.append(top_k_saturation_layer)

    saturation_layer_arr = np.array(saturation_layer_arr)
    if filter_rel_indxs:
        rel_indxs_mask = np.where(saturation_layer_arr[0] <= int(MAX_TOP_1_SATUR_LAYER_RATIO * num_layers))[0]
        return saturation_layer_arr[:, rel_indxs_mask]
    else:
        return saturation_layer_arr


def write_results(results, output_path):
    with open(output_path, "w+") as f:
        f.writelines([r + '\n' for r in results])
