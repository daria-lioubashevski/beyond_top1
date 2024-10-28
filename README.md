# BEYOND THE TOP-1
This repository provides code and resources for our paper, [Looking Beyond The Top-1: Transformers Determine Top Tokens In Order](). 

## Table of Contents
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
  - [Analysis](#1-analysis)
  - [Intervention](#2-intervention)
  - [Practical Applications](#3-practical-applications)
- [Citation](#citation)

## Installation & Setup

### Prerequisites
- Ensure you have Python 3.8+ and `pip` installed.

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/daria-lioubashevski/beyond_top1
   cd beyond_top1

2. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### 1. Analysis
All scripts support the following models: GPT2-XL (pre-trained and randomly initalized), ViT-L/16, Whisper-large.
* To investigate the order of saturation layers for top-k tokens use `analysis/order_of_saturation_layer_analysis.py`.

**Arguments:**

- `-model` or `--model_name` (str): The model to analyze. Choose from `gpt2`, `vit`, `whisper`, or `random_gpt2`.
- `-a` or `--analysis` (str): The type of analysis to perform. Options are `rank_corr` or `kendalls_tau`.
- `-n` or `--num_samples` (int): Number of samples to use in the analysis.
- `-o` or `--output_path` (str): Path to save the analysis results.

 
 Example usage for ViT model and rank correlation analysis over 200 images:
  ```
  python -m analysis.order_of_saturation_layer_analysis -model vit -a rank_corr -n 200 -o rank_corr.png
  ``` 
* To probe the task information in the model's embeddings, first run `analysis/create_data_for_task_probing.py` to create training data.
  
**Arguments:**
- `-model` or `--model_name` (str): The model to analyze. Choose from `gpt2`, `vit`, `whisper`, or `random_gpt2`.
- `-n` or `--num_samples` (int): Number of samples to use.
- `-k` or `--num_tasks` (int): number of tasks (should probably be between 3 and 5).
- `-o` or `--output_path` (str): Path to save the pkl containing the extracted embeddings.
  

  Then use `analysis/run_task_transition_probing.py` to train and evaluate the classifier.
  
**Arguments:**
- `-d` or `--data_path` (str): Path to data (extracted embeddings) for probing. 
- `-n` or `--num_tasks` (int): number of tasks (should probably be between 3 and 5).
- `-k` or `--kfolds` (int): number of kfolds in training.
- `-o` or `--output_path` (str): Path to save the analysis results.

  
  Example usage with GPT2 model for tasks 1 to 5 over 50 texts:
  ```
  python -m analysis.create_data_for_task_probing -model gpt2 -n 50 -k 5 -o gpt2_embds_for_probing.pkl
  python -m analysis.run_task_transition_probing -n 4 -k 5 -d gpt2_embds_for_probing.pkl -o probing_results.txt
  ``` 

### 2. Intervention
All scripts support the following models: GPT2-XL (pre-trained only), ViT-L/16, Whisper-large.
To perform casusal intervention on the model activations causing it to switch from task-1 to task-2 use `analysis/run_intervention_procedure.py`

**Arguments:**

- `-model` or `--model_name` (str): The model to analyze. Choose from `gpt2`, `vit`, `whisper`, or `random_gpt2`.
- `-n` or `--num_pairs` (int): Number of pairs to use in intervention procedure.
- `-o` or `--output_path` (str): Path to save the intervention figure.

 Example usage for Whisper model over 100 pairs:
  ```
  python -m intervention.run_intervention_procedure -model whisper -n 100 -o interv_result.png
  ``` 

### 3. Practical Applications
All script currently support only GPT-2 model.
* To compare the performance of our new early exit measure against existing ones in terms of performance/efficiency tradeoff run 
`practical_applications/new_early_exit.py`.

**Arguments:**

- `-n` or `--num_samples` (int): Number of samples (texts) to use in comparison.
- `-o` or `--output_path` (str): Path to save resulting figure.

 Example usage with already trained task index classifier:
 ```
  python -m practical_applications.new_early_exit -n 10 -c GPT2_top5_clf.pkl -o ee_compar.png
 ```

* To see how the saturation layer affects language modeling use  `practical_applications/better_language_modeling.py`.

**Arguments:**  

- `-n` or `--num_samples` (int): Number of samples (texts) to use in comparison.
- `-o` or `--output_path` (str): Path to save results. 


  Example usage:
 ```
  python -m practical_applications.better_language_modeling -n 20 -o lang_modeling.txt
 ```

## Citation
To cite our work, please use the following BibTeX entry:
