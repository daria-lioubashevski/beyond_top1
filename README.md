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
   git clone https://github.com/daria-lioubashevski/saturation_beyond_top1
   cd saturation_beyond_top1

2. Install required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### 1. Analysis
All scripts support the following models: GPT2-XL (pre-trained and randomly initalized), ViT-L/16, Whisper-large.
* To investigate the order of saturation layers for top-k tokens use `analysis/order_of_saturation_layer_analysis.py`.
  
  Example cmd using ViT model:
  ```
  python analysis/order_of_saturation_layer_analysis.py -m vit -a rank_corr -n 200
  ``` 
* To probe the task information in the model's embeddings, first run `analysis/create_data_for_task_probing.py` to create training data and then use `analysis/run_task_transition_probing.py` to train and evaluate the classifier.

  Example cmds using GPT2 model:
  ```
  python analysis/create_data_for_task_probing.py -m gpt2 -n 50 -k 5 -o gpt2_embds_for_probing.pkl
  python analysis/run_task_transition_probing.py -n 4 -k 5 -d gpt2_embds_for_probing.pkl
  ``` 

### 2. Intervention
All scripts support the following models: GPT2-XL (pre-trained only), ViT-L/16, Whisper-large.
To perform casusal intervention on the model activations causing it to switch from task-1 to task-2 use `analysis/run_intervention_procedure.py`

 Example cmds using Whisper model:
  ```
  python intervention/run_intervention_procedure.py -m whisper -n 100
  ``` 

### 3. Practical Applications
All script currently support only GPT-2 model.
* To compare the performance of our new early exit measure against existing ones run 
`practical_applications/new_early_exit.py`.

 Example cmds using already trained task index classifier:
 ```
  python practical_applications/new_early_exit.py -n 10 -c GPT2_top5_clf.pkl
 ```

* To show how the saturation layer affects language modeling use  `practical_applications/better_language_modeling.py`.

  Example cmd:
 ```
  python practical_applications/better_language_modeling.py -n 20
 ```

## Citation
To cite our work, please use the following BibTeX entry:
