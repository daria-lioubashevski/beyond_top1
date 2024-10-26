TEXT_INPUT_LENGTH = 1024
MAX_TOP_1_SATUR_LAYER_RATIO = 0.85
KENDALLS_TAU_NUM_PERMUTATIONS = 1000
MIN_NUM_EMBDS_IN_CLASS = 5
MIN_LAYER_FOR_PROBING = 5
MODEL_NAME_MAPPING = {"gpt2": "gpt2-xl",
                      "random_gpt2": "gpt2-xl",
                      "vit": ["google/vit-large-patch16-224", "google/vit-large-patch16-224-in21k"],
                      "whisper": "openai/whisper-large"}
NUM_ADJACENT_LAYERS_FOR_INTERV_MAPPING = {"gpt2": 5,
                                          "vit": 3,
                                          "whisper": 3}
MIN_DIFF_IN_SATUR_LAYERS_FOR_INTERV_MAPPING = {"gpt2": 10,
                                               "vit": 5,
                                               "whisper": 5}
INTERV_RESULTS_CSV_NAME = "interv_over_layers.csv"
SOFTMAX_THRS_FOR_EE = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94]
COS_SIM_THR_FOR_EE = [0.986, 0.988, 0.989, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995]