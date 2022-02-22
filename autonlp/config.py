DATA_ROOT = "/Users/rkhilnan/Desktop/auto-nlp-apis/autotune_models/"

# Training params

PRETRAINED_MODEL = "distilbert-base-uncased"
NUM_EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# Model save paths

CLS_MODEL_SAVE_PATH = f"{DATA_ROOT}/classification/"
NER_MODEL_SAVE_PATH = f"{DATA_ROOT}/ner/"
