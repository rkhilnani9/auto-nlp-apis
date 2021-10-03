from transformers import AutoModelForMaskedLM, AutoTokenizer

model_name = "distilbert-base-cased"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_model():
    return model


def get_tokenizer():
    return tokenizer
