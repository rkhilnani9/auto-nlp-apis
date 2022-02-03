from transformers import AutoModelForTokenClassification, AutoTokenizer

model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_model():
    return model


def get_tokenizer():
    return tokenizer
