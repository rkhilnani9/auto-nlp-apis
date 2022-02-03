from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model_name = "distilbert-base-cased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_model():
    return model


def get_tokenizer():
    return tokenizer
