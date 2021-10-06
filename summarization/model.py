from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_model():
    return model


def get_tokenizer():
    return tokenizer
