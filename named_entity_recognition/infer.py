from transformers import pipeline
from named_entity_recognition.model import get_model, get_tokenizer

model = get_model()
tokenizer = get_tokenizer()


def infer(input_text):
    named_entity_recognizer = pipeline(
        "ner", model=model, tokenizer=tokenizer, grouped_entities=True
    )

    result = named_entity_recognizer(input_text.lower())

    # FastAPI cannot json encode numpy.float32
    for entity_dict in result:
        entity_dict["score"] = float(entity_dict["score"])

    return result