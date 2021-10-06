from transformers import pipeline
from fill_mask.model import get_model, get_tokenizer

model = get_model()
tokenizer = get_tokenizer()


def infer(input_text):
    fill_mask_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    result = fill_mask_pipeline(input_text)

    # FastAPI cannot json encode numpy.float32
    for entity_dict in result:
        entity_dict["score"] = float(entity_dict["score"])

    return result
