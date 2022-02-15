import torch
import numpy as np

from transformers import pipeline
from transformers import AutoModelForTokenClassification, AutoTokenizer
from autonlp.ner.model import get_model, get_tokenizer

model = get_model()
tokenizer = get_tokenizer()


def infer(input_text, pretrained):
    if pretrained == "true":
        named_entity_recognizer = pipeline(
        "ner", model=model, tokenizer=tokenizer, grouped_entities=True
        )

        result = named_entity_recognizer(input_text.lower())

    # FastAPI cannot json encode numpy.float32
        for entity_dict in result:
            entity_dict["score"] = float(entity_dict["score"])
            del entity_dict["start"], entity_dict["end"]

    else:
        named_entity_recognizer = AutoModelForTokenClassification.from_pretrained(
            config.NER_MODEL_SAVE_PATH + "checkpoint-500/")
        tokens = tokenizer(input_text)

        # Get predictions

        predictions = named_entity_recognizer(**tokens)
        predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
        predictions = list(predictions.detach().cpu().numpy())
        predicted_entities = [label_list[i] for i in predictions]
        words = tokenizer.batch_decode(test_tokens['input_ids'])

        # Remove [CLS] and [SEP] tokens

        predicted_entities = predicted_entities[1:-1]
        words = words[1:-1]
        predictions = predictions[1:-1]

        result = []
        for idx in range(len(words)):
            result_dict = {"entity_group" : predicted_entities[idx], "score" : predictions[idx], "word" : words[idx]}
            result.append(result_dict)

    return result
