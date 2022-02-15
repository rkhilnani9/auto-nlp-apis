from transformers import pipeline
from autonlp.summarization.model import get_model, get_tokenizer

model = get_model()
tokenizer = get_tokenizer()


def infer(input_text, pretrained):
    summarization_pipeline = pipeline(
        "summarization", model=model, tokenizer=tokenizer
    )

    num_words = len(input_text.split())

    min_summary_length = int(num_words/10)
    max_summary_length = int(num_words/2)

    return summarization_pipeline(input_text, max_length=max_summary_length, min_length=min_summary_length)
