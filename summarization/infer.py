from transformers import pipeline
from summarization.model import get_model, get_tokenizer

model = get_model()
tokenizer = get_tokenizer()


def infer(input_text):
    summarization_pipeline = pipeline(
        "summarization", model=model, tokenizer=tokenizer
    )

    num_words = len(input_text.split())

    min_summary_length = int(num_words//10)
    max_summary_length = int(num_words//2)

    print(min_summary_length)
    print(max_summary_length)

    return summarization_pipeline(input_text, max_length=60, min_length=10)
