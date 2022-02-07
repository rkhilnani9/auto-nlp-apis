from transformers import pipeline
from autonlp.question_answering.model import get_model, get_tokenizer

model = get_model()
tokenizer = get_tokenizer()


def infer(question, context):
    question_answering_pipeline = pipeline(
        "question-answering", model=model, tokenizer=tokenizer
    )
    return question_answering_pipeline(question.lower(), context.lower())
