from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typer import Typer

from sentiment_analysis.main import router as sentiment_analysis_router
from named_entity_recognition.main import router as named_entity_recognition_router
from question_answering.main import router as question_answering_router
from fill_mask.main import router as fill_mask_router
from summarization.main import router as summarization_router

ALLOWED_TASKS = [
    'sentiment_analysis',
    'named_entity_recognition',
    'question_answering',
    'fill_mask'
    'summarization'
]

app = FastAPI()
app.include_router(sentiment_analysis_router, prefix='/sentiment_analysis')
app.include_router(named_entity_recognition_router, prefix='/named_entity_recognition')
app.include_router(question_answering_router, prefix='/question_answering')
app.include_router(fill_mask_router, prefix='/fill_mask')
app.include_router(summarization_router, prefix='/summarization')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

cli = Typer()


@cli.command()
def validate(task=None, text=None, context=None):
    assert task in ALLOWED_TASKS, f'task should be one of {ALLOWED_TASKS}'
    if task == 'sentiment_analysis':
        from sentiment_analysis.infer import infer as sentiment_analysis_infer
        sentiment_analysis_infer(text)

    if task == 'named_entity_recognition':
        from named_entity_recognition.infer import infer as named_entity_recognition_infer
        named_entity_recognition_infer(text)

    if task == 'question_answering':
        from question_answering.infer import infer as question_answering_infer
        question_answering_infer(text, context)

    if task == "fill_mask":
        from fill_mask.infer import infer as fill_mask_infer
        fill_mask_infer(text)

    if task == "summarization":
        from summarization.infer import infer as summarization_infer
        summarization_infer(text)


if __name__ == '__main__':
    cli()
