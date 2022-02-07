from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typer import Typer

from autonlp.classification.main import router as classification_router
from autonlp.ner.main import router as ner_router
from autonlp.question_answering.main import router as question_answering_router
from autonlp.fill_mask.main import router as fill_mask_router
from autonlp.summarization.main import router as summarization_router

ALLOWED_TASKS = [
    'classification',
    'named_entity_recognition',
    'question_answering',
    'fill_mask'
    'summarization'
]


app = FastAPI()
app.include_router(classification_router, prefix='/classification')
app.include_router(ner_router, prefix='/named_entity_recognition')
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
    if task == 'classification':
        from autonlp.classification.infer import infer as classification_infer
        classification_infer(text)

    if task == 'named_entity_recognition':
        from autonlp.ner.infer import infer as ner_infer
        ner_infer(text)

    if task == 'question_answering':
        from autonlp.question_answering.infer import infer as question_answering_infer
        question_answering_infer(text, context)

    if task == "fill_mask":
        from autonlp.fill_mask.infer import infer as fill_mask_infer
        fill_mask_infer(text)

    if task == "summarization":
        from autonlp.summarization.infer import infer as summarization_infer
        summarization_infer(text)

@cli.command()
def train(task=None, dataframe=None):
    assert task in ALLOWED_TASKS, f'task should be one of {ALLOWED_TASKS}'
    if task == 'classification':
        from autonlp.classification.train import train_model as train_classifier
        train_classifier(dataframe)
    elif task == 'named_entity_recognition':
        from autonlp.ner.train import train_model as train_ner_model
        train_ner_model(dataframe)



@cli.command()
def predict(task=None, dataframe=None):
    assert task in ALLOWED_TASKS, f'task should be one of {ALLOWED_TASKS}'
    if task == 'classification':
        from autonlp.classification.predict import predict as classification_predict
        preds = classification_predict(dataframe)
    elif task == 'named_entity_recognition':
        from autonlp.ner.predict import predict as ner_predict
        preds = ner_predict(dataframe)


if __name__ == '__main__':
    cli()
