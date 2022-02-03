from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typer import Typer

from classification.main import router as classification_router
from ner.main import router as ner_router

ALLOWED_TASKS = [
    'classification',
    'named_entity_recognition'
]

app = FastAPI()
app.include_router(classification_router, prefix='/classification')
app.include_router(ner_router, prefix='/named_entity_recognition')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

cli = Typer()

@cli.command()
def train(task=None, dataframe=None):
    assert task in ALLOWED_TASKS, f'task should be one of {ALLOWED_TASKS}'
    if task == 'classification':
        from classification.train import train_model as train_classifier
        train_classifier(dataframe)
    elif task == 'named_entity_recognition':
        from ner.train import train_model as train_ner_model
        train_ner_model(dataframe)



@cli.command()
def validate(task=None, dataframe=None):
    assert task in ALLOWED_TASKS, f'task should be one of {ALLOWED_TASKS}'
    if task == 'classification':
        from classification.infer import infer as classification_infer
        preds = classification_infer(dataframe)
    elif task == 'named_entity_recognition':
        from ner.infer import infer as ner_infer
        preds = ner_infer(dataframe)


if __name__ == '__main__':
    cli()
