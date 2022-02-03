import pandas as pd
import io

from classification.utils import *
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import config


def train_model(dataframe):
    data = pd.read_csv(dataframe.file, index_col=0)
    data.columns = ["text", "label"]

    x = data["text"].values
    y = data["label"].values

    # Split into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(x, y)

    # Load pre-trained AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL)

    # Tokenize
    train_tokens = tokenizer(
        list(train_data),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    val_tokens = tokenizer(
        list(val_data),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    train_dataset = ClassificationTrainDataset(train_tokens, train_labels)
    val_dataset = ClassificationTrainDataset(val_tokens, val_labels)

    # Train the model
    training_args = TrainingArguments(
        output_dir=config.CLS_MODEL_SAVE_PATH,  # output directory
        num_train_epochs=config.NUM_EPOCHS,
        evaluation_strategy="epoch",  # total number of training epochs
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.PRETRAINED_MODEL
    )

    # Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    trainer.save_model(config.MODEL_SAVE_PATH)

    return
