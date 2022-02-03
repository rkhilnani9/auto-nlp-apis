import numpy as np
import pandas as pd
import config

from classification.utils import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer


def infer(dataframe):
    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_SAVE_PATH + "checkpoint-500/")
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL)

    data = pd.read_csv(dataframe.file, index_col=0)
    data.columns = ["text"]
    x = data["text"].values

    test_tokens = tokenizer(
        list(x),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )

    test_dataset = ClassificationTestDataset(test_tokens)

    trainer = Trainer(model=model)

    pred_probs = trainer.predict(test_dataset)[0]

    predicted_labels = np.argmax(pred_probs, axis=1)

    return {"predictions" : list(predicted_labels)}


