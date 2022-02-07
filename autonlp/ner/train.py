import random
import io

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
from autonlp import config
from autonlp.ner.utils import *


def train_model(data):
    lines = io.BytesIO(data.file.read())
    df, label_list = get_tokens_and_ner_tags(lines)

    label_encoding_dict = {}
    for idx, label in enumerate(label_list):
        label_encoding_dict[label] = idx

    df["label_encoding_dict"] = [label_encoding_dict]*df.shape[0]

    num_rows = list(df.index)
    train_rows = random.sample(num_rows, int(df.shape[0]*0.8))

    train_df = df[df.index.isin(train_rows)]
    val_df = df[~df.index.isin(train_rows)]

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
    val_tokenized_datasets = val_dataset.map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(config.PRETRAINED_MODEL, num_labels=len(label_list))
    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    args = TrainingArguments(
        output_dir=config.NER_MODEL_SAVE_PATH,
        evaluation_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        weight_decay=1e-5,
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=val_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

    trainer.save_model(config.NER_MODEL_SAVE_PATH)

    return
