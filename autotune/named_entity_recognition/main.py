#!/usr/bin/env python
# coding: utf-8

import os
from custom_functions import *
from torch_snippets.registry import parse_and_resolve
config = parse_and_resolve(os.environ['AUTOTUNE_CONFIG'])

import pandas as pd
import numpy as np
import re
import sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split


import torch
import pytorch_lightning as pl
import importlib

import transformers
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset
TokenizerModule = getattr(transformers, config.model.tokenizer)
ModelForTokenClassification = getattr(transformers, config.model.classifier)

(
    train_texts, val_texts,
    train_tags, val_tags,
    unique_tags, tag2id, id2tag) = config.data.prepare()

# Load pre-trained DistilBertTokenizer
tokenizer = TokenizerModule.from_pretrained(config.model.pretrained_tokenizer_model)

train_encodings = tokenizer(train_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True)

def encode_tags(tags, encodings):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        # create an empty array of -100
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())

    return encoded_labels

train_labels = encode_tags(train_tags, train_encodings)
val_labels = encode_tags(val_tags, val_encodings)

class WNUTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_encodings.pop("offset_mapping") # we don't want to pass this to the model
val_encodings.pop("offset_mapping")
train_dataset = WNUTDataset(train_encodings, train_labels)
val_dataset = WNUTDataset(val_encodings, val_labels)

model = ModelForTokenClassification.from_pretrained(config.model.pretrained_classifier_model, num_labels=len(unique_tags))

# Train the model
idx = 0
model_path = config.training.checkpoint_progress_directory

training_args = TrainingArguments(
    output_dir=model_path,          # output directory
    num_train_epochs=config.training.n_epochs, 
    evaluation_strategy="epoch"             # total number of training epochs
)

# Trainer object 

trainer = Trainer(
    model=model,                         
    args=training_args,                 
    train_dataset=train_dataset,        
    eval_dataset=val_dataset             
)

trainer.train()

trainer.evaluate()

trainer.save_model(config.training.model_save_path)
