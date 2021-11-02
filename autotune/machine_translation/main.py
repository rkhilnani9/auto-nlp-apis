#!/usr/bin/env python
# coding: utf-8

import os
from custom_functions import *
from torch_snippets.registry import parse_and_resolve
config = parse_and_resolve(os.environ['AUTOTUNE_CONFIG'])

import pandas as pd
import sklearn
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
import transformers
# from transformers import T5Tokenizer, T5ForConditionalGeneration
TokenizerModule = getattr(transformers, config.model.tokenizer)
ConditionalGeneration = getattr(transformers, config.model.conditional_generation)
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset

# In[3]:
# Input - Two list of sentences
english = config.data.download.english()
french = config.data.download.french()

data = pd.DataFrame({"english" : english, "french" : french})

# In[4]:
tokenizer = TokenizerModule.from_pretrained("t5-small")
x = data["english"].values.tolist()
y = data["french"].values.tolist()
train_x, val_x, train_y, val_y = train_test_split(x, y)

train_encodings = tokenizer(train_x, padding=True, truncation=True)
val_encodings = tokenizer(val_x, padding=True, truncation=True)
with tokenizer.as_target_tokenizer():
    train_labels = tokenizer(
            train_y, padding=True, truncation=True, return_tensors="pt")
    val_labels = tokenizer(
            val_y, padding=True, truncation=True, return_tensors="pt")

# In[ ]:
# Dataset class
class MTDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels["input_ids"][idx]
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = MTDataset(train_encodings, train_labels)
val_dataset = MTDataset(val_encodings, val_labels)

model = ConditionalGeneration.from_pretrained(config.model.pretrained_classifier_model)

model_path = config.training.checkpoint_progress_directory

training_args = TrainingArguments(
    output_dir=model_path,          # output directory
    num_train_epochs=1, 
    evaluation_strategy="epoch",
    per_device_train_batch_size = 1          
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

