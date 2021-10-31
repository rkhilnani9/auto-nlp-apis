#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !pip install transformers
# !pip install git+https://github.com/PytorchLightning/pytorch-lightning.git@master --upgrade

import os
from custom_functions import *
from torch_snippets.registry import parse_and_resolve
config = parse_and_resolve(os.environ['AUTOTUNE_CONFIG'])

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

import torch
import pytorch_lightning as pl
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader, TensorDataset

downloaded_data = config.data.download()
data = pd.read_csv(downloaded_data, sep="\t", header=None)
data.columns = ["label", "text"]

x = data["text"].values
y = data["label"].values

# Split into training and validation sets
train_data, val_data, train_labels, val_labels = train_test_split(x, y)

# Load pre-trained DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained(config.model.pretrained_model)

# Tokenize
train_tokens = tokenizer(list(train_data), return_tensors="pt", padding=True, truncation=True, max_length=64)
val_tokens = tokenizer(list(val_data), return_tensors="pt", padding=True, truncation=True, max_length=64)


from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ClassificationDataset(train_tokens, train_labels)
val_dataset = ClassificationDataset(val_tokens, val_labels)

# Train the model
training_args = TrainingArguments(
    output_dir=config.training.checkpoint_progress_directory, # output directory
    num_train_epochs=config.training.n_epochs, 
    evaluation_strategy="epoch"             # total number of training epochs
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Trainer object 
trainer = Trainer(
    model=model,                         
    args=training_args,                 
    train_dataset=train_dataset,        
    eval_dataset=val_dataset             
)

trainer.train()


trainer.evaluate()
# get_ipython().system('mkdir here')
trainer.save_model(config.training.model_save_path)

