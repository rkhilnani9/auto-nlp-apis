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
import pytorch_lightning as pl
import transformers
# from transformers import T5Tokenizer, T5ForConditionalGeneration
TokenizerModule = getattr(transformers, config.model.tokenizer)
ModuleForQuestionAnswering = getattr(transformers, config.model.question_answering_module)
# from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset

train_contexts, train_questions, train_answers = config.data.download.train()
val_contexts, val_questions, val_answers = config.data.download.validate()

# Get ending integer index for each answer -  SQuaD only has starting indices
tokenizer = TokenizerModule.from_pretrained(config.model.pretrained_tokenizer_model)

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(val_encodings, val_answers)

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

model = ModuleForQuestionAnswering.from_pretrained(config.model.pretrained_question_answering_model)

model_path = config.training.checkpoint_progress_directory

training_args = TrainingArguments(
    output_dir=model_path,          # output directory
    num_train_epochs=1, 
    evaluation_strategy="epoch",            # total number of training epochs
    per_device_train_batch_size=config.training.batch_size,
    per_device_eval_batch_size=config.training.batch_size,
)

trainer = Trainer(
    model=model,                         
    args=training_args,                 
    train_dataset=train_dataset,        
    eval_dataset=val_dataset             
)

trainer.train()

trainer.evaluate()

trainer.save_model(config.training.model_save_path)
