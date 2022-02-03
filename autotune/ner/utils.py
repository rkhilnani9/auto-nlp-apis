import pandas as pd
import itertools
import io

import config
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL)


def get_tokens_and_ner_tags(lines):
    lines = lines.readlines()
    lines = [line.decode('utf-8') for line in lines]
    split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
    tokens = [[x.split('\t')[0] for x in y] for y in split_list]
    entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list]

    all_entities = list(itertools.chain(*entities))
    all_entities = list(set(all_entities))
    all_entities.remove('')
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities}), all_entities


def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True)
    label_encoding_dict = examples["label_encoding_dict"][0]

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


