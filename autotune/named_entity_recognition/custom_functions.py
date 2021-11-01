import re
import requests
from sklearn.model_selection import train_test_split
from torch_snippets import logger, P, os, makedir
from torch_snippets.registry import registry

from torch_snippets.registry import registry

registry.create('prepare_dataset')

@registry.prepare_dataset.register("read_conll")
def wrapper(seed, url, folder, test_size):
    def prepare_dataset():
        makedir(folder)
        fname = url.split('/')[-1]
        r = requests.get(url, allow_redirects=True)
        fpath = f'{folder}/{fname}'
        if not os.path.exists(fpath):
            with open(fpath, 'wb') as f:
                f.write(r.content)
            logger.info(f'Downloaded file to {fpath}')
        else:
            logger.info(f'Using the cahed file at {fpath}')
        texts, tags = read_conll(fpath)
        unique_tags = set(tag for doc in tags for tag in doc)
        tag2id = {tag: id for id, tag in enumerate(unique_tags)}
        id2tag = {id: tag for tag, id in tag2id.items()}
        train_texts, val_texts, train_tags, val_tags = (
                train_test_split(texts, tags, test_size=test_size))
        return (
            train_texts, val_texts,
            train_tags, val_tags,
            unique_tags, tag2id, id2tag
        )
    return prepare_dataset

def read_conll(file_path):
    file_path = P(file_path)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            token, tag = line.split('\t')
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)
    return token_docs, tag_docs   
