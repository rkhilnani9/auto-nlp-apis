import re
import requests
from sklearn.model_selection import train_test_split
from torch_snippets import logger, P, os, makedir
from torch_snippets.registry import registry

from torch_snippets.registry import registry

registry.create('download_vocabulary')

@registry.download_vocabulary.register("download_vocabulary")
def wrapper(url, folder):
    def prepare_dataset():
        makedir(folder)
        fname = url.split('/')[-1]
        fpath = f'{folder}/{fname}'
        download(url, fpath)
        texts = []
        with open(fpath, "r") as file:
            text = file.readlines()
            text = [t.strip("\n") for t in text]
            texts.extend(text)
        return texts
    return prepare_dataset

def download(url, fpath):
    r = requests.get(url, allow_redirects=True)
    if not os.path.exists(fpath):
        with open(fpath, 'wb') as f:
            f.write(r.content)
        logger.info(f'Downloaded file to {fpath}')
    else:
        logger.info(f'Using the cahed file at {fpath}')
