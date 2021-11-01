import requests
from torch_snippets import logger, P, os, makedir
from torch_snippets.registry import registry

from torch_snippets.registry import registry

registry.create('download_function')

@registry.download_function.register("download_sentiment_dataset")
def wrapper(url, folder):
    def download_dataset():
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
        return P(fpath)
    return download_dataset
    