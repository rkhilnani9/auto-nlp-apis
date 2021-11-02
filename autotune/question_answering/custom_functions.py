import requests, json
from sklearn.model_selection import train_test_split
from torch_snippets import logger, P, os, makedir
from torch_snippets.registry import registry

from torch_snippets.registry import registry

registry.create('download_data')

@registry.download_data.register("read_squad")
def wrapper(url, folder):
    def read_squad():
        makedir(folder)
        fname = url.split('/')[-1]
        fpath = f'{folder}/{fname}'
        download(url, fpath)

        with open(fpath, 'rb') as f:
            squad_dict = json.load(f)

        contexts = []
        questions = []
        answers = []
        for group in squad_dict['data']:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    for answer in qa['answers']:
                        contexts.append(context)
                        questions.append(question)
                        answers.append(answer)

        add_end_idx(answers, contexts)
        return contexts, questions, answers
    return read_squad

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters


def download(url, fpath):
    r = requests.get(url, allow_redirects=True)
    if not os.path.exists(fpath):
        with open(fpath, 'wb') as f:
            f.write(r.content)
        logger.info(f'Downloaded file to {fpath}')
    else:
        logger.info(f'Using the cahed file at {fpath}')

