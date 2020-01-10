import pandas as pd
from nltk.tokenize import TweetTokenizer
import os

path_to_words = {}
word_to_path = {}
tknzr = TweetTokenizer(preserve_case=True, reduce_len=True)

def load():
    with open(os.path.join(os.path.dirname(__file__),'50mpaths2.csv'), "r") as f:
        for l in f:
            r = l.rstrip().split('\t')
            path = r[0]
            word = r[1]
            count = int(r[2])
            if path not in path_to_words:
                path_to_words[path] = []

            path_to_words[path].append([word, count])
            word_to_path[word] = path
    for k in path_to_words:
        path_to_words[k].sort(key=lambda x: x[1], reverse=True)


def get_path_summary(path, n_words=10):
    if len(path_to_words) == 0:
        load()
    if path not in path_to_words:
        return path
    words = path_to_words[path]
    n = min(len(words), n_words)
    return ', '.join([a[0] for a in words[:n]])



def get_words_for_path(path):
    if len(path_to_words) == 0:
        load()
    return [w[0] for w in path_to_words[path]]


def get_path_for_word(word):
    if len(path_to_words) == 0:
        load()
    if word in word_to_path:
        return word_to_path[word]
    else:
        return word

def tokenize_and_tag(text):
    if len(path_to_words) == 0:
        assert 'Brown clusters not loaded'
    # print(text)
    return [word_to_path[t] for t in tknzr.tokenize(text) if t in word_to_path]


