import os
import re
import imp
import collections
import numpy as np
import urllib.request
import tensorflow as tf
from nltk import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

DATA_DIR = "./data"

def load_20newgroups(vocab_size, max_seq_len):
    train_text, train_labels = load_data(DATA_DIR + '/train.txt')
    test_text, test_labels = load_data(DATA_DIR + '/test.txt')

    return get_BOW(train_text, test_text)

def load_data(path):
    text = []
    labels = []
    with open(path) as f:
        for line in f:
            labels.append(line.split('\t')[0])
            text.append(line.split('\t')[1])
    return text, labels

def get_BOW(train_text, test_text):
    vectorizer = CountVectorizer()
    vectorizer.fit(train_text + test_text)
    train_bow = vectorizer.transform(train_text).toarray().astype(np.float32)
    test_bow = vectorizer.transform(test_text).toarray().astype(np.float32)

    return train_bow, test_bow


