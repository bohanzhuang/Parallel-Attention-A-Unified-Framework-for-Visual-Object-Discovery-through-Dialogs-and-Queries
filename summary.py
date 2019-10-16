import numpy as np 
import torch
import torch.nn as nn
import json
from data import Corpus
import pickle



train_info = json.load(open('../data/train_data_summary.json', 'r'))
num = len(train_info.keys())
corpus = Corpus()


for ii in range(num):
    info = train_info[train_info.keys()[ii]]['question']
    corpus.tokenize(info)
    
    
aa = corpus.dictionary.word2idx
print len(aa.keys())
wordcount = corpus.dictionary.wordcount

new_dict = {key: value for key, value in wordcount.items()
            if value >= 10}


print len(new_dict.keys())
pickle.dump(new_dict, open('word_dict.pkl', 'w'))
