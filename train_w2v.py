#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 1/12/19

"""
Quora Insincere Questions - train word2vec
run with: python3 main.py 
"""

import pandas as pd
from gensim.models import Word2Vec
import os
os.chdir('/home/roman/Documents/Projects/Quora/data')

def main():
    data = pd.read_csv('train.csv')
    texts = [[word for word in document.lower().split()] for document in data['question_text']]
    os.chdir('../models')
    window = 5
    size = 150
    model = Word2Vec(size=size, window=window, min_count=1, workers=4)
    model.build_vocab(texts, update=False)
    model.train(texts, total_examples=model.corpus_count, epochs=1)
    
    # save and load model:
    model.save("wd2v_w{0}_s{1}.model".format(window, size))

if __name__ == '__main__':
    main()
