#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 1/12/19

"""
Quora Insincere Questions - create embedding matrix from w2v model
run with: python3 main.py 
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
import os
os.chdir('/home/roman/Documents/Projects/Quora/data')

def main():
    # load data:
    data = pd.read_csv('train.csv')
    
    # tokenize (words to index)
    tk = Tokenizer(lower = True, filters='')
    full_text = list(data['question_text'].values)
    tk.fit_on_texts(full_text)
    word_index = tk.word_index # dict: key: word, value: index
    
    # Prepare data with Word2Vec embedding:
    # load w2v model
    os.chdir('../models')
    filename = 'w2v_w5_s100'
    model = Word2Vec.load(filename + '.model')
    w2v_dict = dict(zip(model.wv.index2word, model.wv.syn0))
    embed_size = 100
    max_features = 30000
    nb_words = min(max_features, len(w2v_dict))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = w2v_dict.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    os.chdir('../data')
    np.save('embedding_matrix', embedding_matrix)

if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    