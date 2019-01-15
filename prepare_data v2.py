#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Roman Moser, 1/12/19

"""
Quora Insincere Questions - prepare data with word2vec
run with: python3 main.py 
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import os
os.chdir('/home/roman/Documents/Projects/Quora/data')

def main():
    # load data:
    data = pd.read_csv('train.csv')
    X_train, X_val, y_train, y_val = train_test_split(data['question_text'], \
        data['target'], test_size=0.1, random_state=123)
    y_train_np = np.array(y_train)
    y_val_np = np.array(y_val)
    np.save('y_train', y_train_np)
    np.save('y_val', y_val_np)
    # Prepare data with Word2Vec embedding:
    # load w2v model
    os.chdir('../models')
    size = 150
    filename = 'w2v_w5_s150'
    model = Word2Vec.load(filename + '.model')
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))

    X_train_texts = [[word for word in document.lower().split()] for document in X_train]  
    X_val_texts = [[word for word in document.lower().split()] for document in X_val]
    
    X_train_texts_w2v_mean = np.array([np.mean([w2v[word] if word in w2v else \
            np.zeros(size) for word in text], axis=0) for text in X_train_texts])
    X_val_texts_w2v_mean = np.array([np.mean([w2v[word] if word in w2v else \
            np.zeros(size) for word in text], axis=0) for text in X_val_texts])
    os.chdir('../data')
    np.save('X_train_' + filename, X_train_texts_w2v_mean)
    np.save('X_val_' + filename, X_val_texts_w2v_mean)

if __name__ == '__main__':
        main()
