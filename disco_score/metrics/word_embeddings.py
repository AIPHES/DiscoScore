#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import io
import bz2

def _convert_to_numpy(vector):
    return np.array([float(x) for x in vector])

def load_embeddings(model_type, filepath, modelpath = None):    
    embedding_model = {}    
    if model_type == 'glove':    
        for line in open(filepath, 'r'):
            tmp = line.strip().split()  
            word, vec = tmp[0], np.array(list(map(float, tmp[1:])), dtype=np.float32)
            if word not in embedding_model:
                embedding_model[word.lower()] = vec
                
    if model_type == 'deps':
        with bz2.BZ2File(filepath, 'rb') as f:
            for line in f:
                line = line.decode('utf-8').rstrip().split(" ")
                word = line[0]
                vec = line[1::]
                embedding_model[word.lower()] = _convert_to_numpy(vec)        

    if model_type == 'ft':
        fin = io.open(filepath, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        embedding_model = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            embedding_model[tokens[0]] = map(float, tokens[1:])       
    
    return embedding_model






