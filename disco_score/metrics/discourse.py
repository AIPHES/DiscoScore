#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 22:24:22 2021

@author: zhaowi
"""
import itertools
import numpy as np
import spacy_udpipe
from nltk import tokenize
import networkx as nx
import torch
from torch import nn

spacy_udpipe.download("en")
udpipe = spacy_udpipe.load("en")

def get_nouns_dict(sents, is_lexical_graph=False, we=None, threshold=0.5):
    nouns_dict = {}
    for i, s in enumerate(sents):
        for token in udpipe(s):
            if token.pos_ in ['PROPN', 'NOUN']:
                token_str = token.lemma_
                if token_str not in nouns_dict:
                    nouns_dict[token_str] = []
                nouns_dict[token_str] = list(nouns_dict[token_str] + [i])
    if is_lexical_graph and we:
        edges = []
        for a, b in itertools.combinations(list(nouns_dict.keys()), 2):
            if a in we and b in we:
                w_a = we[a]
                w_b = we[b]
                sim = np.dot(w_a, w_b) / (np.linalg.norm(w_a) * np.linalg.norm(w_b))
                if sim > threshold:
                    edges.append([a,b])
        graph = nx.Graph(edges) 
        groups = [tuple(c) for c in nx.connected_components(graph)]
        for g in groups:
            new = []
            for g_i in g:
                new += nouns_dict.pop(g_i)
            nouns_dict['-'.join(g)] = new    
    return nouns_dict

def RemoveBadLexicalChain(grid):
    new_grid = {}
    for n, lc in grid.items():
        if len(set(lc)) > 1:
            new_grid[n] = lc  
    return new_grid

def LexicalChain(sys, ref):
    
    sys_entity_grid = RemoveBadLexicalChain(get_nouns_dict(tokenize.sent_tokenize(sys)))
    sys_nouns = list(sys_entity_grid.keys())
    K = len(sys_nouns) # the number of valid lexical chains in a hypothesis
    
    if K == 0: return 0
    
    scores = []
    for r in ref:
        cohesion = 0
        r_entity_grid = RemoveBadLexicalChain(get_nouns_dict(tokenize.sent_tokenize(r)))
        for n in sys_nouns:
            if n in r_entity_grid:
                sys_lc = set(sys_entity_grid[n])
                r_lc = set(r_entity_grid[n]) 
                
                common_seq = sys_lc & r_lc
                precision = len(common_seq) / len(sys_lc)
                recall = len(common_seq) / len(r_lc)
                cohesion += recall
        scores.append(cohesion)  
        
    return np.max(scores)

def RC(sys):
    entity_grid = get_nouns_dict(tokenize.sent_tokenize(sys))
    ms_repetitions_cnt = sum([len(v) for k, v in entity_grid.items() if len(v) > 1])
    score = ms_repetitions_cnt / len(entity_grid)
    return score

def LC(sys):
    entity_grid = get_nouns_dict(tokenize.sent_tokenize(sys))
    repetitions_token_cnt = len([k for k, v in entity_grid.items() if len(v) > 1])    
    score = repetitions_token_cnt / len(entity_grid)
    return score

def EntityGraph(sys, is_lexical_graph=False, we=None, threshold=0):
    # https://github.com/aylai/DiscourseCoherence/blob/main/extract_graph_from_grid.py
    
    sents = tokenize.sent_tokenize(sys)
    entity_grid = get_nouns_dict(sents, is_lexical_graph, we, threshold)
    
    nouns = list(entity_grid.keys())
    num_sentences = len(sents)

    u_adjacency = np.zeros([num_sentences, num_sentences])
    w_adjacency = np.zeros([num_sentences, num_sentences])
    u_adjacency_dist = np.zeros([num_sentences, num_sentences])
    w_adjacency_dist = np.zeros([num_sentences, num_sentences])
    
    for n in nouns:
        idx = entity_grid[n]
        idx = set(idx)
        for pair in itertools.combinations(idx, 2):
            i = min(pair)
            j = max(pair)
            u_adjacency[i][j] = 1
            w_adjacency[i][j] += 1
            u_adjacency_dist[i][j] = 1/(j-i)
            w_adjacency_dist[i][j] += 1/(j-i)
    
    return u_adjacency_dist, num_sentences    

def _safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-30)


def DS_Focus(model, tokenizer, sys, ref, is_semantic_entity=False, we=None, threshold=0):
    
    from disco_score.metrics.focus_diff import get_embeddings
    
    scores = []
    s_embedding, s_noun_positions = get_embeddings(model, tokenizer, sys, is_semantic_entity, 
                                                   we, threshold)    
    for r in ref:

        r_embedding, r_noun_positions = get_embeddings(model, tokenizer, r, is_semantic_entity, 
                                                       we, threshold)
        _, s_ind, r_ind = np.intersect1d(list(s_noun_positions.keys()),
                                  list(r_noun_positions.keys()), 
                                  return_indices=True)
        
        if len(s_ind) == 0: 
            scores.append(0)
            continue
            
        diff = torch.sum((s_embedding[s_ind, :] - r_embedding[r_ind, :])**2, 1)    
        P = diff.sum().item() / (len(s_noun_positions) + 1e-9)
        R = diff.sum().item() / (len(r_noun_positions) + 1e-9)    
        F = (P + R)/2
        scores.append(P)        
    return np.mean(scores)


def DS_Sent(model, tokenizer, sys, ref, is_lexical_graph=False, we=None, threshold=0):
    
    from disco_score.metrics.sent_graph import get_embeddings
    
    scores = []
    sys_u_a, num_sentences = EntityGraph(sys, is_lexical_graph, we, threshold)    
    sys_u_a = np.identity(num_sentences) + sys_u_a

    s_embedding = get_embeddings(model, tokenizer, sys, sys_u_a)    
    s_embedding.div_(torch.norm(s_embedding, dim=-1).unsqueeze(-1) + 1e-30)

    for r in ref:
        
        ref_u_a, num_sentences = EntityGraph(r, is_lexical_graph, we, threshold)
        ref_u_a = np.identity(num_sentences) + ref_u_a
                
        r_embedding = get_embeddings(model, tokenizer, r, ref_u_a)
        r_embedding.div_(torch.norm(r_embedding, dim=-1).unsqueeze(-1) + 1e-30) 
    
            
        if len(r_embedding) == 1:
            score = nn.functional.cosine_similarity(r_embedding, s_embedding, dim=1).mean().item()
            scores.append(score)
        else:
            r_embedding_max, _ = torch.max(r_embedding, dim=0, out=None)
            s_embedding_max, _ = torch.max(s_embedding, dim=0, out=None)
    
            r_embedding_min, _ = torch.min(r_embedding, dim=0, out=None)
            s_embedding_min, _ = torch.min(s_embedding, dim=0, out=None)

            r_embedding_avg = r_embedding.mean(0)
            s_embedding_avg = s_embedding.mean(0)

            r_embedding_sum = r_embedding.sum(0)
            s_embedding_sum = s_embedding.sum(0)
            
            r_embedding_new = torch.cat([r_embedding_min, r_embedding_avg, r_embedding_max, r_embedding_sum], -1)
            s_embedding_new = torch.cat([s_embedding_min, s_embedding_avg, s_embedding_max, s_embedding_sum], -1)

            score = nn.functional.cosine_similarity(r_embedding_new, s_embedding_new, dim=0).item()
        scores.append(score)
            
    return np.mean(scores)