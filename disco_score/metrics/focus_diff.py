from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import networkx as nx
import itertools

import nltk

import spacy_udpipe
        
spacy_udpipe.download("en")
udpipe = spacy_udpipe.load("en")


def get_bert_embedding(model, tokenizer, text, 
                       is_semantic_entity=False, we=None, threshold = 0, device='cuda:0'):

    padded_sents, mask, noun_positions = data_process(text,
                                                      tokenizer,
                                                      is_semantic_entity, we, threshold,
                                                      device=device)

    embeddings = bert_encode(model, padded_sents, attention_mask=mask)
    return embeddings, mask, noun_positions

def bert_encode(model, x, attention_mask):
    model.eval()
    with torch.no_grad():
        output = model(input_ids = x, token_type_ids = None, attention_mask = attention_mask)
        return output.last_hidden_state

def get_nouns(sents):
    nouns = []
    for s in sents:
        for token in udpipe(s):
            if token.pos_ in ['PROPN', 'NOUN']:
                nouns.append(token.lemma_)
    return list(set(nouns))

def data_process(text, tokenizer, is_semantic_entity=False, we=None, threshold = 0,
                device='cuda:0'):

    nouns = get_nouns(nltk.tokenize.sent_tokenize(text))
    token_ids = []
    nn_pos = {}
    
    pos = 0
    for word in ["[CLS]"] + text.split():
        word_pieces = tokenizer.tokenize(word)
        word_pieces_ids = tokenizer.convert_tokens_to_ids(word_pieces)
            
        w_pos = []
        for _, idx in zip(word_pieces, word_pieces_ids):
            token_ids.append(idx)
            w_pos.append(pos)
            pos += 1 

        if word not in nouns: continue
        if word not in nn_pos: nn_pos[word] = []
        
        nn_pos[word] = list(nn_pos[word] + [w_pos[-1]]) # only consider the last wordpiece for each word
        
    if is_semantic_entity and we:
        edges = []
        for a, b in itertools.combinations(list(nn_pos.keys()), 2):
            if a in we and b in we:
                w_a = we[a]
                w_b = we[b]
                sim = np.dot(w_a, w_b) / (np.linalg.norm(w_a) * np.linalg.norm(w_b))
                if sim > threshold:
                    edges.append([a,b])
        graph = nx.Graph(edges) 
        entities = [tuple(c) for c in nx.connected_components(graph)]
        for e in entities:
            new = []
            e = list(e)
            e.sort()
            for e_i in e:
                new += nn_pos.pop(e_i)
            nn_pos['-'.join(e)] = new  
                    
    padded = [token_ids]
    
    mask = torch.zeros(1, len(token_ids), dtype=torch.long)
    
    padded = torch.LongTensor(padded)
    padded = padded.to(device=device)
    mask = mask.to(device=device)
    
    return padded, mask, nn_pos

def get_embeddings(model, tokenizer, text, is_semantic_entity=False, we=None, threshold = 0, device='cuda:0'):
    
    text_embedding, text_masks, text_noun_positions = get_bert_embedding(model, tokenizer, text,
                                                        is_semantic_entity, we, threshold,
                                                        device=device)
    
    text_embedding = text_embedding[-1]
    
    text_embedding.div_(torch.norm(text_embedding, dim=-1).unsqueeze(-1) + 1e-30) 
    
    text_adjacency = np.zeros((len(text_noun_positions), len(text_embedding)), dtype=int)
        
    for i, k in enumerate(list(text_noun_positions.keys())): 
        for j in text_noun_positions[k]:
            text_adjacency[i, j] = 1
        
    text_adjacency = torch.from_numpy(text_adjacency).to(device=device).float()
    text_noun_embedding = torch.mm(text_adjacency, text_embedding)
    
    return text_noun_embedding, text_noun_positions