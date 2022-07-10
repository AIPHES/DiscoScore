from __future__ import absolute_import, division, print_function
import torch

import nltk

def bert_encode(model, x, attention_mask):
    model.eval()
    with torch.no_grad():
        output = model(input_ids = x, token_type_ids = None, attention_mask = attention_mask)
        return output.last_hidden_state

      
def data_process(text, tokenizer, device='cuda:0'):

    token_ids = []
    sentids = []            
    sents = nltk.tokenize.sent_tokenize("[CLS] " + text)
    
    cur_id = 1
    for s in sents:
        for w in s.split():
            word_pieces = tokenizer.tokenize(w)
            word_pieces_ids = tokenizer.convert_tokens_to_ids(word_pieces)
            for _, idx in zip(word_pieces, word_pieces_ids):
                token_ids.append(idx)
                sentids.append(cur_id)
        cur_id += 1                        
                 
    padded = [token_ids]
    
    mask = torch.zeros(1, len(token_ids), dtype=torch.long)
    
    padded = torch.LongTensor(padded)
    padded = padded.to(device=device)
    
    padded_sentids = torch.LongTensor(sentids)    
    padded_sentids = padded_sentids.to(device=device)
    
    mask = mask.to(device=device)
        
    return padded, mask, padded_sentids

def get_bert_embedding(text, model, tokenizer,
                       batch_size=-1, device='cuda:0'):

    padded_sents, mask, padded_sentids = data_process(text, tokenizer, device=device)

    embeddings = bert_encode(model, padded_sents, attention_mask=mask)

    return embeddings, mask, padded_sentids


def get_embeddings(model, tokenizer, text, text_u_a, device='cuda:0'):

    text_embedding, text_masks, text_padded_sentids = get_bert_embedding(text, model, tokenizer, 
                                   device=device)                      
    
    text_embedding = text_embedding[-1]
    
    def _aggregation_to_sent_embedding(positions, data):
        # https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
        M = torch.zeros(positions.max().int(), len(data)).cuda()
        M[positions.long() - 1, torch.arange(len(data))] = 1 # sentid starting from 1
        M = torch.nn.functional.normalize(M, p=1, dim=1)
        return torch.mm(M, data)
    
    text_embedding = _aggregation_to_sent_embedding(text_padded_sentids, text_embedding)
 
    text_u_a = torch.from_numpy(text_u_a).to(device=device).float()    
            
    text_embedding = torch.mm(text_u_a, text_embedding)
    
    return text_embedding