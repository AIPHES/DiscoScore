import tqdm
import pandas as pd
import json
import numpy as np

def load_json(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))  
    return data

class SUMMEval():

    def __init__(self, sys_path):
        
        self.sys_path = sys_path

    def sort_out_data(self, orgin_data, num_sys, num_batches):
        data = {}                 
        for i in range(0, num_batches):
            per_instance = orgin_data[i * num_sys: (i+1) * num_sys]
            
            sid = set([_['id'] for _ in per_instance])
            assert len(sid) == 1, "the set should contain only one instance_id"    
            sid = sid.pop() 
            
            data[sid] = {}
            for j in range(0, num_sys):                
                data[sid][per_instance[j]['model_id']] = per_instance[j]
            
            if i == num_batches - 1:
                system_names = list(data[sid].keys())
        return data, system_names  

    def generate_system_scores(self, num_instances, eval_metric):               
        
        num_sys = 16
        dataset = load_json(self.sys_path)        
        dataset, system_names = self.sort_out_data(dataset, num_sys, int(len(dataset) / num_sys))

        if num_instances == -1:
            num_instances = len(dataset) 

        df = np.zeros([num_instances, num_sys])
        
        for i, (sid, data) in enumerate(tqdm.tqdm(dataset.items(), total=len(dataset))):            
            
            system_names.sort(key= lambda x: float(x.strip('M')))
            
            for j, system in enumerate(system_names): 
                
                # only consider abstractive systems
                if system in ['M' + str(_) for _ in range(8)]:continue
            
                s, references = data[system]['decoded'], data[system]['references']                
                df[i][j] = eval_metric(s.lower(), [r.lower() for r in references])

        df = pd.DataFrame(data = df, columns = ['M' + str(k) for k in range(num_sys)])
        return df
    
    def generate_human_scores(self, num_instances, qa):
        
        num_sys = 16
        dataset = load_json(self.sys_path)               
        dataset, system_names = self.sort_out_data(dataset, num_sys, int(len(dataset) / num_sys))

        if num_instances == -1:
            num_instances = len(dataset) 

        df = np.zeros([num_instances, num_sys])
    
        for i, (sid, data) in enumerate(dataset.items()):     
                                
            system_names.sort(key= lambda x: float(x.strip('M')))
            
            for j, system in enumerate(system_names): 
                
                # only consider abstractive systems
                if system in ['M' + str(_) for _ in range(8)]:continue 
                
                human_scores = data[system]['expert_annotations']
                df[i][j] = np.mean([v[qa] for v in human_scores])
                
        df = pd.DataFrame(data = df, columns = ['M' + str(k) for k in range(num_sys)])
                
        return df


import argparse
from disco_score.metrics.metrics import Metrics
from scipy.stats import kendalltau                        
    
parser = argparse.ArgumentParser()
parser.add_argument("--m", default='DS_SENT_NN', type=str, help="specify the name of discourse metric")
parser.add_argument("--data_path", default='../model_annotations.aligned.jsonl', type=str, help="the path of SUMMEval data")
parser.add_argument("--we", type=str, help="the path of static word embeddings") # our experiments use deps.words.bz2

if __name__ == "__main__":    
    args = parser.parse_args()

    summ_task = SUMMEval(sys_path = args.data_path)
    
    df_mt = summ_task.generate_system_scores(-1, getattr(Metrics(args = args), args.m))
    df_mt = df_mt.to_numpy()[:, 4:] # only consider abstractive systems
    
    output = args.m + ': '
    all_corr = []
    for qa in ['coherence', 'consistency', 'fluency', 'relevance']:
        df_ht = summ_task.generate_human_scores(-1, qa=qa)
        df_ht = df_ht.to_numpy()[:, 4:]
        
        corr, _ = kendalltau(np.mean(df_mt, axis=0), np.mean(df_ht, axis=0))
        output += "{:.2f}".format(abs(corr) * 100) + ' & '
        all_corr.append(abs(corr))
    
    avg_corr = np.mean(all_corr)
    print(output + "{:.2f}".format(avg_corr * 100))

