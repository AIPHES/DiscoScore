import argparse
import time
import numpy as np
from metrics.utils import *

# We adapt the evaluation framework of BARTScore to evaluate DiscoScore on NewsRoom.
# See https://github.com/neulab/BARTScore/blob/main/SUM/score.py

class Scorer:

    def __init__(self, args, device='cuda:0'):

        self.args = args
        self.device = device
        self.data = read_pickle(self.args.file_path)
        print(f'Data loaded from {self.args.file_path}.')

        self.sys_names = self.get_sys_names()

        self.single_ref_lines = self.get_single_ref_lines()
        print(f'In a single-reference setting.')

    def get_sys_names(self):
        first_id = list(self.data.keys())[0]
        return list(self.data[first_id]['sys_summs'].keys())

    def get_single_ref_lines(self):
        ref_lines = []
        for doc_id in self.data:
            ref_lines.append(self.data[doc_id]['ref_summ'])
        return ref_lines

    def get_multi_ref_lines(self):
        ref_lines = []
        for doc_id in self.data:
            ref_lines.append(self.data[doc_id]['ref_summs'])
        return ref_lines

    def get_sys_lines(self, sys_name):
        sys_lines = []
        for doc_id in self.data:
            sys_lines.append(self.data[doc_id]['sys_summs'][sys_name]['sys_summ'])
        return sys_lines

    def get_src_lines(self):
        src_lines = []
        for doc_id in self.data:
            src_lines.append(self.data[doc_id]['src'])
        return src_lines

    def save_data(self, path):
        save_pickle(self.data, path)

    def score(self):
        """ metrics: list of metrics """

        if self.args.m == 'bart_score' or self.args.m == 'bart_score_cnn' or self.args.m == 'bart_score_para':
            """ Vanilla BARTScore, BARTScore-CNN, BARTScore-CNN-Para """
            from metrics.bart_score import BARTScorer
            # Set up BARTScore
            if 'cnn' in self.args.m:
                bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
            elif 'para' in self.args.m:
                bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large-cnn')
                bart_scorer.load()
            else:
                bart_scorer = BARTScorer(device=self.device, checkpoint='facebook/bart-large')
            print(f'BARTScore setup finished. Begin calculating BARTScore.')

            start = time.time()
            # Keep capitalization, detokenize everything
            src_lines = self.get_src_lines()
            src_lines = [detokenize(line) for line in src_lines]
            if not self.multi_ref:
                ref_lines = [detokenize(line) for line in self.single_ref_lines]
            else:
                ref_lines = [[detokenize(text) for text in line] for line in self.multi_ref_lines]
            for sys_name in self.sys_names:
                sys_lines = self.get_sys_lines(sys_name)
                sys_lines = [detokenize(line) for line in sys_lines]
                src_hypo = bart_scorer.score(src_lines, sys_lines, batch_size=4)
                if not self.multi_ref:
                    ref_hypo = np.array(bart_scorer.score(ref_lines, sys_lines, batch_size=4))
                    hypo_ref = np.array(bart_scorer.score(sys_lines, ref_lines, batch_size=4))
                else:
                    ref_hypo, hypo_ref = np.zeros(len(sys_lines)), np.zeros(len(sys_lines))
                    for i in range(self.ref_num):
                        ref_list = [x[i] for x in ref_lines]
                        curr_ref_hypo = np.array(bart_scorer.score(ref_list, sys_lines, batch_size=4))
                        curr_hypo_ref = np.array(bart_scorer.score(sys_lines, ref_list, batch_size=4))
                        ref_hypo += curr_ref_hypo
                        hypo_ref += curr_hypo_ref
                    ref_hypo = ref_hypo / self.ref_num
                    hypo_ref = hypo_ref / self.ref_num
                avg_f = (ref_hypo + hypo_ref) / 2
                harm_f = (ref_hypo * hypo_ref) / (ref_hypo + hypo_ref)
                counter = 0
                for doc_id in self.data:
                    self.data[doc_id]['sys_summs'][sys_name]['scores'].update({
                        f'{self.args.m}_src_hypo': src_hypo[counter],
                        f'{self.args.m}_hypo_ref': hypo_ref[counter],
                        f'{self.args.m}_ref_hypo': ref_hypo[counter],
                        f'{self.args.m}_avg_f': avg_f[counter],
                        f'{self.args.m}_harm_f': harm_f[counter]
                    })
                    counter += 1
            print(f'Finished calculating BARTScore, time passed {time.time() - start}s.')

        elif self.args.m.startswith('DS_Focus') or self.args.m.startswith('DS_SENT'):

            from metrics.metrics import Metrics

            eval_metric = getattr(Metrics(args = self.args), self.args.m)                        
            
            print(f'{self.args.m} setup finished. Begin calculating {self.args.m}.')
            start = time.time()
            src_lines = self.get_src_lines()
            ref_lines = self.single_ref_lines
            
            for sys_name in self.sys_names:
                sys_lines = self.get_sys_lines(sys_name)
                hypo_ref = []
                for sys, ref in zip(sys_lines, ref_lines):
                    hypo_ref.append(eval_metric(sys.lower(), [ref.lower()]))
                    
                counter = 0
                for doc_id in self.data:
                    self.data[doc_id]['sys_summs'][sys_name]['scores'].update({
                        f'{self.args.m}_hypo_ref': hypo_ref[counter],
                    })
                    counter += 1
            print(f'Finished calculating BARTScore, time passed {time.time() - start}s.')



def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--file_path', type=str, default='Newsroom/data.pkl',
                        help='The data to load from.')
    parser.add_argument('--output', type=str, default='Newsroom/scores.pkl',
                        help='The output path to save the calculated scores.')
    parser.add_argument('--m', type=str, default='DS_Focus_NN',
                        help='specify the name of discourse metric')

    parser.add_argument("--we", type=str, help="the path of static word embeddings") # our experiments use deps.words.bz2
         
    args = parser.parse_args()

    scorer = Scorer(args)
    
    scorer.score()
    scorer.save_data(args.output)


if __name__ == '__main__':
    main()
    
    exec(open("Newsroom/analysis_Newsroom.py").read())
