# DiscoScore: Evaluating Text Generation with BERT and Discourse Coherence

DiscoScore is a parameterized, reference-based evaluation, with parameters on the choice of approaches (DS_Focus and DS_SENT), and the choice of readers' focus (Noun and Semantic Entity), which model text coherence from different perspectives, driven by Centering theory. 

# Current Version

1. We provide the implementations on a range of discourse metrics: LC and RC, EntityGraph, LexicalGraph and Lexical Chain. We refer readers to the paper for the details of these metrics. 
2. DiscoScore contains two variants: DS_Focus and DS_SENT, each paramerized with the choice of Focus. 
3. We provide the code for running experiments on SUMMEval. 

# Usage

Please run [SUMMEval.py](https://github.com/AIPHES/DiscoScore/blob/main/SUMMEval.py) to reproduce the results in our paper. 

Note that DS_Focus uses [Conpono]() (finetuned BERT on discourse corpora), and DS_SENT uses [BERT-NLI]() (finetuned BERT on NLI). Because this configuration performs best in our intial trials (see the paper for details). 

Semantic Entity refers to a set of semantically related expressions. We detect such entities with semantic relatedness of words in vector space given by [deps.words.bz2]().

# Usage
1. We will release the code to run experiments on NewsRoom and document-level MT.

## References

```
@misc{zhao:2022-discoscore,
      title={DiscoScore: Evaluating Text Generation with BERT and Discourse Coherence}, 
      author={Wei Zhao and Michael Strube and Steffen Eger},
      year={2022},
      eprint={2201.11176},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
