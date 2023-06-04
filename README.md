# DiscoScore: Evaluating Text Generation with BERT and Discourse Coherence

DiscoScore is a parameterized, reference-based evaluation, with parameters on the choice of approaches (DS_Focus and DS_SENT), and the choice of readers' focus (Noun and Semantic Entity), which model text coherence from different perspectives, driven by Centering theory. 

# Current Version

1. We provide the implementations on a range of discourse metrics: LC and RC, EntityGraph, LexicalGraph and Lexical Chain. We refer readers to the paper for the details of these metrics. 
2. DiscoScore contains two variants: DS_Focus and DS_SENT, each paramerized with the choice of Focus. 
3. We provide the code for running experiments on SUMMEval and NewsRoom.

# Usage

Install from GitHub with pip:

```bash
pip install "git+https://github.com/AIPHES/DiscoScore.git"
```

Please run [SUMMEval.py](https://github.com/AIPHES/DiscoScore/blob/main/SUMMEval.py) to reproduce the results in our paper. 

Note that DS_Focus uses [Conpono](https://drive.google.com/drive/folders/1FE2loCSfdBbYrYk_qHg6W_PTqvA9w46T?usp=sharing) (finetuned BERT on discourse corpora), and DS_SENT uses [BERT-NLI](https://drive.google.com/drive/folders/19-6TgHdfAVL6xzpqzMoTpxXKLkXOCBiO?usp=sharing) (finetuned BERT on NLI). Because this configuration performs best in our intial trials (see the paper for details). 

Semantic Entity refers to a set of semantically related expressions. We detect such entities with semantic relatedness of words in vector space derived from [deps.words.bz2](https://drive.google.com/file/d/1epNYyAKban3XYg6FCMvPhQw7Yf-Y78E4/view?usp=sharing).

# Running Example
The following code supports 6 discourse metrics. Please refer to Appendix A.1 in the paper for the details of these metrics. 

Note that if system and reference texts do not contain coherence phenomena (e.g., no word repetition), then the discourse metrics would return 0.

```python
from disco_score import DiscoScorer

disco_scorer = DiscoScorer(device='cuda:0', model_name='bert-base-uncased')

system = ["Paul Merson has restarted his row with andros townsend after the Tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley. Townsend was brought on in the 83rd minute for Tottenham as they drew 0-0 against Burnley ."]
references = [["Paul Merson has restarted his row with burnley on sunday. Townsend was brought on in the 83rd minute for tottenham. Andros Townsend scores england 's equaliser in their 1-1 friendly draw. Townsend hit a stunning equaliser for england against italy."]]

for s, refs in zip(system, references):
   s = s.lower()
   refs = [r.lower() for r in refs]
   print(disco_scorer.EntityGraph(s, refs))
   print(disco_scorer.LexicalChain(s, refs))
   print(disco_scorer.RC(s, refs))    
   print(disco_scorer.LC(s, refs)) 
   print(disco_scorer.DS_Focus_NN(s, refs)) # FocusDiff 
   print(disco_scorer.DS_SENT_NN(s, refs)) # SentGraph
```

# TODO
1. We will release the code to run experiments on document-level MT.

## References

```
@inproceedings{zhao-etal-2023-discoscore,
    title = "{D}isco{S}core: Evaluating Text Generation with {BERT} and Discourse Coherence",
    author = "Zhao, Wei  and
      Strube, Michael  and
      Eger, Steffen",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.278",
    pages = "3865--3883"
}
```
We refer to [NLLG](https://nl2g.github.io/publications) for more recent work if you are interested in evaluation of text generation.

# Acknowledgement 
We are thankful to [@JohnGiorgi](https://github.com/AIPHES/DiscoScore/pull/2) for code organization and adding the feature of pip install. 
