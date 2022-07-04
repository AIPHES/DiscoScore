import setuptools

setuptools.setup(
    name="disco_score",
    version="0.1.0",
    description="DiscoScore: Evaluating Text Generation with BERT and Discourse Coherence",
    author="Wei Zhao",
    url="https://github.com/AIPHES/DiscoScore",
    python_requires=">=3.6",
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.6.0",
        "transformers>=4.6.1",
        "nltk>=3.7.0",
        "spacy>=3.3.1",
        "spacy_udpipe>=1.0.0",
        "networkx>=2.8.4",
    ],
)
