# Automating the Generation of a Functional Semantic Types Ontology Codebase for NAACL 2024

Contains Codebase for generating FSTO Graphs for Kaggle and Harvard Data Universes. [Paper link](https://aclanthology.org/2024.naacl-industry.21/) 

## Installation
Install requirements from `requirements.txt`

## Table Download
```
gsutil -m cp -r gs://fstogendata/kaggle/tables/ assets/kaggle/tables/
gsutil -m cp -r gs://fstogendata/harvard/tables/ assets/harvard/tables/
```

## Generate Kaggle and Harvard Ontologies
Run `jupyter notebook` and then run the cells in `GenerateGraphs.ipynb`

## Analyze the Graphs
Run `jupyter notebook` and then run the cells in `AnalyzeGraphs.ipynb`

## Codebase Explanation
- `pipeline.py`: Classes for generating the Kaggle and Harvard Graphs
- `code_parsing.py`: Code for extracting classes from LLM response and adding it to the graph
- `graph_analysis.py`: Code for generating throughput and human evaluation results
- `graph_construction.py`: Code for generating parts of the graph and finding G-FST matches with embedding model
- `prompt_utils.py`: Code for getting compiling code from LLM and fixing errors
- `ray_cmds.py`: Code for interacting with Ray and using cached data
- `semantic_type_bas_classes.py`: Code for generating base classes and getting imports needed for each stage as a string
- `util.py`: Code for reading in data, creating column summaries, etc.

## Disclaimer

Two Sigma doesn't supply or endorse any of the provided in the Kaggle/Harvard Universes. Licensing information for the 
Kaggle datasets can be found here in `assets/kaggle/licenses.csv`.
