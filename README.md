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

## Cite

```
@inproceedings{konan-etal-2024-automating,
    title = "Automating the Generation of a Functional Semantic Types Ontology with Foundational Models",
    author = "Konan, Sachin  and
      Rudolph, Larry  and
      Affens, Scott",
    editor = "Yang, Yi  and
      Davani, Aida  and
      Sil, Avi  and
      Kumar, Anoop",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 6: Industry Track)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-industry.21",
    doi = "10.18653/v1/2024.naacl-industry.21",
    pages = "248--265",
    abstract = "The rise of data science, the inherent dirtiness of data, and the proliferation of vast data providers have increased the value proposition of Semantic Types. Semantic Types are a way of encoding contextual information onto a data schema that informs the user about the definitional meaning of data, its broader context, and relationships to other types. We increasingly see a world where providing structure to this information, attached directly to data, will enable both people and systems to better understand the content of a dataset and the ability to efficiently automate data tasks such as validation, mapping/joins, and eventually machine learning. While ontological systems exist, they have not had widespread adoption due to challenges in mapping to operational datasets and lack of specificity of entity-types. Additionally, the validation checks associated with data are stored in code bases separate from the datasets that are distributed. In this paper, we address both challenges holistically by proposing a system that efficiently maps and encodes functional meaning on Semantic Types.",
}
```
## Disclaimer

Two Sigma doesn't supply or endorse any of the provided in the Kaggle/Harvard Universes. Licensing information for the 
Kaggle datasets can be found here in `assets/kaggle/licenses.csv`.
