# Hate Speech Detection: ML and Deep Learning

Experimental sources for comparing Logistic Regression and SVM with TF-IDF features against BERT and DistilBERT on the Davidson and HateXplain datasets. The experiments use small (20%), medium (50%), and large (100%) dataset conditions, with macro F1 as the primary metric.

## Contents

- `hate_speech_experiment.ipynb`: experimental notebook with saved outputs.
- `run_experiment.py`: full experimental pipeline.
- `run_transformers.py`: transformer training pipeline.
- `create_notebook.py`: source used to generate the notebook. Running it overwrites the notebook.
- `data/`: input datasets, processed variants, and supplied attribution/license files.
- `results/`: saved metrics, predictions, figures, and error analyses.
- `models/`: local trained artifacts, excluded from Git.

## Setup and execution

From the project directory, create a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Open the notebook in VS Code and select the virtual environment as the kernel, or run:

```bash
python run_experiment.py
```

The separate transformer script can be run with `python run_transformers.py` after the processed data files exist. Execution writes to `data/`, `results/`, and `models/`, and may overwrite saved artifacts. Downloads of pretrained models and NLTK resources require internet access. The scripts select MPS, CUDA, or CPU according to availability.

The dependency list is inferred from source imports and is not a pinned record of the original experiment environment. Full training has not been rerun as part of repository preparation; identical results across environments are not guaranteed.

## Datasets and attribution

The dataset text includes hateful and offensive language. Preserve the supplied dataset notices and cite the original dataset publications in work using these resources:

- Davidson et al. (2017), *Automated Hate Speech Detection and the Problem of Offensive Language*: https://github.com/t-davidson/hate-speech-and-offensive-language
- Mathew et al. (2021), *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*: https://github.com/hate-alert/HateXplain

The supplied `data/LICENSE` and dataset README files concern their upstream resources; they do not establish a license for all project code. Confirm the redistribution terms for each dataset before making the repository public.
