"""
Script to generate the complete hate_speech_experiment.ipynb notebook.
Run: python3 create_notebook.py
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []

def md(text):
    cells.append(new_markdown_cell(text))

def code(src):
    cells.append(new_code_cell(src))

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
md("""# Hate Speech Detection: ML vs. DL — Full Experimental Pipeline

**MSc Dissertation Experiment**
*Investigation of ML vs. DL Algorithms for Hate Speech Detection: Impact of Feature Distribution and Dataset Size*

This notebook implements the complete experimental pipeline covering:
- Two datasets: Davidson Hate Speech and HateXplain
- Four models: Logistic Regression, SVM (TF-IDF), BERT, DistilBERT
- Three dataset size conditions: Small (20%), Medium (50%), Large (100%)
- Full evaluation: metrics, statistical tests, visualisations, error analysis

**Primary metric:** Macro-averaged F1-score
**Device:** Apple M1 MPS (Metal Performance Shaders)
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — PROJECT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 0 — Create Project Structure

Creates the required directory hierarchy under the project root. All outputs
(metrics, plots, checkpoints) are written to fixed paths so that later steps
can reliably locate them.
""")

code("""\
import os, pathlib

PROJECT_ROOT = pathlib.Path(".").resolve()
dirs = [
    "data",
    "results/metrics",
    "results/confusion_matrices",
    "results/learning_curves",
    "results/robustness",
    "results/error_analysis",
    "models/checkpoints",
]
for d in dirs:
    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

print("✅ Step 0 complete — project structure ready")
print(f"   Root: {PROJECT_ROOT}")
for d in dirs:
    print(f"   {d}/")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — INSTALL AND IMPORT
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 1 — Install and Import All Libraries

Installs any missing packages, imports every module used by the pipeline,
downloads NLTK corpora, and sets the global random seed (42) for
Python `random`, NumPy, PyTorch, and Transformers. Device detection follows
the MPS → CUDA → CPU priority required for Apple M1 hardware.
""")

code("""\
import subprocess, sys

def pip_install(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

try:
    import transformers
except ImportError:
    pip_install("transformers", "datasets", "accelerate")

try:
    import torch
except ImportError:
    pip_install("torch", "torchvision", "torchaudio")

try:
    import nltk
except ImportError:
    pip_install("nltk")

try:
    import statsmodels
except ImportError:
    pip_install("statsmodels")

print("All packages available")
""")

code("""\
# ── Standard library ─────────────────────────────────────────────────────────
import os, sys, random, time, json, zipfile, warnings, pathlib, re
from collections import Counter
from copy import deepcopy

# ── Numeric / data ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")
warnings.filterwarnings("ignore")

# ── ML ────────────────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils import resample
import joblib

# ── SciPy / statsmodels ───────────────────────────────────────────────────────
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

# ── Deep learning ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed as transformers_set_seed,
)

# ── NLTK ──────────────────────────────────────────────────────────────────────
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Progress ──────────────────────────────────────────────────────────────────
from tqdm.auto import tqdm

# ── Device detection ─────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"🖥️  Using device: {DEVICE}")
print(f"   PyTorch version : {torch.__version__}")
print(f"   Transformers    : {__import__('transformers').__version__}")
print(f"   NumPy           : {np.__version__}")
print(f"   Pandas          : {pd.__version__}")
print(f"   scikit-learn    : {__import__('sklearn').__version__}")

# ── Global seed ───────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed=SEED):
    \"\"\"Set all random seeds for full reproducibility.\"\"\"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    transformers_set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(SEED)
print(f"\\n🌱 Random seed {SEED} set globally")

# ── NLTK downloads — macOS SSL workaround ─────────────────────────────────────
import ssl as _ssl
try:
    _ssl._create_default_https_context = _ssl._create_unverified_context
except Exception:
    pass

for resource in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception as _e:
        print(f"  [WARN] NLTK download '{resource}': {_e} (may already be cached)")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
print("✅ Step 1 complete — all imports and seed configured")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD DATASETS
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 2 — Load Datasets

Extracts both zip files found in `data/`, identifies the Davidson CSV and the
HateXplain JSON, loads them, applies binary label mapping, and saves cleaned
versions. If local files are absent falls back to HuggingFace Hub.

**Label binarisation:**
- Davidson: class 0 (hate) → 1; classes 1, 2 → 0
- HateXplain: majority-vote 'hatespeech' → 1; 'offensive' / 'normal' → 0
""")

code("""\
set_seed(SEED)
DATA_DIR = pathlib.Path("data")

# ── Extract all zips ──────────────────────────────────────────────────────────
for zp in DATA_DIR.glob("*.zip"):
    try:
        with zipfile.ZipFile(zp) as z:
            z.extractall(DATA_DIR)
        print(f"  Extracted: {zp.name}")
    except Exception as e:
        print(f"  Could not extract {zp.name}: {e}")

# ─────────────────────────── Davidson ────────────────────────────────────────
davidson_csv_candidates = list(DATA_DIR.rglob("labeled_data.csv"))
if davidson_csv_candidates:
    davidson_raw = pd.read_csv(davidson_csv_candidates[0])
    print(f"\\nDavidson loaded from: {davidson_csv_candidates[0]}")
else:
    print("\\nDavidson CSV not found locally — loading from HuggingFace Hub …")
    from datasets import load_dataset
    hf_ds = load_dataset("tdavidson/hate_speech_offensive", split="train")
    davidson_raw = hf_ds.to_pandas()
    davidson_raw.rename(columns={"tweet": "tweet", "class": "class"}, inplace=True)

print(f"  Shape     : {davidson_raw.shape}")
print(f"  Columns   : {davidson_raw.columns.tolist()}")
print(f"  Class dist: {davidson_raw['class'].value_counts().to_dict()}")
print("\\nFirst 5 rows:")
print(davidson_raw.head())

# Binarise: 0=hate→1, 1=offensive→0, 2=neither→0
davidson_clean = davidson_raw[["tweet", "class"]].copy()
davidson_clean.rename(columns={"tweet": "text"}, inplace=True)
davidson_clean["label_original"] = davidson_clean["class"]
davidson_clean["label"] = davidson_clean["class"].apply(lambda x: 1 if x == 0 else 0)

print(f"\\nDavidson class distribution BEFORE binarisation:")
print(davidson_clean["label_original"].value_counts().to_string())
print(f"\\nDavidson class distribution AFTER binarisation (1=hate, 0=non-hate):")
vc = davidson_clean["label"].value_counts()
print(vc.to_string())
n = len(davidson_clean)
print(f"  Hate   : {vc.get(1,0):5d} ({100*vc.get(1,0)/n:.1f}%)")
print(f"  Non-hate: {vc.get(0,0):5d} ({100*vc.get(0,0)/n:.1f}%)")

davidson_clean = davidson_clean[["text", "label"]].dropna().reset_index(drop=True)
davidson_clean.to_csv(DATA_DIR / "davidson_clean.csv", index=False)
print(f"\\nSaved → data/davidson_clean.csv  ({len(davidson_clean)} rows)")
""")

code("""\
# ─────────────────────────── HateXplain ──────────────────────────────────────
hatexplain_json_candidates = list(DATA_DIR.rglob("*.json"))
hatexplain_json_path = None
for p in hatexplain_json_candidates:
    if "hatexplain" in p.name.lower() or "dataset" in p.name.lower():
        hatexplain_json_path = p
        break
if hatexplain_json_path is None and hatexplain_json_candidates:
    hatexplain_json_path = hatexplain_json_candidates[0]

if hatexplain_json_path:
    print(f"HateXplain loaded from: {hatexplain_json_path}")
    with open(hatexplain_json_path, "r", encoding="utf-8") as f:
        hx_raw = json.load(f)

    rows = []
    for post_id, post_data in hx_raw.items():
        tokens = post_data.get("post_tokens", [])
        text = " ".join(tokens)
        annotator_labels = [a["label"] for a in post_data.get("annotators", [])]
        if not annotator_labels:
            continue
        majority_label = Counter(annotator_labels).most_common(1)[0][0]
        rows.append({"post_id": post_id, "text": text,
                     "label_original": majority_label})

    hatexplain_df = pd.DataFrame(rows)
else:
    print("HateXplain JSON not found locally — loading from HuggingFace Hub …")
    from datasets import load_dataset
    hf_hx = load_dataset("Hate-speech-CNERG/hatexplain", split="train+test+validation")
    rows = []
    for item in hf_hx:
        tokens = item.get("post_tokens", [])
        text = " ".join(tokens) if isinstance(tokens, list) else str(tokens)
        ann_labels = [a for a in item.get("annotators", {}).get("label", [])]
        if not ann_labels:
            continue
        majority = Counter(ann_labels).most_common(1)[0][0]
        rows.append({"post_id": item.get("id", ""), "text": text,
                     "label_original": majority})
    hatexplain_df = pd.DataFrame(rows)

print(f"\\n  Shape     : {hatexplain_df.shape}")
print(f"  Columns   : {hatexplain_df.columns.tolist()}")
print(f"\\nHateXplain label distribution BEFORE binarisation (majority vote):")
print(hatexplain_df["label_original"].value_counts().to_string())

# Binarise: hatespeech→1, offensive/normal→0
hatexplain_df["label"] = hatexplain_df["label_original"].apply(
    lambda x: 1 if x == "hatespeech" else 0
)

print(f"\\nHateXplain class distribution AFTER binarisation (1=hate, 0=non-hate):")
vc2 = hatexplain_df["label"].value_counts()
n2 = len(hatexplain_df)
print(vc2.to_string())
print(f"  Hate    : {vc2.get(1,0):5d} ({100*vc2.get(1,0)/n2:.1f}%)")
print(f"  Non-hate: {vc2.get(0,0):5d} ({100*vc2.get(0,0)/n2:.1f}%)")

hatexplain_clean = hatexplain_df[["text", "label"]].dropna().reset_index(drop=True)
hatexplain_clean.to_csv(DATA_DIR / "hatexplain_clean.csv", index=False)
print(f"\\nSaved → data/hatexplain_clean.csv  ({len(hatexplain_clean)} rows)")
print("\\n✅ Step 2 complete — both datasets loaded and cleaned")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 3 — Text Preprocessing

Two separate preprocessing functions are defined:

1. **`preprocess_ml(text)`** — for Logistic Regression and SVM.
   Removes URLs, mentions, lowercases, expands contractions, normalises
   repeated characters, removes punctuation, tokenises, strips stop words,
   applies WordNet lemmatisation, and rejoins.

2. **`preprocess_transformer(text)`** — for BERT and DistilBERT.
   Removes URLs and mentions only. The uncased model tokeniser handles
   lowercasing. Stop-word removal and lemmatisation are deliberately omitted
   because they disrupt the contextual dependencies that self-attention relies on.
""")

code("""\
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

CONTRACTIONS = {
    "don't": "do not", "dont": "do not",
    "can't": "cannot", "cant": "cannot",
    "won't": "will not", "wont": "will not",
    "it's": "it is", "its": "it is",
    "i'm": "i am", "im": "i am",
    "you're": "you are", "youre": "you are",
    "they're": "they are", "theyre": "they are",
    "we're": "we are", "were": "we are",
    "i've": "i have", "ive": "i have",
    "i'll": "i will", "ill": "i will",
    "isn't": "is not", "isnt": "is not",
    "aren't": "are not", "arent": "are not",
    "wasn't": "was not", "wasnt": "was not",
    "weren't": "were not", "werent": "were not",
    "hasn't": "has not", "hasnt": "has not",
    "haven't": "have not", "havent": "have not",
    "couldn't": "could not", "couldnt": "could not",
    "wouldn't": "would not", "wouldnt": "would not",
    "shouldn't": "should not", "shouldnt": "should not",
    "that's": "that is", "thats": "that is",
    "there's": "there is", "theres": "there is",
    "he's": "he is", "hes": "he is",
    "she's": "she is", "shes": "she is",
    "what's": "what is", "whats": "what is",
    "let's": "let us", "lets": "let us",
    "who's": "who is", "whos": "who is",
}

def _expand_contractions(text):
    words = text.split()
    expanded = []
    for w in words:
        expanded.append(CONTRACTIONS.get(w, w))
    return " ".join(expanded)

def preprocess_ml(text):
    \"\"\"Full preprocessing pipeline for ML (LR, SVM) models.\"\"\"
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\\S+|https\\S+|www\\.\\S+", "", text)
    # Remove user mentions
    text = re.sub(r"@\\w+", "", text)
    # Lowercase
    text = text.lower()
    # Expand contractions
    text = _expand_contractions(text)
    # Normalise repeated characters (3+ → 2)
    text = re.sub(r"(.)\\1{2,}", r"\\1\\1", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenise
    tokens = text.split()
    # Remove stop words
    tokens = [t for t in tokens if t not in STOP_WORDS]
    # Lemmatise
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    # Rejoin and strip
    return " ".join(tokens).strip()


def preprocess_transformer(text):
    \"\"\"Minimal preprocessing for transformer models.\"\"\"
    if not isinstance(text, str):
        return ""
    # Remove URLs only
    text = re.sub(r"http\\S+|https\\S+|www\\.\\S+", "", text)
    # Remove user mentions only
    text = re.sub(r"@\\w+", "", text)
    # Strip extra whitespace
    return re.sub(r"\\s+", " ", text).strip()


# ── Verify preprocessing on 5 examples ───────────────────────────────────────
sample_texts = [
    "RT @user: I hate all those fucking people!! http://t.co/link",
    "Don't you dare call me that you stupid idiot!!!",
    "She's sooooo beautiful <3 @friend",
    "These immigrants are ruining our country https://news.com",
    "I'm not racist but they're all the same smh",
]

print("=" * 80)
print("PREPROCESSING VERIFICATION — 5 EXAMPLES")
print("=" * 80)
for i, txt in enumerate(sample_texts, 1):
    print(f"\\n[{i}] ORIGINAL  : {txt}")
    print(f"    ML OUT    : {preprocess_ml(txt)}")
    print(f"    TRANS OUT : {preprocess_transformer(txt)}")

# ── Apply to both datasets ────────────────────────────────────────────────────
print("\\n" + "=" * 80)
print("Applying preprocessing to both datasets …")
print("=" * 80)

for name, df in [("davidson", davidson_clean), ("hatexplain", hatexplain_clean)]:
    print(f"\\n  Processing {name} ({len(df)} rows) …", end=" ")
    df["text_ml"] = df["text"].apply(preprocess_ml)
    df["text_transformer"] = df["text"].apply(preprocess_transformer)

    df_ml = df[["text_ml", "label"]].rename(columns={"text_ml": "text"})
    df_tr = df[["text_transformer", "label"]].rename(columns={"text_transformer": "text"})
    df["text_raw"] = df["text"]   # keep original

    df_ml.to_csv(DATA_DIR / f"{name}_ml.csv", index=False)
    df_tr.to_csv(DATA_DIR / f"{name}_transformer.csv", index=False)
    print(f"done. Saved {name}_ml.csv, {name}_transformer.csv")

print("\\n✅ Step 3 complete — preprocessing applied and saved")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SPLITS
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 4 — Dataset Splitting and Subset Creation

This is the **most critical step** for experimental validity.

**Protocol (identical for both datasets):**

1. Draw a fixed stratified 20% test set from the **full** dataset using seed 42.
   This test set never changes — all 4 models are evaluated on it.

2. From the remaining 80% pool draw three training pools:
   - **Small** : stratified sample of size = 20% of *full* dataset
   - **Medium** : stratified sample of size = 50% of *full* dataset
   - **Large** : the entire 80% pool

3. For **transformer models only**, split each training pool 90/10 into
   train and validation (stratified, seed 42).
   ML models receive the full training pool (no validation set needed).

The same test set across all conditions ensures that performance differences
reflect training data volume, not test set difficulty.
""")

code("""\
set_seed(SEED)

DATASETS = {
    "davidson": davidson_clean,
    "hatexplain": hatexplain_clean,
}

# Sizes expressed as fraction of FULL dataset
SUBSET_FRACS = {"small": 0.20, "medium": 0.50, "large": 0.80}
SUBSET_NAMES = ["small", "medium", "large"]

# Storage for all splits — used by every subsequent step
SPLITS = {}      # SPLITS[dataset][subset] = dict with all indices/dfs
TEST_SETS = {}   # TEST_SETS[dataset] = fixed test df (raw text + label)

def stratified_sample(df, n, seed=SEED, label_col="label"):
    \"\"\"Draw a stratified sample of exactly n rows from df.\"\"\"
    if n >= len(df):
        return df.copy().reset_index(drop=True)
    _, sample = train_test_split(
        df, test_size=n / len(df), stratify=df[label_col], random_state=seed
    )
    return sample.reset_index(drop=True)

for ds_name, df_full in DATASETS.items():
    N = len(df_full)
    test_n = int(round(N * 0.20))

    # ── Fixed test set ────────────────────────────────────────────────────────
    pool, test_df = train_test_split(
        df_full, test_size=test_n / N, stratify=df_full["label"], random_state=SEED
    )
    pool = pool.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    TEST_SETS[ds_name] = test_df
    SPLITS[ds_name] = {}

    for subset in SUBSET_NAMES:
        frac = SUBSET_FRACS[subset]
        if subset == "large":
            train_pool = pool.copy().reset_index(drop=True)
        else:
            target_n = int(round(N * frac))
            train_pool = stratified_sample(pool, target_n, seed=SEED)

        # ML: full pool is train (no validation)
        ml_train = train_pool.copy()

        # Transformer: 90/10 split of pool for train / validation
        if len(train_pool) > 10:
            val_n = max(2, int(round(len(train_pool) * 0.10)))
            tr_train, tr_val = train_test_split(
                train_pool, test_size=val_n / len(train_pool),
                stratify=train_pool["label"], random_state=SEED
            )
        else:
            tr_train = train_pool.copy()
            tr_val = train_pool.copy()

        SPLITS[ds_name][subset] = {
            "ml_train": ml_train.reset_index(drop=True),
            "tr_train": tr_train.reset_index(drop=True),
            "tr_val": tr_val.reset_index(drop=True),
            "test": test_df,
        }

# ── Verification table ────────────────────────────────────────────────────────
print("=" * 90)
print("DATASET SPLIT VERIFICATION")
print("=" * 90)
header = f"{'Dataset':10} | {'Subset':8} | {'ML Train':8} | {'TR Train':8} | {'Val':5} | {'Test':5} | {'Hate%':6} | {'NonHate%':8}"
print(header)
print("-" * 90)
for ds_name in ["davidson", "hatexplain"]:
    for subset in SUBSET_NAMES:
        s = SPLITS[ds_name][subset]
        ml_n   = len(s["ml_train"])
        tr_n   = len(s["tr_train"])
        val_n  = len(s["tr_val"])
        test_n = len(s["test"])
        hp = 100 * s["ml_train"]["label"].mean()
        nhp = 100 - hp
        print(f"{ds_name:10} | {subset:8} | {ml_n:8} | {tr_n:8} | {val_n:5} | {test_n:5} | {hp:5.1f}% | {nhp:6.1f}%")
print("=" * 90)

# Confirm test set balance
print("\\nTest set class balance:")
for ds_name in ["davidson", "hatexplain"]:
    ts = TEST_SETS[ds_name]
    hp = 100 * ts["label"].mean()
    print(f"  {ds_name}: {len(ts)} rows — hate {hp:.1f}%, non-hate {100-hp:.1f}%")

print("\\n✅ Step 4 complete — all splits created and verified")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TF-IDF
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 5 — TF-IDF Feature Extraction

For ML models, text is represented using TF-IDF with unigrams + bigrams,
`max_features=50_000`, and `sublinear_tf=True`.

One vectoriser is fitted per condition (dataset × subset size) on the **ML
training set only** — never on validation or test data — preventing information
leakage. Each fitted vectoriser is saved to `models/` for reproducibility.
""")

code("""\
set_seed(SEED)
MODELS_DIR = pathlib.Path("models")

TFIDF_VECTORIZERS = {}    # key: (ds_name, subset)
TFIDF_FEATURES = {}       # key: (ds_name, subset) → {"X_train","X_test","y_train","y_test"}

for ds_name in ["davidson", "hatexplain"]:
    for subset in SUBSET_NAMES:
        split = SPLITS[ds_name][subset]

        # Preprocess text using the ML pipeline directly from raw text in the split
        ml_train_texts = [preprocess_ml(t) for t in split["ml_train"]["text"].tolist()]
        test_texts     = [preprocess_ml(t) for t in split["test"]["text"].tolist()]

        y_train = split["ml_train"]["label"].values
        y_test  = split["test"]["label"].values

        # Fit TF-IDF on training only
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50_000, sublinear_tf=True)
        X_train = vec.fit_transform(ml_train_texts)
        X_test  = vec.transform(test_texts)

        key = (ds_name, subset)
        TFIDF_VECTORIZERS[key] = vec
        TFIDF_FEATURES[key] = {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test,   "y_test":  y_test,
        }

        # Save vectoriser
        vpath = MODELS_DIR / f"tfidf_{ds_name}_{subset}.pkl"
        joblib.dump(vec, vpath)
        print(f"  [{ds_name} | {subset:6}] vocab size: {len(vec.vocabulary_):,}  → {vpath.name}")

print("\\n✅ Step 5 complete — TF-IDF vectorisers fitted and saved")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — ML TRAINING
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 6 — ML Model Training and Evaluation

Trains **Logistic Regression** and **Linear SVM** (wrapped in
`CalibratedClassifierCV` for probability estimates needed by AUC-ROC) across
all 6 conditions.

For each condition the following are recorded and saved:
- Wall-clock training and inference times
- Accuracy, Macro Precision/Recall/F1, Weighted F1, AUC-ROC
- Class-wise metrics (hate class and non-hate class)
- 95% bootstrap confidence intervals for Macro F1 and AUC-ROC (n=1 000)
- Full classification report and confusion matrix
- Prediction arrays saved as `.npy`
""")

code("""\
set_seed(SEED)

def bootstrap_ci(y_true, y_pred, y_prob, metric_fn, n_iter=1000, seed=SEED):
    \"\"\"95% CI via bootstrap for a scalar metric.\"\"\"
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        try:
            s = metric_fn(y_true[idx], y_pred[idx], y_prob[idx])
            scores.append(s)
        except Exception:
            pass
    scores = np.array(scores)
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)

def f1_macro_fn(yt, yp, _):
    return f1_score(yt, yp, average="macro", zero_division=0)

def auc_fn(yt, _, yprob):
    try:
        return roc_auc_score(yt, yprob)
    except Exception:
        return 0.0

def compute_metrics(y_true, y_pred, y_prob, train_time, infer_time, n_params=None):
    \"\"\"Compute all evaluation metrics for one condition.\"\"\"
    acc   = accuracy_score(y_true, y_pred)
    mac_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mac_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    mac_f1= f1_score(y_true, y_pred, average="macro", zero_division=0)
    wt_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0

    # Class-wise
    p_cls = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_cls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f_cls = f1_score(y_true, y_pred, average=None, zero_division=0)

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    p0, p1 = (p_cls[0], p_cls[1]) if len(p_cls) >= 2 else (p_cls[0], 0)
    r0, r1 = (r_cls[0], r_cls[1]) if len(r_cls) >= 2 else (r_cls[0], 0)
    f0, f1_ = (f_cls[0], f_cls[1]) if len(f_cls) >= 2 else (f_cls[0], 0)

    # Bootstrap CIs
    f1_lo, f1_hi = bootstrap_ci(y_true, y_pred, y_prob, f1_macro_fn)
    auc_lo, auc_hi = bootstrap_ci(y_true, y_pred, y_prob, auc_fn)

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "macro_precision": mac_p,
        "macro_recall": mac_r,
        "macro_f1": mac_f1,
        "macro_f1_ci_lo": f1_lo,
        "macro_f1_ci_hi": f1_hi,
        "weighted_f1": wt_f1,
        "auc_roc": auc,
        "auc_ci_lo": auc_lo,
        "auc_ci_hi": auc_hi,
        "hate_precision": p1, "hate_recall": r1, "hate_f1": f1_,
        "nonhate_precision": p0, "nonhate_recall": r0, "nonhate_f1": f0,
        "train_time": train_time,
        "infer_time": infer_time,
        "n_params": n_params if n_params is not None else 0,
        "confusion_matrix": cm,
        "report": classification_report(y_true, y_pred, target_names=["Non-Hate","Hate"], zero_division=0),
    }


ML_RESULTS = []  # list of dicts for the results CSV

for ds_name in ["davidson", "hatexplain"]:
    for subset in SUBSET_NAMES:
        key = (ds_name, subset)
        feat = TFIDF_FEATURES[key]
        X_train, y_train = feat["X_train"], feat["y_train"]
        X_test,  y_test  = feat["X_test"],  feat["y_test"]

        for model_name, clf in [
            ("LR",  LogisticRegression(
                        solver="lbfgs", max_iter=1000,
                        class_weight="balanced", C=1.0,
                        random_state=SEED)),
            ("SVM", CalibratedClassifierCV(
                        LinearSVC(C=1.0, class_weight="balanced",
                                  max_iter=2000, random_state=SEED))),
        ]:
            set_seed(SEED)
            print(f"  Training {model_name:3} | {ds_name:10} | {subset:6} …", end=" ")
            t0 = time.time()
            clf.fit(X_train, y_train)
            train_time = time.time() - t0

            t1 = time.time()
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            infer_time = time.time() - t1

            metrics = compute_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob,
                                      train_time=train_time, infer_time=infer_time)
            print(f"  Macro F1={metrics['macro_f1']:.4f}  AUC={metrics['auc_roc']:.4f}  "
                  f"train={train_time:.1f}s")

            # Save predictions
            pred_key = f"{model_name.lower()}_{ds_name}_{subset}"
            np.save(f"results/metrics/{pred_key}_predictions.npy", y_pred)
            np.save(f"results/metrics/{pred_key}_proba.npy", y_prob)

            # Save confusion matrix
            np.save(f"results/metrics/{pred_key}_cm.npy", metrics["confusion_matrix"])

            row = {
                "Model": model_name, "Dataset": ds_name, "Subset": subset,
                "Train_Size": len(y_train), "Test_Size": len(y_test),
                **{k: v for k, v in metrics.items()
                   if k not in ("confusion_matrix", "report")},
            }
            ML_RESULTS.append(row)

ml_df_results = pd.DataFrame(ML_RESULTS)
ml_df_results.drop(columns=["confusion_matrix"], errors="ignore").to_csv(
    "results/metrics/ml_results.csv", index=False
)

print("\\n── ML Results Summary ──────────────────────────────────────────────────")
print(ml_df_results[["Model","Dataset","Subset","macro_f1","auc_roc","train_time"]].to_string(index=False))
print("\\n✅ Step 6 complete — ML results saved to results/metrics/ml_results.csv")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — BERT
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 7 — BERT Fine-Tuning

Fine-tunes `bert-base-uncased` on all 6 conditions using a **manual training
loop** with MPS device placement (avoids known HuggingFace Trainer / MPS
compatibility issues).

**Configuration:**
- Optimiser: AdamW, lr = 2e-5, weight_decay = 0.01
- Linear warmup over 10% of total steps
- Max token length: 128
- Batch size: 16 (auto-reduces to 8 on OOM)
- Epochs: 3 for small/medium, 4 for large
- Early stopping: patience = 2 epochs on validation macro F1
- Best checkpoint saved per condition
""")

code("""\
set_seed(SEED)

class HateSpeechDataset(Dataset):
    \"\"\"PyTorch Dataset for tokenised hate speech posts.\"\"\"
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def train_transformer(model_name_hf, ds_name, subset,
                      checkpoint_prefix, result_list,
                      batch_size=16, device=DEVICE):
    \"\"\"Fine-tune a transformer model for one condition.\"\"\"
    set_seed(SEED)
    print(f"\\n{'='*70}")
    print(f"  {checkpoint_prefix.upper()} | {ds_name} | {subset}")
    print(f"{'='*70}")

    split = SPLITS[ds_name][subset]
    epochs = 4 if subset == "large" else 3

    # Get preprocessed transformer texts
    tr_texts  = split["tr_train"]["text"].apply(preprocess_transformer).tolist()
    val_texts = split["tr_val"]["text"].apply(preprocess_transformer).tolist()
    te_texts  = split["test"]["text"].apply(preprocess_transformer).tolist()

    y_tr  = split["tr_train"]["label"].values
    y_val = split["tr_val"]["label"].values
    y_te  = split["test"]["label"].values

    # Tokeniser
    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)

    # Datasets
    train_ds = HateSpeechDataset(tr_texts, y_tr, tokenizer)
    val_ds   = HateSpeechDataset(val_texts, y_val, tokenizer)
    test_ds  = HateSpeechDataset(te_texts, y_te, tokenizer)

    # DataLoaders — try batch_size, fall back to 8 on memory errors
    def make_loaders(bs):
        tr_loader  = DataLoader(train_ds, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_ds,   batch_size=bs, shuffle=False)
        te_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False)
        return tr_loader, val_loader, te_loader

    try:
        tr_loader, val_loader, te_loader = make_loaders(batch_size)
    except Exception:
        batch_size = 8
        tr_loader, val_loader, te_loader = make_loaders(batch_size)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_hf, num_labels=2, ignore_mismatched_sizes=True
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = model.to(device)

    # Optimiser and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(tr_loader) * epochs
    warmup_steps = int(0.10 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Training loop
    best_val_f1 = -1
    patience = 2
    no_improve = 0
    best_state = None
    checkpoint_dir = pathlib.Path(f"models/checkpoints/{checkpoint_prefix}_{ds_name}_{subset}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_train_time = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        t_epoch = time.time()
        for batch in tr_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels    = batch["labels"].to(device)
            try:
                outputs = model(input_ids=input_ids,
                                attention_mask=attn_mask,
                                labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\\n  OOM in epoch {epoch} — reducing batch size and retrying condition")
                    del model
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
                    return train_transformer(
                        model_name_hf, ds_name, subset,
                        checkpoint_prefix, result_list,
                        batch_size=8, device=device
                    )
                raise

        epoch_time = time.time() - t_epoch
        total_train_time += epoch_time

        # Validation
        model.eval()
        val_preds, val_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attn_mask)
                logits = outputs.logits
                probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                preds  = logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_probs.extend(probs.tolist())

        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        avg_loss = epoch_loss / max(len(tr_loader), 1)
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_macro_F1={val_f1:.4f}  "
              f"time={epoch_time:.0f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping after {epoch} epochs (no improvement for {patience} epochs)")
                break

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # Inference on test set
    model.eval()
    te_preds, te_probs = [], []
    t_infer = time.time()
    with torch.no_grad():
        for batch in te_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attn_mask)
            logits = outputs.logits
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=-1).cpu().numpy()
            te_preds.extend(preds.tolist())
            te_probs.extend(probs.tolist())
    infer_time = time.time() - t_infer

    te_preds = np.array(te_preds)
    te_probs = np.array(te_probs)

    metrics = compute_metrics(y_true=y_te, y_pred=te_preds, y_prob=te_probs,
                              train_time=total_train_time,
                              infer_time=infer_time, n_params=n_params)
    print(f"  TEST → Macro F1={metrics['macro_f1']:.4f}  AUC={metrics['auc_roc']:.4f}  "
          f"train_total={total_train_time:.0f}s  infer={infer_time:.2f}s")

    # Save predictions and checkpoint
    pred_key = f"{checkpoint_prefix}_{ds_name}_{subset}"
    np.save(f"results/metrics/{pred_key}_predictions.npy", te_preds)
    np.save(f"results/metrics/{pred_key}_proba.npy", te_probs)
    np.save(f"results/metrics/{pred_key}_cm.npy", metrics["confusion_matrix"])
    torch.save(best_state, checkpoint_dir / "best_model.pt")

    row = {
        "Model": checkpoint_prefix.upper(), "Dataset": ds_name, "Subset": subset,
        "Train_Size": len(y_tr), "Test_Size": len(y_te),
        **{k: v for k, v in metrics.items()
           if k not in ("confusion_matrix", "report")},
    }
    result_list.append(row)

    # Clean up GPU/MPS memory
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    return row


BERT_RESULTS = []
for ds_name in ["davidson", "hatexplain"]:
    for subset in SUBSET_NAMES:
        train_transformer(
            model_name_hf="bert-base-uncased",
            ds_name=ds_name,
            subset=subset,
            checkpoint_prefix="bert",
            result_list=BERT_RESULTS,
            device=DEVICE,
        )

bert_df_results = pd.DataFrame(BERT_RESULTS)
bert_df_results.to_csv("results/metrics/bert_results.csv", index=False)
print("\\n✅ Step 7 complete — BERT results saved to results/metrics/bert_results.csv")
print(bert_df_results[["Model","Dataset","Subset","macro_f1","auc_roc","train_time"]].to_string(index=False))
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — DISTILBERT
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 8 — DistilBERT Fine-Tuning

Identical training configuration to BERT (Step 7) but using
`distilbert-base-uncased`. DistilBERT has ~40% fewer parameters and runs
~60% faster, providing the efficiency contrast central to Research Question 3.
""")

code("""\
set_seed(SEED)

DISTILBERT_RESULTS = []
for ds_name in ["davidson", "hatexplain"]:
    for subset in SUBSET_NAMES:
        train_transformer(
            model_name_hf="distilbert-base-uncased",
            ds_name=ds_name,
            subset=subset,
            checkpoint_prefix="distilbert",
            result_list=DISTILBERT_RESULTS,
            device=DEVICE,
        )

distilbert_df_results = pd.DataFrame(DISTILBERT_RESULTS)
distilbert_df_results.to_csv("results/metrics/distilbert_results.csv", index=False)
print("\\n✅ Step 8 complete — DistilBERT results saved to results/metrics/distilbert_results.csv")
print(distilbert_df_results[["Model","Dataset","Subset","macro_f1","auc_roc","train_time"]].to_string(index=False))
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — McNemar's TEST
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 9 — Statistical Significance Testing (McNemar's Test)

Applies McNemar's test to all pairwise model comparisons on the **fixed
held-out test set** for each condition (6 pairs × 6 conditions = 36 tests).

McNemar's test is the standard non-parametric test for comparing two
classifiers on the same test set. It operates on the contingency table of
disagreements (where one model is right and the other wrong, and vice versa).
Significance level α = 0.05.
""")

code("""\
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test

MODEL_NAMES = ["lr", "svm", "bert", "distilbert"]
MODEL_PAIRS = [
    ("lr", "svm"), ("lr", "bert"), ("lr", "distilbert"),
    ("svm", "bert"), ("svm", "distilbert"), ("bert", "distilbert"),
]

def load_preds(model, ds, subset):
    path = pathlib.Path(f"results/metrics/{model}_{ds}_{subset}_predictions.npy")
    return np.load(path)

mcnemar_rows = []
print("=" * 80)
print("McNemar's Test Results")
print("=" * 80)
print(f"{'Model A':12} {'Model B':12} {'Dataset':10} {'Subset':8} {'Stat':8} {'p-val':8} {'Sig.':5}")
print("-" * 80)

for ds_name in ["davidson", "hatexplain"]:
    for subset in SUBSET_NAMES:
        for ma, mb in MODEL_PAIRS:
            try:
                pa = load_preds(ma, ds_name, subset)
                pb = load_preds(mb, ds_name, subset)
                y_te = split = SPLITS[ds_name][subset]["test"]["label"].values

                # 2×2 contingency table
                n00 = int(np.sum((pa == y_te) & (pb == y_te)))   # both correct
                n01 = int(np.sum((pa == y_te) & (pb != y_te)))   # A correct, B wrong
                n10 = int(np.sum((pa != y_te) & (pb == y_te)))   # A wrong, B correct
                n11 = int(np.sum((pa != y_te) & (pb != y_te)))   # both wrong

                table = [[n00, n01], [n10, n11]]
                result = mcnemar_test(table, exact=False, correction=True)
                stat   = result.statistic
                pval   = result.pvalue
                sig    = "Yes" if pval < 0.05 else "No"

                row = {
                    "Model_A": ma.upper(), "Model_B": mb.upper(),
                    "Dataset": ds_name, "Subset": subset,
                    "McNemar_Stat": round(stat, 4), "p_value": round(pval, 6),
                    "Significant": sig,
                    "n01": n01, "n10": n10,
                }
                mcnemar_rows.append(row)
                print(f"{ma.upper():12} {mb.upper():12} {ds_name:10} {subset:8} "
                      f"{stat:8.3f} {pval:8.4f} {sig:5}")
            except Exception as e:
                print(f"  [WARN] {ma} vs {mb} | {ds_name} | {subset}: {e}")

mcnemar_df = pd.DataFrame(mcnemar_rows)
mcnemar_df.to_csv("results/metrics/mcnemar_results.csv", index=False)
print("\\n✅ Step 9 complete — McNemar results saved to results/metrics/mcnemar_results.csv")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 10 — Confusion Matrix Visualisations

Generates 24 individual confusion matrix heatmaps (4 models × 2 datasets ×
3 subsets) showing raw counts and row-normalised percentages, plus one combined
4 × 6 grid figure.
""")

code("""\
CM_DIR = pathlib.Path("results/confusion_matrices")
MODEL_LABELS = {"lr": "Logistic Regression", "svm": "Linear SVM",
                "bert": "BERT", "distilbert": "DistilBERT"}
SUBSET_LABELS = {"small": "Small (20%)", "medium": "Medium (50%)", "large": "Large (100%)"}
TICK_LABELS = ["Non-Hate (0)", "Hate (1)"]

def plot_cm(cm, title, save_path, ax=None):
    \"\"\"Plot a single confusion matrix with counts and percentages.\"\"\"
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax,
                xticklabels=TICK_LABELS, yticklabels=TICK_LABELS,
                linewidths=0.5, linecolor="grey", cbar=True)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i, j]}\\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")

    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)

    if standalone:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()

# Individual plots
for model_key in MODEL_NAMES:
    for ds_name in ["davidson", "hatexplain"]:
        for subset in SUBSET_NAMES:
            try:
                cm = np.load(f"results/metrics/{model_key}_{ds_name}_{subset}_cm.npy")
                title = (f"{MODEL_LABELS[model_key]} | "
                         f"{ds_name.capitalize()} | {SUBSET_LABELS[subset]}")
                save_path = CM_DIR / f"cm_{model_key}_{ds_name}_{subset}.png"
                plot_cm(cm, title, save_path)
            except Exception as e:
                print(f"  [WARN] CM for {model_key}_{ds_name}_{subset}: {e}")

# Combined 4×6 grid
fig, axes = plt.subplots(4, 6, figsize=(28, 20))
fig.suptitle("All 24 Confusion Matrices — Hate Speech Detection Experiment",
             fontsize=16, y=1.01)
for row_i, model_key in enumerate(MODEL_NAMES):
    for col_i, (ds_name, subset) in enumerate(
        [(d, s) for d in ["davidson", "hatexplain"] for s in SUBSET_NAMES]
    ):
        ax = axes[row_i, col_i]
        try:
            cm = np.load(f"results/metrics/{model_key}_{ds_name}_{subset}_cm.npy")
            title = f"{MODEL_LABELS[model_key]}\\n{ds_name.capitalize()} {SUBSET_LABELS[subset]}"
            plot_cm(cm, title, None, ax=ax)
        except Exception as e:
            ax.set_title(f"{model_key}\\n{ds_name} {subset}\\n(missing)", fontsize=7)
            ax.axis("off")

plt.tight_layout()
plt.savefig(CM_DIR / "all_confusion_matrices.png", dpi=90, bbox_inches="tight")
plt.close()
print("✅ Step 10 complete — confusion matrices saved to results/confusion_matrices/")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — LEARNING CURVES
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 11 — Learning Curve Visualisations

Plots Macro F1-score vs. training subset size for all 4 models on each
dataset. Shaded 95% CI bands (from bootstrap CIs computed in Step 6–8)
show uncertainty. Reveals at which dataset size DL models begin to
outperform ML models (Research Question 1).
""")

code("""\
LC_DIR = pathlib.Path("results/learning_curves")

# Assemble all results into one DataFrame
all_results_dfs = []
for fpath in ["results/metrics/ml_results.csv",
              "results/metrics/bert_results.csv",
              "results/metrics/distilbert_results.csv"]:
    try:
        all_results_dfs.append(pd.read_csv(fpath))
    except Exception:
        pass

ALL_RESULTS = pd.concat(all_results_dfs, ignore_index=True) if all_results_dfs else pd.DataFrame()

MODEL_COLORS  = {"LR": "#1f77b4", "SVM": "#ff7f0e", "BERT": "#2ca02c", "DISTILBERT": "#d62728"}
MODEL_MARKERS = {"LR": "o", "SVM": "s", "BERT": "^", "DISTILBERT": "D"}
X_LABELS = {"small": "Small\\n(20%)", "medium": "Medium\\n(50%)", "large": "Large\\n(100%)"}
X_POS    = [0.20, 0.50, 1.00]

def make_learning_curve(ds_name, ax, title_suffix=""):
    ds_results = ALL_RESULTS[ALL_RESULTS["Dataset"] == ds_name]
    for m in ["LR", "SVM", "BERT", "DISTILBERT"]:
        m_res = ds_results[ds_results["Model"] == m].copy()
        if m_res.empty:
            continue
        m_res["x"] = m_res["Subset"].map({"small": 0.20, "medium": 0.50, "large": 1.00})
        m_res = m_res.sort_values("x")
        xs  = m_res["x"].values
        ys  = m_res["macro_f1"].values
        los = m_res["macro_f1_ci_lo"].values if "macro_f1_ci_lo" in m_res else ys * 0.98
        his = m_res["macro_f1_ci_hi"].values if "macro_f1_ci_hi" in m_res else ys * 1.02
        ax.plot(xs, ys, color=MODEL_COLORS[m], marker=MODEL_MARKERS[m],
                linewidth=2, markersize=8, label=m)
        ax.fill_between(xs, los, his, alpha=0.15, color=MODEL_COLORS[m])

    ax.set_xlim(0.10, 1.10)
    ax.set_ylim(0, 1)
    ax.set_xticks(X_POS)
    ax.set_xticklabels([X_LABELS[s] for s in ["small", "medium", "large"]])
    ax.set_xlabel("Training Subset Size", fontsize=11)
    ax.set_ylabel("Macro F1-Score", fontsize=11)
    ax.set_title(f"Learning Curves — {ds_name.capitalize()}{title_suffix}\\n"
                 f"Macro F1-Score vs Training Dataset Size", fontsize=12)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(bottom=0)

for ds_name in ["davidson", "hatexplain"]:
    fig, ax = plt.subplots(figsize=(10, 6))
    make_learning_curve(ds_name, ax)
    plt.tight_layout()
    plt.savefig(LC_DIR / f"learning_curve_{ds_name}.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved learning_curve_{ds_name}.png")

# Combined side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
make_learning_curve("davidson", ax1)
make_learning_curve("hatexplain", ax2)
fig.suptitle("Learning Curves: ML vs DL Models on Both Datasets", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(LC_DIR / "learning_curves_combined.png", dpi=120, bbox_inches="tight")
plt.close()

print("✅ Step 11 complete — learning curves saved to results/learning_curves/")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 12 — Robustness Analysis

Three analyses examine how robust each model is to:
1. **Performance consistency across dataset sizes** — how much does Macro F1
   drop from Large → Small?
2. **Cross-dataset stability** — does performance differ between Davidson
   (explicit) and HateXplain (implicit)?
3. **Class-wise comparison** — is each model better at detecting hate or
   non-hate content?
""")

code("""\
ROB_DIR = pathlib.Path("results/robustness")

# ── Analysis 1: Degradation Large → Small ────────────────────────────────────
print("=" * 60)
print("Analysis 1 — Performance Degradation: Large → Small")
print("=" * 60)

degrad_rows = []
for m in ["LR", "SVM", "BERT", "DISTILBERT"]:
    for ds_name in ["davidson", "hatexplain"]:
        sub = ALL_RESULTS[(ALL_RESULTS["Model"] == m) & (ALL_RESULTS["Dataset"] == ds_name)]
        if sub.empty:
            continue
        f1_large = sub[sub["Subset"] == "large"]["macro_f1"].values
        f1_small = sub[sub["Subset"] == "small"]["macro_f1"].values
        if len(f1_large) and len(f1_small):
            drop = float(f1_large[0]) - float(f1_small[0])
            degrad_rows.append({
                "Model": m, "Dataset": ds_name,
                "F1_Large": round(float(f1_large[0]), 4),
                "F1_Small": round(float(f1_small[0]), 4),
                "Degradation": round(drop, 4),
            })
            print(f"  {m:10} | {ds_name:10} | Large={f1_large[0]:.4f}  "
                  f"Small={f1_small[0]:.4f}  Drop={drop:+.4f}")

degrad_df = pd.DataFrame(degrad_rows)
degrad_df.to_csv(ROB_DIR / "performance_degradation.csv", index=False)

# ── Analysis 2: Cross-dataset gap heatmap ────────────────────────────────────
print("\\n" + "=" * 60)
print("Analysis 2 — Cross-Dataset Stability (Davidson vs HateXplain)")
print("=" * 60)

gap_data = {}
for m in ["LR", "SVM", "BERT", "DISTILBERT"]:
    gap_data[m] = {}
    for subset in SUBSET_NAMES:
        d_f1 = ALL_RESULTS[
            (ALL_RESULTS["Model"] == m) &
            (ALL_RESULTS["Dataset"] == "davidson") &
            (ALL_RESULTS["Subset"] == subset)
        ]["macro_f1"].values
        h_f1 = ALL_RESULTS[
            (ALL_RESULTS["Model"] == m) &
            (ALL_RESULTS["Dataset"] == "hatexplain") &
            (ALL_RESULTS["Subset"] == subset)
        ]["macro_f1"].values
        if len(d_f1) and len(h_f1):
            gap = float(d_f1[0]) - float(h_f1[0])
            gap_data[m][subset] = round(gap, 4)
            print(f"  {m:10} | {subset:6} | gap={gap:+.4f}")

gap_df = pd.DataFrame(gap_data).T[SUBSET_NAMES]
fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(gap_df.astype(float), annot=True, fmt=".3f", cmap="RdYlGn_r",
            center=0, ax=ax,
            xticklabels=["Small", "Medium", "Large"],
            yticklabels=gap_df.index)
ax.set_title("Cross-Dataset Gap: Davidson F1 − HateXplain F1\\n"
             "(Positive = Davidson higher = ML benefits from explicit content)")
ax.set_xlabel("Training Subset")
ax.set_ylabel("Model")
plt.tight_layout()
plt.savefig(ROB_DIR / "cross_dataset_gap.png", dpi=120, bbox_inches="tight")
plt.close()

# ── Analysis 3: Class-wise F1 comparison bar chart ───────────────────────────
print("\\n" + "=" * 60)
print("Analysis 3 — Class-wise F1 Comparison")
print("=" * 60)

for ds_name in ["davidson", "hatexplain"]:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f"Class-wise F1 — {ds_name.capitalize()}", fontsize=13)
    for si, subset in enumerate(SUBSET_NAMES):
        ax = axes[si]
        sub_res = ALL_RESULTS[
            (ALL_RESULTS["Dataset"] == ds_name) &
            (ALL_RESULTS["Subset"] == subset)
        ]
        models = sub_res["Model"].tolist()
        hate_f1   = sub_res["hate_f1"].tolist() if "hate_f1" in sub_res.columns else []
        nonhate_f1= sub_res["nonhate_f1"].tolist() if "nonhate_f1" in sub_res.columns else []
        x = np.arange(len(models))
        w = 0.35
        if models:
            ax.bar(x - w/2, hate_f1, w, label="Hate (1)", color="#e74c3c", alpha=0.8)
            ax.bar(x + w/2, nonhate_f1, w, label="Non-Hate (0)", color="#3498db", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(SUBSET_LABELS[subset])
        ax.set_xlabel("Model")
        ax.set_ylabel("F1-Score")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ROB_DIR / f"classwise_f1_{ds_name}.png", dpi=120, bbox_inches="tight")
    plt.close()

# Combined classwise for both datasets
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Class-wise F1 Comparison — All Models, Datasets, Conditions", fontsize=14)
for di, ds_name in enumerate(["davidson", "hatexplain"]):
    for si, subset in enumerate(SUBSET_NAMES):
        ax = axes[di][si]
        sub_res = ALL_RESULTS[
            (ALL_RESULTS["Dataset"] == ds_name) &
            (ALL_RESULTS["Subset"] == subset)
        ]
        models   = sub_res["Model"].tolist()
        hate_f1  = sub_res.get("hate_f1", pd.Series([])).tolist()
        nhate_f1 = sub_res.get("nonhate_f1", pd.Series([])).tolist()
        x = np.arange(len(models))
        if models:
            ax.bar(x - 0.175, hate_f1,  0.35, label="Hate",     color="#e74c3c", alpha=0.8)
            ax.bar(x + 0.175, nhate_f1, 0.35, label="Non-Hate", color="#3498db", alpha=0.8)
            ax.set_xticks(x); ax.set_xticklabels(models, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{ds_name.capitalize()} — {SUBSET_LABELS[subset]}", fontsize=9)
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(ROB_DIR / "classwise_f1_comparison.png", dpi=110, bbox_inches="tight")
plt.close()

print("\\n✅ Step 12 complete — robustness analyses saved to results/robustness/")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 — COMPUTATIONAL EFFICIENCY
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 13 — Computational Efficiency Analysis

Compares the four models on training time, inference time, parameter count,
and Macro F1 performance. These measures are critical for practitioners
choosing between ML and DL approaches under resource constraints.
""")

code("""\
# ── Chart 1: Training time by model and condition ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
fig.suptitle("Training Time Comparison", fontsize=14)

for di, ds_name in enumerate(["davidson", "hatexplain"]):
    ax = axes[di]
    ds_res = ALL_RESULTS[ALL_RESULTS["Dataset"] == ds_name]
    models = ["LR", "SVM", "BERT", "DISTILBERT"]
    n_models = len(models)
    n_subsets = len(SUBSET_NAMES)
    x = np.arange(n_models)
    width = 0.25
    for si, subset in enumerate(SUBSET_NAMES):
        vals = []
        for m in models:
            r = ds_res[(ds_res["Model"] == m) & (ds_res["Subset"] == subset)]["train_time"].values
            vals.append(float(r[0]) if len(r) else 0)
        offset = (si - 1) * width
        ax.bar(x + offset, vals, width, label=SUBSET_LABELS[subset], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title(f"{ds_name.capitalize()}")
    ax.set_xlabel("Model")
    ax.set_ylabel("Training Time (seconds)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("results/training_time_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# ── Chart 2: Inference time ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
models = ["LR", "SVM", "BERT", "DISTILBERT"]
for di, ds_name in enumerate(["davidson", "hatexplain"]):
    infer_times = []
    for m in models:
        r = ALL_RESULTS[(ALL_RESULTS["Model"] == m) & (ALL_RESULTS["Dataset"] == ds_name) &
                        (ALL_RESULTS["Subset"] == "large")]["infer_time"].values
        infer_times.append(float(r[0]) if len(r) else 0)
    x = np.arange(len(models))
    ax.bar(x + di * 0.35, infer_times, 0.35, label=ds_name.capitalize(), alpha=0.8)
ax.set_xticks(np.arange(len(models)) + 0.175)
ax.set_xticklabels(models)
ax.set_xlabel("Model")
ax.set_ylabel("Inference Time (seconds)")
ax.set_title("Inference Time on Test Set (Large condition)")
ax.legend()
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("results/inference_time_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# ── Chart 3: Model size vs performance ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
model_colors_scatter = {"LR": "#1f77b4", "SVM": "#ff7f0e",
                        "BERT": "#2ca02c", "DISTILBERT": "#d62728"}
for m in ["LR", "SVM", "BERT", "DISTILBERT"]:
    r = ALL_RESULTS[(ALL_RESULTS["Model"] == m) & (ALL_RESULTS["Subset"] == "large")]
    if r.empty:
        continue
    mean_f1 = r["macro_f1"].mean()
    n_params = r["n_params"].mean() if "n_params" in r.columns else 0
    ax.scatter(n_params, mean_f1, s=200, color=model_colors_scatter[m],
               label=m, zorder=5)
    ax.annotate(m, (n_params, mean_f1), textcoords="offset points",
                xytext=(10, 5), fontsize=11)
ax.set_xlabel("Number of Trainable Parameters", fontsize=11)
ax.set_ylabel("Mean Macro F1-Score (Large condition)", fontsize=11)
ax.set_title("Model Size vs Performance")
ax.legend(fontsize=10)
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("results/model_size_vs_performance.png", dpi=120, bbox_inches="tight")
plt.close()

# ── Chart 4: Efficiency summary table ────────────────────────────────────────
eff_rows = []
for m in ["LR", "SVM", "BERT", "DISTILBERT"]:
    for ds_name in ["davidson", "hatexplain"]:
        r_large = ALL_RESULTS[(ALL_RESULTS["Model"] == m) &
                              (ALL_RESULTS["Dataset"] == ds_name) &
                              (ALL_RESULTS["Subset"] == "large")]
        if r_large.empty:
            continue
        row = r_large.iloc[0]
        tt  = float(row.get("train_time", 0))
        it  = float(row.get("infer_time", 0))
        mf1 = float(row.get("macro_f1",  0))
        np_ = int(row.get("n_params",    0))
        f1ps = mf1 / max(tt, 1e-9)
        eff_rows.append({
            "Model": m, "Dataset": ds_name,
            "Train_Time_s": round(tt, 2),
            "Infer_Time_s": round(it, 4),
            "Parameters":   np_,
            "Macro_F1_Large": round(mf1, 4),
            "F1_per_TrainSec": round(f1ps, 6),
        })

eff_df = pd.DataFrame(eff_rows)
eff_df.to_csv("results/metrics/efficiency_summary.csv", index=False)
print(eff_df.to_string(index=False))
print("\\n✅ Step 13 complete — efficiency charts saved to results/")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 14 — ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 14 — Error Analysis

For the **Large (100%)** condition only, systematically examines
misclassifications for each of the 8 model-dataset combinations.

**Error categories:**
- **False Positive (FP)**: predicted hate, true non-hate
- **False Negative (FN)**: predicted non-hate, true hate

Up to 20 examples of each error type are extracted and saved, along with
summary statistics.
""")

code("""\
ERR_DIR = pathlib.Path("results/error_analysis")

for model_key in MODEL_NAMES:
    for ds_name in ["davidson", "hatexplain"]:
        try:
            pred_path = pathlib.Path(
                f"results/metrics/{model_key}_{ds_name}_large_predictions.npy"
            )
            y_pred = np.load(pred_path)
            y_true = SPLITS[ds_name]["large"]["test"]["label"].values
            raw_texts = SPLITS[ds_name]["large"]["test"]["text"].values

            # Preprocess text columns
            text_ml   = np.array([preprocess_ml(t) for t in raw_texts])
            text_trans= np.array([preprocess_transformer(t) for t in raw_texts])

     
            fp_mask = (y_pred == 1) & (y_true == 0)
            fn_mask = (y_pred == 0) & (y_true == 1)
            total_err = int(fp_mask.sum() + fn_mask.sum())

            print(f"\\n{'='*60}")
            print(f"  {model_key.upper():12} | {ds_name:10} | Large subset")
            print(f"{'='*60}")
            print(f"  Total misclassifications : {total_err}")
            print(f"  False Positives (FP)     : {fp_mask.sum()} "
                  f"({100*fp_mask.sum()/max(len(y_true),1):.1f}%)")
            print(f"  False Negatives (FN)     : {fn_mask.sum()} "
                  f"({100*fn_mask.sum()/max(len(y_true),1):.1f}%)")

            # Collect examples
            rows = []
            for idx in np.where(fp_mask)[0][:20]:
                rows.append({
                    "error_type": "FP", "true_label": int(y_true[idx]),
                    "pred_label": int(y_pred[idx]),
                    "raw_text": raw_texts[idx],
                    "ml_text": text_ml[idx],
                    "trans_text": text_trans[idx],
                })
            for idx in np.where(fn_mask)[0][:20]:
                rows.append({
                    "error_type": "FN", "true_label": int(y_true[idx]),
                    "pred_label": int(y_pred[idx]),
                    "raw_text": raw_texts[idx],
                    "ml_text": text_ml[idx],
                    "trans_text": text_trans[idx],
                })

            err_df = pd.DataFrame(rows)
            err_path = ERR_DIR / f"{model_key}_{ds_name}_errors.csv"
            err_df.to_csv(err_path, index=False)

            # Print 3 examples of each
            for etype, label in [("FP", "False Positive"), ("FN", "False Negative")]:
                examples = err_df[err_df["error_type"] == etype].head(3)
                print(f"\\n  {label} examples:")
                for _, row in examples.iterrows():
                    txt = str(row["raw_text"])[:120]
                    print(f"    → {txt}")

        except Exception as e:
            print(f"  [WARN] Error analysis for {model_key}_{ds_name}: {e}")

print("\\n✅ Step 14 complete — error analyses saved to results/error_analysis/")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 15 — MASTER RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 15 — Master Results Table

Combines all 24 experimental conditions (4 models × 2 datasets × 3 subsets)
into a single comprehensive CSV. Also adds McNemar p-values for all pairwise
comparisons. This is the primary output for Chapter 4 of the dissertation.
""")

code("""\
# Reload all result CSVs freshly
all_parts = []
for fpath in ["results/metrics/ml_results.csv",
              "results/metrics/bert_results.csv",
              "results/metrics/distilbert_results.csv"]:
    try:
        df_part = pd.read_csv(fpath)
        all_parts.append(df_part)
    except Exception as e:
        print(f"  [WARN] Could not load {fpath}: {e}")

if all_parts:
    MASTER = pd.concat(all_parts, ignore_index=True)
else:
    MASTER = pd.DataFrame()

# Attach McNemar p-values
if not mcnemar_df.empty and not MASTER.empty:
    mcn = mcnemar_df.copy()
    for target_model in ["LR", "SVM", "BERT", "DISTILBERT"]:
        col_name = f"McNemar_vs_{target_model}_p"
        MASTER[col_name] = None
        for i, row in MASTER.iterrows():
            mA = str(row.get("Model", "")).upper()
            ds = str(row.get("Dataset", ""))
            sub = str(row.get("Subset", ""))
            # Find the matching McNemar test
            match = mcn[
                (((mcn["Model_A"] == mA) & (mcn["Model_B"] == target_model)) |
                 ((mcn["Model_A"] == target_model) & (mcn["Model_B"] == mA))) &
                (mcn["Dataset"] == ds) & (mcn["Subset"] == sub)
            ]
            if not match.empty:
                MASTER.at[i, col_name] = match.iloc[0]["p_value"]

# Clean and save
cols_to_keep = [
    "Model", "Dataset", "Subset", "Train_Size", "Test_Size",
    "accuracy", "macro_precision", "macro_recall",
    "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi",
    "weighted_f1", "auc_roc", "auc_ci_lo", "auc_ci_hi",
    "hate_precision", "hate_recall", "hate_f1",
    "nonhate_precision", "nonhate_recall", "nonhate_f1",
    "train_time", "infer_time", "n_params",
    "McNemar_vs_LR_p", "McNemar_vs_SVM_p",
    "McNemar_vs_BERT_p", "McNemar_vs_DISTILBERT_p",
]
existing_cols = [c for c in cols_to_keep if c in MASTER.columns]
MASTER = MASTER[existing_cols].copy()

# Sort for readability
if not MASTER.empty:
    MASTER = MASTER.sort_values(["Dataset", "Subset", "Model"]).reset_index(drop=True)
    MASTER.to_csv("results/master_results.csv", index=False)

    print("=" * 100)
    print("MASTER RESULTS TABLE (24 experimental conditions)")
    print("=" * 100)
    display_cols = ["Model", "Dataset", "Subset", "Train_Size",
                    "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi",
                    "auc_roc", "hate_f1", "train_time"]
    disp_cols = [c for c in display_cols if c in MASTER.columns]
    print(MASTER[disp_cols].to_string(index=False))
    print(f"\\nTotal rows: {len(MASTER)}")
    print("\\n✅ Step 15 complete — master results saved to results/master_results.csv")
else:
    print("[WARN] Master results table is empty — check earlier steps")
""")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 16 — RESEARCH QUESTION ANSWERS
# ─────────────────────────────────────────────────────────────────────────────
md("""## Step 16 — Research Question Summary

Generates explicit, data-driven answers to all three research questions
using the actual experimental results. Saved as a text file for reference
in Chapter 4 and Chapter 5 of the dissertation.
""")

code("""\
if MASTER.empty:
    print("[WARN] Master results not available — cannot generate RQ answers")
else:
    rq_lines = []
    rq_lines.append("=" * 80)
    rq_lines.append("RESEARCH QUESTION ANSWERS — BASED ON EXPERIMENTAL RESULTS")
    rq_lines.append("=" * 80)

    # ── RQ1 ──────────────────────────────────────────────────────────────────
    rq_lines.append("\\nRQ1: How does the comparative performance of ML and DL models change")
    rq_lines.append("     as training dataset size increases from small to large subsets?")
    rq_lines.append("")

    for ds_name in ["davidson", "hatexplain"]:
        rq_lines.append(f"  Dataset: {ds_name.capitalize()}")
        for m in ["LR", "SVM", "BERT", "DISTILBERT"]:
            row_data = MASTER[(MASTER["Dataset"] == ds_name) & (MASTER["Model"] == m)]
            line = f"    {m:12}"
            for subset in ["small", "medium", "large"]:
                r = row_data[row_data["Subset"] == subset]["macro_f1"].values
                val = f"{float(r[0]):.4f}" if len(r) else "N/A"
                line += f"  {subset}={val}"
            rq_lines.append(line)

        # At which size does DL outperform ML?
        for subset in ["small", "medium", "large"]:
            ml_f1s = MASTER[(MASTER["Dataset"] == ds_name) &
                            (MASTER["Subset"] == subset) &
                            (MASTER["Model"].isin(["LR", "SVM"]))]["macro_f1"].max()
            dl_f1s = MASTER[(MASTER["Dataset"] == ds_name) &
                            (MASTER["Subset"] == subset) &
                            (MASTER["Model"].isin(["BERT", "DISTILBERT"]))]["macro_f1"].max()
            gap = dl_f1s - ml_f1s if (not pd.isna(dl_f1s) and not pd.isna(ml_f1s)) else 0
            rq_lines.append(f"    [{subset:6}] Best ML={ml_f1s:.4f}  Best DL={dl_f1s:.4f}  "
                            f"DL advantage={gap:+.4f}")
        rq_lines.append("")

    # ── RQ2 ──────────────────────────────────────────────────────────────────
    rq_lines.append("\\nRQ2: Does linguistic dataset complexity moderate the ML vs DL gap?")
    rq_lines.append("     (Davidson = explicit; HateXplain = implicit)")
    rq_lines.append("")
    for subset in SUBSET_NAMES:
        rq_lines.append(f"  Subset: {SUBSET_LABELS[subset]}")
        for m in ["LR", "SVM", "BERT", "DISTILBERT"]:
            dav = MASTER[(MASTER["Dataset"] == "davidson") &
                         (MASTER["Model"] == m) &
                         (MASTER["Subset"] == subset)]["macro_f1"].values
            hx  = MASTER[(MASTER["Dataset"] == "hatexplain") &
                         (MASTER["Model"] == m) &
                         (MASTER["Subset"] == subset)]["macro_f1"].values
            if len(dav) and len(hx):
                diff = float(dav[0]) - float(hx[0])
                rq_lines.append(f"    {m:12}  Davidson={dav[0]:.4f}  HateXplain={hx[0]:.4f}  "
                                 f"Gap={diff:+.4f}")
        rq_lines.append("")

    # ── RQ3 ──────────────────────────────────────────────────────────────────
    rq_lines.append("\\nRQ3: Is feature representation or model architecture the primary driver")
    rq_lines.append("     of performance differences?")
    rq_lines.append("")
    rq_lines.append("  Full-dataset (Large condition) results:")
    large = MASTER[MASTER["Subset"] == "large"]
    for _, row in large.sort_values(["Dataset", "macro_f1"], ascending=[True, False]).iterrows():
        rq_lines.append(
            f"    {row['Model']:12}  {row['Dataset']:10}  "
            f"Macro_F1={row['macro_f1']:.4f}  "
            f"Params={int(row.get('n_params', 0)):,}"
        )
    rq_lines.append("")
    rq_lines.append("  Interpretation:")
    rq_lines.append("  - If DL consistently outperforms ML on HateXplain but not Davidson,")
    rq_lines.append("    the architecture + contextual embeddings advantage is dataset-conditional.")
    rq_lines.append("  - If the gap holds on both datasets at large scale, architecture and")
    rq_lines.append("    feature representation together drive performance.")
    rq_lines.append("  - If ML achieves competitive results at small sizes on Davidson,")
    rq_lines.append("    TF-IDF feature quality is the key determinant for explicit content.")

    rq_text = "\\n".join(rq_lines)
    print(rq_text)

    with open("results/research_question_answers.txt", "w", encoding="utf-8") as f:
        f.write(rq_text)

    print("\\n✅ Step 16 complete — RQ answers saved to results/research_question_answers.txt")
""")

md("""## Experiment Complete

All 16 steps have been executed. Below is a summary of outputs:

| Output | Location |
|--------|----------|
| Cleaned datasets | `data/davidson_clean.csv`, `data/hatexplain_clean.csv` |
| Preprocessed datasets | `data/davidson_ml.csv`, `data/davidson_transformer.csv`, etc. |
| ML results | `results/metrics/ml_results.csv` |
| BERT results | `results/metrics/bert_results.csv` |
| DistilBERT results | `results/metrics/distilbert_results.csv` |
| McNemar tests | `results/metrics/mcnemar_results.csv` |
| Master results | `results/master_results.csv` |
| Confusion matrices | `results/confusion_matrices/` |
| Learning curves | `results/learning_curves/` |
| Robustness plots | `results/robustness/` |
| Error analysis | `results/error_analysis/` |
| Efficiency summary | `results/metrics/efficiency_summary.csv` |
| RQ answers | `results/research_question_answers.txt` |
""")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
nb = new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.14.0",
    },
}

out_path = "hate_speech_experiment.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"\n✅ Notebook written → {out_path}")
print(f"   Cells: {len(nb.cells)}")
