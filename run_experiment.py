"""
Full experimental pipeline for MSc Dissertation:
"Investigation of ML vs DL Algorithms for Hate Speech Detection"
Runs all steps 0-16 sequentially, saves results to disk as generated.
"""

import os, sys, pathlib, zipfile, json, warnings, random, time, string, re
from collections import Counter
from copy import deepcopy
warnings.filterwarnings("ignore")

# Force unbuffered output so log file is written in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── SSL fix for macOS NLTK downloads ─────────────────────────────────────────
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ── Numeric / data ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

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
import joblib

# ── Stats ──────────────────────────────────────────────────────────────────────
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_test

# ── Deep learning ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed as transformers_set_seed,
)

# ── Visualisation ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")

# ── NLTK ──────────────────────────────────────────────────────────────────────
import nltk
for _r in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.download(_r, quiet=True)
    except Exception:
        pass
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
PROJECT_ROOT = pathlib.Path(__file__).parent.resolve()
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"

def set_seed(seed=SEED):
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

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"{'='*70}")
print(f"  HATE SPEECH DETECTION — FULL EXPERIMENT")
print(f"{'='*70}")
print(f"  Device  : {DEVICE}")
print(f"  PyTorch : {torch.__version__}")
print(f"  Seed    : {SEED}")
print(f"  Root    : {PROJECT_ROOT}")
print(f"  Start   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}\n")

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — PROJECT STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 0 — Creating project structure ──────────────────────────────────")
for d in ["data","results/metrics","results/confusion_matrices",
          "results/learning_curves","results/robustness",
          "results/error_analysis","models/checkpoints"]:
    (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
print("✅ Step 0 complete\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — already done (imports above)
# ─────────────────────────────────────────────────────────────────────────────
print("✅ Step 1 complete — all imports and seeds configured\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 2 — Loading datasets ────────────────────────────────────────────")

# Extract zips
for zp in DATA_DIR.glob("*.zip"):
    try:
        with zipfile.ZipFile(zp) as z:
            z.extractall(DATA_DIR)
        print(f"  Extracted: {zp.name}")
    except Exception as e:
        print(f"  [WARN] Could not extract {zp.name}: {e}")

# Davidson
davidson_csv = list(DATA_DIR.rglob("labeled_data.csv"))
if davidson_csv:
    davidson_raw = pd.read_csv(davidson_csv[0])
else:
    from datasets import load_dataset
    davidson_raw = load_dataset("tdavidson/hate_speech_offensive", split="train").to_pandas()

davidson_clean = davidson_raw[["tweet","class"]].rename(columns={"tweet":"text"}).copy()
davidson_clean["label"] = davidson_clean["class"].apply(lambda x: 1 if x == 0 else 0)
davidson_clean = davidson_clean[["text","label"]].dropna().reset_index(drop=True)
davidson_clean.to_csv(DATA_DIR / "davidson_clean.csv", index=False)

dvc = davidson_clean["label"].value_counts()
print(f"\n  Davidson   : {len(davidson_clean):,} rows")
print(f"    Hate (1)    : {dvc.get(1,0):,} ({100*dvc.get(1,0)/len(davidson_clean):.1f}%)")
print(f"    Non-hate (0): {dvc.get(0,0):,} ({100*dvc.get(0,0)/len(davidson_clean):.1f}%)")

# HateXplain
hx_json = [p for p in DATA_DIR.rglob("*.json") if "hatexplain" in p.name.lower() or "dataset" in p.name.lower()]
if not hx_json:
    hx_json = list(DATA_DIR.rglob("*.json"))

if hx_json:
    with open(hx_json[0], encoding="utf-8") as f:
        hx_raw = json.load(f)
    rows = []
    for pid, pd_ in hx_raw.items():
        text = " ".join(pd_.get("post_tokens",[]))
        anns = [a["label"] for a in pd_.get("annotators",[])]
        if not anns: continue
        majority = Counter(anns).most_common(1)[0][0]
        rows.append({"text": text, "label": 1 if majority == "hatespeech" else 0})
    hatexplain_clean = pd.DataFrame(rows).dropna().reset_index(drop=True)
else:
    from datasets import load_dataset
    hf_hx = load_dataset("Hate-speech-CNERG/hatexplain", split="train+test+validation")
    rows = []
    for item in hf_hx:
        tokens = item.get("post_tokens",[])
        text = " ".join(tokens) if isinstance(tokens, list) else str(tokens)
        anns = item.get("annotators",{}).get("label",[])
        if not anns: continue
        majority = Counter(anns).most_common(1)[0][0]
        rows.append({"text": text, "label": 1 if majority == "hatespeech" else 0})
    hatexplain_clean = pd.DataFrame(rows).dropna().reset_index(drop=True)

hatexplain_clean.to_csv(DATA_DIR / "hatexplain_clean.csv", index=False)
hxc = hatexplain_clean["label"].value_counts()
print(f"\n  HateXplain : {len(hatexplain_clean):,} rows")
print(f"    Hate (1)    : {hxc.get(1,0):,} ({100*hxc.get(1,0)/len(hatexplain_clean):.1f}%)")
print(f"    Non-hate (0): {hxc.get(0,0):,} ({100*hxc.get(0,0)/len(hatexplain_clean):.1f}%)")
print("✅ Step 2 complete\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 3 — Text preprocessing ──────────────────────────────────────────")

CONTRACTIONS = {
    "don't":"do not","dont":"do not","can't":"cannot","cant":"cannot",
    "won't":"will not","wont":"will not","it's":"it is","its":"it is",
    "i'm":"i am","im":"i am","you're":"you are","youre":"you are",
    "they're":"they are","theyre":"they are","we're":"we are","were":"we are",
    "i've":"i have","ive":"i have","i'll":"i will","ill":"i will",
    "isn't":"is not","isnt":"is not","aren't":"are not","arent":"are not",
    "wasn't":"was not","wasnt":"was not","weren't":"were not","werent":"were not",
    "hasn't":"has not","hasnt":"has not","haven't":"have not","havent":"have not",
    "couldn't":"could not","couldnt":"could not","wouldn't":"would not","wouldnt":"would not",
    "shouldn't":"should not","shouldnt":"should not","that's":"that is","thats":"that is",
    "there's":"there is","theres":"there is","he's":"he is","hes":"he is",
    "she's":"she is","shes":"she is","what's":"what is","whats":"what is",
    "let's":"let us","lets":"let us","who's":"who is","whos":"who is",
}

def _expand_contractions(text):
    return " ".join(CONTRACTIONS.get(w, w) for w in text.split())

def preprocess_ml(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+|https\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.lower()
    text = _expand_contractions(text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = text.translate(str.maketrans("","",string.punctuation))
    tokens = [LEMMATIZER.lemmatize(t) for t in text.split() if t not in STOP_WORDS]
    return " ".join(tokens).strip()

def preprocess_transformer(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+|https\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()

sample_texts = [
    "RT @user: I hate all those fucking people!! http://t.co/link",
    "Don't you dare call me that you stupid idiot!!!",
    "She's sooooo beautiful <3 @friend",
    "These immigrants are ruining our country https://news.com",
    "I'm not racist but they're all the same smh",
]
print("  5 before/after examples:\n")
for i, txt in enumerate(sample_texts, 1):
    print(f"  [{i}] ORIGINAL  : {txt}")
    print(f"      ML OUT    : {preprocess_ml(txt)}")
    print(f"      TRANS OUT : {preprocess_transformer(txt)}")
    print()

for name, df in [("davidson", davidson_clean), ("hatexplain", hatexplain_clean)]:
    df["text_ml"]          = df["text"].apply(preprocess_ml)
    df["text_transformer"] = df["text"].apply(preprocess_transformer)
    df[["text_ml","label"]].rename(columns={"text_ml":"text"}).to_csv(DATA_DIR/f"{name}_ml.csv", index=False)
    df[["text_transformer","label"]].rename(columns={"text_transformer":"text"}).to_csv(DATA_DIR/f"{name}_transformer.csv", index=False)
    print(f"  Saved {name}_ml.csv and {name}_transformer.csv")

print("✅ Step 3 complete\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SPLITS
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 4 — Dataset splitting ───────────────────────────────────────────")
set_seed(SEED)

DATASETS    = {"davidson": davidson_clean, "hatexplain": hatexplain_clean}
SUBSET_NAMES = ["small","medium","large"]
SPLITS      = {}
TEST_SETS   = {}

for ds_name, df_full in DATASETS.items():
    N      = len(df_full)
    test_n = int(round(N * 0.20))
    pool, test_df = train_test_split(
        df_full, test_size=test_n/N, stratify=df_full["label"], random_state=SEED
    )
    pool    = pool.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    TEST_SETS[ds_name] = test_df
    SPLITS[ds_name]    = {}

    targets = {
        "small":  int(round(N * 0.20)),
        "medium": int(round(N * 0.50)),
        "large":  len(pool),
    }
    for subset in SUBSET_NAMES:
        tn = targets[subset]
        if subset == "large":
            tp = pool.copy().reset_index(drop=True)
        else:
            _, tp = train_test_split(
                pool, test_size=tn/len(pool), stratify=pool["label"], random_state=SEED
            )
            tp = tp.reset_index(drop=True)
        ml_train = tp.copy()
        val_n    = max(2, int(round(len(tp) * 0.10)))
        tr_train, tr_val = train_test_split(
            tp, test_size=val_n/len(tp), stratify=tp["label"], random_state=SEED
        )
        SPLITS[ds_name][subset] = {
            "ml_train": ml_train.reset_index(drop=True),
            "tr_train": tr_train.reset_index(drop=True),
            "tr_val":   tr_val.reset_index(drop=True),
            "test":     test_df,
        }

print(f"\n{'Dataset':10} | {'Subset':8} | {'ML Train':8} | {'TR Train':8} | {'Val':5} | {'Test':5} | {'Hate%':6} | {'NonHate%'}")
print("="*80)
for ds in ["davidson","hatexplain"]:
    for sub in SUBSET_NAMES:
        s  = SPLITS[ds][sub]
        hp = 100*s["ml_train"]["label"].mean()
        print(f"{ds:10} | {sub:8} | {len(s['ml_train']):8} | {len(s['tr_train']):8} | "
              f"{len(s['tr_val']):5} | {len(s['test']):5} | {hp:5.1f}% | {100-hp:.1f}%")
print("✅ Step 4 complete\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — TF-IDF
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 5 — TF-IDF feature extraction ───────────────────────────────────")
set_seed(SEED)
TFIDF_FEATURES = {}

for ds_name in ["davidson","hatexplain"]:
    for subset in SUBSET_NAMES:
        split    = SPLITS[ds_name][subset]
        tr_texts = [preprocess_ml(t) for t in split["ml_train"]["text"].tolist()]
        te_texts = [preprocess_ml(t) for t in split["test"]["text"].tolist()]
        y_tr     = split["ml_train"]["label"].values
        y_te     = split["test"]["label"].values
        vec      = TfidfVectorizer(ngram_range=(1,2), max_features=50_000, sublinear_tf=True)
        X_tr     = vec.fit_transform(tr_texts)
        X_te     = vec.transform(te_texts)
        TFIDF_FEATURES[(ds_name,subset)] = {
            "X_train":X_tr,"y_train":y_tr,"X_test":X_te,"y_test":y_te
        }
        vpath = MODELS_DIR / f"tfidf_{ds_name}_{subset}.pkl"
        joblib.dump(vec, vpath)
        print(f"  [{ds_name} | {subset:6}] vocab={len(vec.vocabulary_):,}  "
              f"train={X_tr.shape}  test={X_te.shape}")
print("✅ Step 5 complete\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — ML TRAINING
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 6 — ML model training (LR + SVM × 6 conditions) ─────────────────")

def bootstrap_ci(yt, yp, yprob, metric_fn, n_iter=1000, seed=SEED):
    rng = np.random.RandomState(seed); scores = []
    for _ in range(n_iter):
        idx = rng.randint(0, len(yt), len(yt))
        try:
            scores.append(metric_fn(yt[idx], yp[idx], yprob[idx]))
        except Exception:
            pass
    return (np.percentile(scores, 2.5), np.percentile(scores, 97.5)) if scores else (0.0, 0.0)

f1_macro_fn = lambda yt,yp,_:   f1_score(yt, yp, average="macro", zero_division=0)
auc_fn      = lambda yt,_,yprob: roc_auc_score(yt, yprob) if len(np.unique(yt))>1 else 0.0

def compute_metrics(y_true, y_pred, y_prob, train_time, infer_time, n_params=None):
    acc  = accuracy_score(y_true, y_pred)
    mf1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    wf1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mp   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mr   = recall_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    p_cls = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_cls = recall_score(y_true, y_pred, average=None, zero_division=0)
    f_cls = f1_score(y_true, y_pred, average=None, zero_division=0)
    p0,p1   = (p_cls[0],p_cls[1]) if len(p_cls)>=2 else (p_cls[0],0)
    r0,r1   = (r_cls[0],r_cls[1]) if len(r_cls)>=2 else (r_cls[0],0)
    f0,f1_  = (f_cls[0],f_cls[1]) if len(f_cls)>=2 else (f_cls[0],0)
    f1lo,f1hi   = bootstrap_ci(y_true, y_pred, y_prob, f1_macro_fn)
    auclo,auchi = bootstrap_ci(y_true, y_pred, y_prob, auc_fn)
    cm = confusion_matrix(y_true, y_pred)
    return dict(
        accuracy=acc, macro_precision=mp, macro_recall=mr,
        macro_f1=mf1, macro_f1_ci_lo=f1lo, macro_f1_ci_hi=f1hi,
        weighted_f1=wf1, auc_roc=auc, auc_ci_lo=auclo, auc_ci_hi=auchi,
        hate_precision=p1, hate_recall=r1, hate_f1=f1_,
        nonhate_precision=p0, nonhate_recall=r0, nonhate_f1=f0,
        train_time=train_time, infer_time=infer_time,
        n_params=n_params if n_params is not None else 0,
        confusion_matrix=cm,
        report=classification_report(y_true, y_pred,
                                     target_names=["Non-Hate","Hate"], zero_division=0),
    )

ML_RESULTS = []
print(f"\n{'Model':5} | {'Dataset':10} | {'Subset':8} | {'MacroF1':8} | {'AUC':8} | {'HateF1':7} | {'Train(s)'}")
print("-"*65)
for ds_name in ["davidson","hatexplain"]:
    for subset in SUBSET_NAMES:
        feat = TFIDF_FEATURES[(ds_name,subset)]
        Xtr,ytr,Xte,yte = feat["X_train"],feat["y_train"],feat["X_test"],feat["y_test"]
        for mname, clf in [
            ("LR",  LogisticRegression(solver="lbfgs",max_iter=1000,
                                       class_weight="balanced",C=1.0,random_state=SEED)),
            ("SVM", CalibratedClassifierCV(LinearSVC(C=1.0,class_weight="balanced",
                                                      max_iter=2000,random_state=SEED))),
        ]:
            set_seed(SEED)
            t0=time.time(); clf.fit(Xtr,ytr); tt=time.time()-t0
            t1=time.time(); yp=clf.predict(Xte); yprob=clf.predict_proba(Xte)[:,1]; it=time.time()-t1
            m = compute_metrics(yte,yp,yprob,tt,it)
            pk = f"{mname.lower()}_{ds_name}_{subset}"
            np.save(RESULTS_DIR/f"metrics/{pk}_predictions.npy", yp)
            np.save(RESULTS_DIR/f"metrics/{pk}_proba.npy",       yprob)
            np.save(RESULTS_DIR/f"metrics/{pk}_cm.npy",          m["confusion_matrix"])
            ML_RESULTS.append({"Model":mname,"Dataset":ds_name,"Subset":subset,
                "Train_Size":len(ytr),"Test_Size":len(yte),
                **{k:v for k,v in m.items() if k not in("confusion_matrix","report")}})
            print(f"{mname:5} | {ds_name:10} | {subset:8} | {m['macro_f1']:.4f}   | "
                  f"{m['auc_roc']:.4f}   | {m['hate_f1']:.4f}  | {tt:.2f}s")

pd.DataFrame(ML_RESULTS).to_csv(RESULTS_DIR/"metrics/ml_results.csv", index=False)
print(f"\n✅ Step 6 complete — ml_results.csv saved ({len(ML_RESULTS)} rows)\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEPS 7 & 8 — TRANSFORMER FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────
class HateSpeechDataset(Dataset):
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
    set_seed(SEED)
    print(f"\n{'='*65}")
    print(f"  {checkpoint_prefix.upper()} | {ds_name} | {subset}  [{time.strftime('%H:%M:%S')}]")
    print(f"{'='*65}")

    split   = SPLITS[ds_name][subset]
    epochs  = 4 if subset == "large" else 3

    tr_texts  = [preprocess_transformer(t) for t in split["tr_train"]["text"].tolist()]
    val_texts = [preprocess_transformer(t) for t in split["tr_val"]["text"].tolist()]
    te_texts  = [preprocess_transformer(t) for t in split["test"]["text"].tolist()]
    y_tr  = split["tr_train"]["label"].values
    y_val = split["tr_val"]["label"].values
    y_te  = split["test"]["label"].values

    tokenizer = AutoTokenizer.from_pretrained(model_name_hf)
    train_ds  = HateSpeechDataset(tr_texts, y_tr, tokenizer)
    val_ds    = HateSpeechDataset(val_texts, y_val, tokenizer)
    test_ds   = HateSpeechDataset(te_texts, y_te, tokenizer)

    def make_loaders(bs):
        return (DataLoader(train_ds, batch_size=bs, shuffle=True),
                DataLoader(val_ds,   batch_size=bs, shuffle=False),
                DataLoader(test_ds,  batch_size=bs, shuffle=False))

    try:
        tr_loader, val_loader, te_loader = make_loaders(batch_size)
    except Exception:
        batch_size = 8
        tr_loader, val_loader, te_loader = make_loaders(batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_hf, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  |  Batch: {batch_size}  |  Epochs: {epochs}")

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps  = len(tr_loader) * epochs
    warmup_steps = int(0.10 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_f1  = -1.0
    best_state   = None
    no_improve   = 0
    patience     = 2
    total_train_t = 0.0
    ckpt_dir = PROJECT_ROOT / f"models/checkpoints/{checkpoint_prefix}_{ds_name}_{subset}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        ep_loss = 0.0
        t_ep    = time.time()
        for batch in tr_loader:
            optimizer.zero_grad()
            try:
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                ep_loss += out.loss.item()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n  OOM — reducing batch to 8 and restarting condition")
                    del model
                    if device.type == "mps":   torch.mps.empty_cache()
                    elif device.type == "cuda": torch.cuda.empty_cache()
                    return train_transformer(model_name_hf, ds_name, subset,
                                             checkpoint_prefix, result_list,
                                             batch_size=8, device=device)
                raise

        ep_t = time.time() - t_ep
        total_train_t += ep_t

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                out   = model(input_ids=batch["input_ids"].to(device),
                              attention_mask=batch["attention_mask"].to(device))
                preds = out.logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds.tolist())

        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        avg_loss = ep_loss / max(len(tr_loader),1)
        print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  val_F1={val_f1:.4f}  "
              f"time={ep_t:.0f}s  [{time.strftime('%H:%M:%S')}]", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop: no improvement for {patience} epochs")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Test inference
    model.eval()
    te_preds, te_probs = [], []
    t_inf = time.time()
    with torch.no_grad():
        for batch in te_loader:
            out   = model(input_ids=batch["input_ids"].to(device),
                          attention_mask=batch["attention_mask"].to(device))
            probs = torch.softmax(out.logits, dim=-1)[:,1].cpu().numpy()
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            te_preds.extend(preds.tolist())
            te_probs.extend(probs.tolist())
    it = time.time() - t_inf

    te_preds = np.array(te_preds)
    te_probs = np.array(te_probs)
    m = compute_metrics(y_te, te_preds, te_probs, total_train_t, it, n_params)

    print(f"\n  ★ TEST → MacroF1={m['macro_f1']:.4f}  AUC={m['auc_roc']:.4f}  "
          f"HateF1={m['hate_f1']:.4f}  train={total_train_t:.0f}s  infer={it:.2f}s")

    pk = f"{checkpoint_prefix}_{ds_name}_{subset}"
    np.save(RESULTS_DIR/f"metrics/{pk}_predictions.npy", te_preds)
    np.save(RESULTS_DIR/f"metrics/{pk}_proba.npy",       te_probs)
    np.save(RESULTS_DIR/f"metrics/{pk}_cm.npy",          m["confusion_matrix"])
    torch.save(best_state, ckpt_dir/"best_model.pt")

    result_list.append({"Model": checkpoint_prefix.upper(), "Dataset": ds_name,
        "Subset": subset, "Train_Size": len(y_tr), "Test_Size": len(y_te),
        **{k:v for k,v in m.items() if k not in("confusion_matrix","report")}})

    del model
    if device.type == "mps":   torch.mps.empty_cache()
    elif device.type == "cuda": torch.cuda.empty_cache()


# ── Step 7 — BERT ─────────────────────────────────────────────────────────────
print("\n── STEP 7 — BERT fine-tuning (6 conditions) ─────────────────────────────")
BERT_RESULTS = []
for ds_name in ["davidson","hatexplain"]:
    for subset in SUBSET_NAMES:
        train_transformer("bert-base-uncased", ds_name, subset,
                          "bert", BERT_RESULTS, device=DEVICE)

pd.DataFrame(BERT_RESULTS).to_csv(RESULTS_DIR/"metrics/bert_results.csv", index=False)
print(f"\n✅ Step 7 complete — bert_results.csv saved ({len(BERT_RESULTS)} rows)\n")

# ── Step 8 — DistilBERT ───────────────────────────────────────────────────────
print("── STEP 8 — DistilBERT fine-tuning (6 conditions) ───────────────────────")
DISTILBERT_RESULTS = []
for ds_name in ["davidson","hatexplain"]:
    for subset in SUBSET_NAMES:
        train_transformer("distilbert-base-uncased", ds_name, subset,
                          "distilbert", DISTILBERT_RESULTS, device=DEVICE)

pd.DataFrame(DISTILBERT_RESULTS).to_csv(RESULTS_DIR/"metrics/distilbert_results.csv", index=False)
print(f"\n✅ Step 8 complete — distilbert_results.csv saved ({len(DISTILBERT_RESULTS)} rows)\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — McNemar's TEST
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 9 — McNemar's significance tests ────────────────────────────────")
MODEL_NAMES = ["lr","svm","bert","distilbert"]
MODEL_PAIRS = [("lr","svm"),("lr","bert"),("lr","distilbert"),
               ("svm","bert"),("svm","distilbert"),("bert","distilbert")]

mcnemar_rows = []
print(f"\n{'Model A':12} {'Model B':12} {'Dataset':10} {'Subset':8} {'Stat':8} {'p-val':8} {'Sig.'}")
print("-"*70)
for ds_name in ["davidson","hatexplain"]:
    for subset in SUBSET_NAMES:
        y_te = SPLITS[ds_name][subset]["test"]["label"].values
        for ma, mb in MODEL_PAIRS:
            try:
                pa = np.load(RESULTS_DIR/f"metrics/{ma}_{ds_name}_{subset}_predictions.npy")
                pb = np.load(RESULTS_DIR/f"metrics/{mb}_{ds_name}_{subset}_predictions.npy")
                n00=int(((pa==y_te)&(pb==y_te)).sum())
                n01=int(((pa==y_te)&(pb!=y_te)).sum())
                n10=int(((pa!=y_te)&(pb==y_te)).sum())
                n11=int(((pa!=y_te)&(pb!=y_te)).sum())
                res = mcnemar_test([[n00,n01],[n10,n11]], exact=False, correction=True)
                sig = "Yes" if res.pvalue < 0.05 else "No"
                mcnemar_rows.append({"Model_A":ma.upper(),"Model_B":mb.upper(),
                    "Dataset":ds_name,"Subset":subset,
                    "McNemar_Stat":round(res.statistic,4),"p_value":round(res.pvalue,6),
                    "Significant":sig,"n01":n01,"n10":n10})
                print(f"{ma.upper():12} {mb.upper():12} {ds_name:10} {subset:8} "
                      f"{res.statistic:8.3f} {res.pvalue:8.4f} {sig}")
            except Exception as e:
                print(f"  [WARN] {ma} vs {mb} | {ds_name} | {subset}: {e}")

mcnemar_df = pd.DataFrame(mcnemar_rows)
mcnemar_df.to_csv(RESULTS_DIR/"metrics/mcnemar_results.csv", index=False)
print(f"\n✅ Step 9 complete — mcnemar_results.csv saved ({len(mcnemar_rows)} rows)\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 10 — Confusion matrix visualisations ────────────────────────────")
CM_DIR      = RESULTS_DIR / "confusion_matrices"
MODEL_LABELS = {"lr":"Logistic Regression","svm":"Linear SVM",
                "bert":"BERT","distilbert":"DistilBERT"}
SUBSET_LABELS= {"small":"Small (20%)","medium":"Medium (50%)","large":"Large (100%)"}
TICK_LABELS  = ["Non-Hate (0)","Hate (1)"]

def plot_cm(cm, title, save_path, ax=None):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct   = np.where(row_sums>0, cm/row_sums*100, 0)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax,
                xticklabels=TICK_LABELS, yticklabels=TICK_LABELS,
                linewidths=0.5, linecolor="grey", cbar=True)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j+0.5, i+0.5,
                    f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm[i,j] > cm.max()*0.6 else "black")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label",      fontsize=9)
    if standalone:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()

for mk in MODEL_NAMES:
    for ds in ["davidson","hatexplain"]:
        for sub in SUBSET_NAMES:
            try:
                cm   = np.load(RESULTS_DIR/f"metrics/{mk}_{ds}_{sub}_cm.npy")
                title = f"{MODEL_LABELS[mk]} | {ds.capitalize()} | {SUBSET_LABELS[sub]}"
                plot_cm(cm, title, CM_DIR/f"cm_{mk}_{ds}_{sub}.png")
            except Exception as e:
                print(f"  [WARN] CM {mk}_{ds}_{sub}: {e}")

# Combined 4×6 grid
fig, axes = plt.subplots(4, 6, figsize=(28,20))
fig.suptitle("All 24 Confusion Matrices — Hate Speech Detection Experiment",
             fontsize=16, y=1.01)
for ri, mk in enumerate(MODEL_NAMES):
    for ci, (ds,sub) in enumerate([(d,s) for d in ["davidson","hatexplain"] for s in SUBSET_NAMES]):
        ax = axes[ri,ci]
        try:
            cm    = np.load(RESULTS_DIR/f"metrics/{mk}_{ds}_{sub}_cm.npy")
            title = f"{MODEL_LABELS[mk]}\n{ds.capitalize()} {SUBSET_LABELS[sub]}"
            plot_cm(cm, title, None, ax=ax)
        except Exception:
            ax.set_title(f"{mk}\n{ds} {sub}\n(missing)", fontsize=7)
            ax.axis("off")
plt.tight_layout()
plt.savefig(CM_DIR/"all_confusion_matrices.png", dpi=90, bbox_inches="tight")
plt.close()
print("✅ Step 10 complete — confusion matrices saved\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — LEARNING CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 11 — Learning curve visualisations ──────────────────────────────")
LC_DIR = RESULTS_DIR / "learning_curves"

all_dfs = []
for fp in [RESULTS_DIR/"metrics/ml_results.csv",
           RESULTS_DIR/"metrics/bert_results.csv",
           RESULTS_DIR/"metrics/distilbert_results.csv"]:
    try:
        all_dfs.append(pd.read_csv(fp))
    except Exception:
        pass
ALL_RESULTS = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

MC = {"LR":"#1f77b4","SVM":"#ff7f0e","BERT":"#2ca02c","DISTILBERT":"#d62728"}
MM = {"LR":"o","SVM":"s","BERT":"^","DISTILBERT":"D"}
XP = [0.20,0.50,1.00]
SL = {"small":"Small\n(20%)","medium":"Medium\n(50%)","large":"Large\n(100%)"}

def make_lc(ds_name, ax):
    ds_r = ALL_RESULTS[ALL_RESULTS["Dataset"]==ds_name]
    for m in ["LR","SVM","BERT","DISTILBERT"]:
        mr = ds_r[ds_r["Model"]==m].copy()
        if mr.empty: continue
        mr["x"] = mr["Subset"].map({"small":0.20,"medium":0.50,"large":1.00})
        mr = mr.sort_values("x")
        xs=mr["x"].values; ys=mr["macro_f1"].values
        los=mr.get("macro_f1_ci_lo", mr["macro_f1"]*0.98).values
        his=mr.get("macro_f1_ci_hi", mr["macro_f1"]*1.02).values
        ax.plot(xs,ys,color=MC[m],marker=MM[m],linewidth=2,markersize=8,label=m)
        ax.fill_between(xs,los,his,alpha=0.15,color=MC[m])
    ax.set_xlim(0.10,1.10); ax.set_ylim(0,1)
    ax.set_xticks(XP); ax.set_xticklabels([SL[s] for s in SUBSET_NAMES])
    ax.set_xlabel("Training Subset Size",fontsize=11)
    ax.set_ylabel("Macro F1-Score",fontsize=11)
    ax.set_title(f"Learning Curves — {ds_name.capitalize()}\nMacro F1-Score vs Training Dataset Size",fontsize=12)
    ax.legend(loc="lower right",fontsize=9); ax.grid(True,alpha=0.4)

for ds in ["davidson","hatexplain"]:
    fig,ax = plt.subplots(figsize=(10,6))
    make_lc(ds,ax); plt.tight_layout()
    plt.savefig(LC_DIR/f"learning_curve_{ds}.png",dpi=120,bbox_inches="tight"); plt.close()

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,7))
make_lc("davidson",ax1); make_lc("hatexplain",ax2)
fig.suptitle("Learning Curves: ML vs DL Models on Both Datasets",fontsize=14,y=1.02)
plt.tight_layout()
plt.savefig(LC_DIR/"learning_curves_combined.png",dpi=120,bbox_inches="tight"); plt.close()
print("✅ Step 11 complete — learning curves saved\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 12 — ROBUSTNESS
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 12 — Robustness analysis ────────────────────────────────────────")
ROB_DIR = RESULTS_DIR / "robustness"

# Analysis 1
degrad_rows = []
print("\n  Analysis 1 — Degradation Large → Small:")
for m in ["LR","SVM","BERT","DISTILBERT"]:
    for ds in ["davidson","hatexplain"]:
        sub = ALL_RESULTS[(ALL_RESULTS["Model"]==m)&(ALL_RESULTS["Dataset"]==ds)]
        f1l = sub[sub["Subset"]=="large"]["macro_f1"].values
        f1s = sub[sub["Subset"]=="small"]["macro_f1"].values
        if len(f1l) and len(f1s):
            drop = float(f1l[0])-float(f1s[0])
            degrad_rows.append({"Model":m,"Dataset":ds,"F1_Large":round(float(f1l[0]),4),
                "F1_Small":round(float(f1s[0]),4),"Degradation":round(drop,4)})
            print(f"    {m:12} | {ds:10} | Large={f1l[0]:.4f} Small={f1s[0]:.4f} Drop={drop:+.4f}")

pd.DataFrame(degrad_rows).to_csv(ROB_DIR/"performance_degradation.csv", index=False)

# Analysis 2 — cross-dataset gap heatmap
print("\n  Analysis 2 — Cross-dataset gap:")
gap_data = {}
for m in ["LR","SVM","BERT","DISTILBERT"]:
    gap_data[m] = {}
    for sub in SUBSET_NAMES:
        dv = ALL_RESULTS[(ALL_RESULTS["Model"]==m)&(ALL_RESULTS["Dataset"]=="davidson")&(ALL_RESULTS["Subset"]==sub)]["macro_f1"].values
        hx = ALL_RESULTS[(ALL_RESULTS["Model"]==m)&(ALL_RESULTS["Dataset"]=="hatexplain")&(ALL_RESULTS["Subset"]==sub)]["macro_f1"].values
        if len(dv) and len(hx):
            g = float(dv[0])-float(hx[0])
            gap_data[m][sub] = round(g,4)
            print(f"    {m:12} | {sub:6} | gap={g:+.4f}")

gap_df = pd.DataFrame(gap_data).T[SUBSET_NAMES] if gap_data else pd.DataFrame()
if not gap_df.empty:
    fig,ax = plt.subplots(figsize=(8,5))
    sns.heatmap(gap_df.astype(float),annot=True,fmt=".3f",cmap="RdYlGn_r",center=0,ax=ax,
                xticklabels=["Small","Medium","Large"],yticklabels=gap_df.index)
    ax.set_title("Cross-Dataset Gap: Davidson F1 − HateXplain F1\n(Positive = Davidson higher)")
    ax.set_xlabel("Training Subset"); ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(ROB_DIR/"cross_dataset_gap.png",dpi=120,bbox_inches="tight"); plt.close()

# Analysis 3 — class-wise F1
fig,axes = plt.subplots(2,3,figsize=(18,10))
fig.suptitle("Class-wise F1 Comparison — All Models, Datasets, Conditions",fontsize=14)
for di,ds in enumerate(["davidson","hatexplain"]):
    for si,sub in enumerate(SUBSET_NAMES):
        ax = axes[di][si]
        sr = ALL_RESULTS[(ALL_RESULTS["Dataset"]==ds)&(ALL_RESULTS["Subset"]==sub)]
        models = sr["Model"].tolist()
        hf  = sr.get("hate_f1",   pd.Series([])).tolist()
        nhf = sr.get("nonhate_f1",pd.Series([])).tolist()
        x = np.arange(len(models))
        if models:
            ax.bar(x-0.175,hf, 0.35,label="Hate",    color="#e74c3c",alpha=0.8)
            ax.bar(x+0.175,nhf,0.35,label="Non-Hate",color="#3498db",alpha=0.8)
            ax.set_xticks(x); ax.set_xticklabels(models,fontsize=8)
        ax.set_ylim(0,1)
        ax.set_title(f"{ds.capitalize()} — {SUBSET_LABELS[sub]}",fontsize=9)
        ax.legend(fontsize=7); ax.grid(axis="y",alpha=0.3)
plt.tight_layout()
plt.savefig(ROB_DIR/"classwise_f1_comparison.png",dpi=110,bbox_inches="tight"); plt.close()
print("\n✅ Step 12 complete — robustness analyses saved\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 13 — COMPUTATIONAL EFFICIENCY
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 13 — Computational efficiency ──────────────────────────────────")

# Chart 1: Training time
fig,axes = plt.subplots(1,2,figsize=(16,6))
fig.suptitle("Training Time Comparison",fontsize=14)
for di,ds in enumerate(["davidson","hatexplain"]):
    ax = axes[di]
    models = ["LR","SVM","BERT","DISTILBERT"]
    x = np.arange(len(models))
    for si,sub in enumerate(SUBSET_NAMES):
        vals = [float(ALL_RESULTS[(ALL_RESULTS["Model"]==m)&(ALL_RESULTS["Dataset"]==ds)&(ALL_RESULTS["Subset"]==sub)]["train_time"].values[0]) if not ALL_RESULTS[(ALL_RESULTS["Model"]==m)&(ALL_RESULTS["Dataset"]==ds)&(ALL_RESULTS["Subset"]==sub)].empty else 0 for m in models]
        ax.bar(x+(si-1)*0.25,vals,0.25,label=SUBSET_LABELS[sub],alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_title(ds.capitalize()); ax.set_xlabel("Model"); ax.set_ylabel("Training Time (s)")
    ax.legend(fontsize=9); ax.grid(axis="y",alpha=0.4)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"training_time_comparison.png",dpi=120,bbox_inches="tight"); plt.close()

# Chart 2: Inference time
fig,ax = plt.subplots(figsize=(10,5))
models = ["LR","SVM","BERT","DISTILBERT"]
for di,ds in enumerate(["davidson","hatexplain"]):
    it_vals = []
    for m in models:
        r = ALL_RESULTS[(ALL_RESULTS["Model"]==m)&(ALL_RESULTS["Dataset"]==ds)&(ALL_RESULTS["Subset"]=="large")]["infer_time"].values
        it_vals.append(float(r[0]) if len(r) else 0)
    ax.bar(np.arange(len(models))+di*0.35,it_vals,0.35,label=ds.capitalize(),alpha=0.8)
ax.set_xticks(np.arange(len(models))+0.175); ax.set_xticklabels(models)
ax.set_xlabel("Model"); ax.set_ylabel("Inference Time (seconds)")
ax.set_title("Inference Time on Test Set (Large condition)")
ax.legend(); ax.grid(axis="y",alpha=0.4)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"inference_time_comparison.png",dpi=120,bbox_inches="tight"); plt.close()

# Chart 3: Model size vs performance
fig,ax = plt.subplots(figsize=(9,6))
sc = {"LR":"#1f77b4","SVM":"#ff7f0e","BERT":"#2ca02c","DISTILBERT":"#d62728"}
for m in ["LR","SVM","BERT","DISTILBERT"]:
    r = ALL_RESULTS[(ALL_RESULTS["Model"]==m)&(ALL_RESULTS["Subset"]=="large")]
    if r.empty: continue
    mf1 = r["macro_f1"].mean()
    np_ = r["n_params"].mean() if "n_params" in r.columns else 0
    ax.scatter(np_,mf1,s=200,color=sc[m],label=m,zorder=5)
    ax.annotate(m,(np_,mf1),textcoords="offset points",xytext=(10,5),fontsize=11)
ax.set_xlabel("Trainable Parameters",fontsize=11)
ax.set_ylabel("Mean Macro F1 (Large)",fontsize=11)
ax.set_title("Model Size vs Performance")
ax.legend(fontsize=10); ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig(RESULTS_DIR/"model_size_vs_performance.png",dpi=120,bbox_inches="tight"); plt.close()

# Efficiency summary CSV
eff_rows = []
for m in ["LR","SVM","BERT","DISTILBERT"]:
    for ds in ["davidson","hatexplain"]:
        r = ALL_RESULTS[(ALL_RESULTS["Model"]==m)&(ALL_RESULTS["Dataset"]==ds)&(ALL_RESULTS["Subset"]=="large")]
        if r.empty: continue
        row = r.iloc[0]
        tt=float(row.get("train_time",0)); it=float(row.get("infer_time",0))
        mf1=float(row.get("macro_f1",0)); np_=int(row.get("n_params",0))
        eff_rows.append({"Model":m,"Dataset":ds,"Train_Time_s":round(tt,2),
            "Infer_Time_s":round(it,4),"Parameters":np_,
            "Macro_F1_Large":round(mf1,4),"F1_per_TrainSec":round(mf1/max(tt,1e-9),6)})
eff_df = pd.DataFrame(eff_rows)
eff_df.to_csv(RESULTS_DIR/"metrics/efficiency_summary.csv", index=False)
print(eff_df.to_string(index=False))
print("✅ Step 13 complete\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 14 — ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 14 — Error analysis (large condition) ───────────────────────────")
ERR_DIR = RESULTS_DIR / "error_analysis"
for mk in MODEL_NAMES:
    for ds in ["davidson","hatexplain"]:
        try:
            yp     = np.load(RESULTS_DIR/f"metrics/{mk}_{ds}_large_predictions.npy")
            y_true = SPLITS[ds]["large"]["test"]["label"].values
            raw_texts = SPLITS[ds]["large"]["test"]["text"].values
            fp_mask = (yp==1)&(y_true==0)
            fn_mask = (yp==0)&(y_true==1)
            print(f"\n  {mk.upper():12} | {ds:10} | FP={fp_mask.sum()}  FN={fn_mask.sum()}  "
                  f"total_err={fp_mask.sum()+fn_mask.sum()}")
            rows = []
            for idx in np.where(fp_mask)[0][:20]:
                rows.append({"error_type":"FP","true_label":int(y_true[idx]),
                    "pred_label":int(yp[idx]),"raw_text":raw_texts[idx],
                    "ml_text":preprocess_ml(raw_texts[idx]),
                    "trans_text":preprocess_transformer(raw_texts[idx])})
            for idx in np.where(fn_mask)[0][:20]:
                rows.append({"error_type":"FN","true_label":int(y_true[idx]),
                    "pred_label":int(yp[idx]),"raw_text":raw_texts[idx],
                    "ml_text":preprocess_ml(raw_texts[idx]),
                    "trans_text":preprocess_transformer(raw_texts[idx])})
            pd.DataFrame(rows).to_csv(ERR_DIR/f"{mk}_{ds}_errors.csv", index=False)
            for etype in ["FP","FN"]:
                examples = [r for r in rows if r["error_type"]==etype][:3]
                if examples:
                    print(f"    {etype}:", examples[0]["raw_text"][:100])
        except Exception as e:
            print(f"  [WARN] Error analysis {mk}_{ds}: {e}")
print("\n✅ Step 14 complete\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 15 — MASTER RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 15 — Master results table ──────────────────────────────────────")
parts = []
for fp in [RESULTS_DIR/"metrics/ml_results.csv",
           RESULTS_DIR/"metrics/bert_results.csv",
           RESULTS_DIR/"metrics/distilbert_results.csv"]:
    try: parts.append(pd.read_csv(fp))
    except: pass

MASTER = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

if not MASTER.empty and not mcnemar_df.empty:
    for tm in ["LR","SVM","BERT","DISTILBERT"]:
        col = f"McNemar_vs_{tm}_p"
        MASTER[col] = None
        for i,row in MASTER.iterrows():
            mA=str(row.get("Model","")).upper(); ds=str(row.get("Dataset","")); sub=str(row.get("Subset",""))
            match = mcnemar_df[
                (((mcnemar_df["Model_A"]==mA)&(mcnemar_df["Model_B"]==tm))|
                 ((mcnemar_df["Model_A"]==tm) &(mcnemar_df["Model_B"]==mA)))&
                (mcnemar_df["Dataset"]==ds)&(mcnemar_df["Subset"]==sub)]
            if not match.empty:
                MASTER.at[i,col] = match.iloc[0]["p_value"]

if not MASTER.empty:
    MASTER = MASTER.sort_values(["Dataset","Subset","Model"]).reset_index(drop=True)
    MASTER.to_csv(RESULTS_DIR/"master_results.csv", index=False)
    disp_cols = [c for c in ["Model","Dataset","Subset","Train_Size","macro_f1",
                              "macro_f1_ci_lo","macro_f1_ci_hi","auc_roc","hate_f1","train_time"]
                 if c in MASTER.columns]
    print("\n" + "="*90)
    print("MASTER RESULTS TABLE")
    print("="*90)
    print(MASTER[disp_cols].to_string(index=False))
    print(f"\nTotal rows: {len(MASTER)}")
print("✅ Step 15 complete\n")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 16 — RESEARCH QUESTION ANSWERS
# ─────────────────────────────────────────────────────────────────────────────
print("── STEP 16 — Research question answers ──────────────────────────────────")
SUBSET_LABELS2 = {"small":"Small (20%)","medium":"Medium (50%)","large":"Large (100%)"}

if not MASTER.empty:
    lines = ["="*80,
             "RESEARCH QUESTION ANSWERS — BASED ON EXPERIMENTAL RESULTS",
             "="*80]

    lines += ["","RQ1: How does ML vs DL comparative performance change as dataset size increases?",""]
    for ds in ["davidson","hatexplain"]:
        lines.append(f"  Dataset: {ds.capitalize()}")
        for m in ["LR","SVM","BERT","DISTILBERT"]:
            row_data = MASTER[(MASTER["Dataset"]==ds)&(MASTER["Model"]==m)]
            line = f"    {m:12}"
            for sub in ["small","medium","large"]:
                r=row_data[row_data["Subset"]==sub]["macro_f1"].values
                line += f"  {sub}={float(r[0]):.4f}" if len(r) else f"  {sub}=N/A"
            lines.append(line)
        for sub in ["small","medium","large"]:
            ml_f = MASTER[(MASTER["Dataset"]==ds)&(MASTER["Subset"]==sub)&(MASTER["Model"].isin(["LR","SVM"]))]["macro_f1"].max()
            dl_f = MASTER[(MASTER["Dataset"]==ds)&(MASTER["Subset"]==sub)&(MASTER["Model"].isin(["BERT","DISTILBERT"]))]["macro_f1"].max()
            gap  = (dl_f-ml_f) if not (pd.isna(dl_f) or pd.isna(ml_f)) else 0
            lines.append(f"    [{sub:6}] Best_ML={ml_f:.4f}  Best_DL={dl_f:.4f}  DL_advantage={gap:+.4f}")
        lines.append("")

    lines += ["","RQ2: Does linguistic complexity moderate the ML vs DL performance gap?",""]
    for sub in SUBSET_NAMES:
        lines.append(f"  Subset: {SUBSET_LABELS2[sub]}")
        for m in ["LR","SVM","BERT","DISTILBERT"]:
            dav=MASTER[(MASTER["Dataset"]=="davidson")&(MASTER["Model"]==m)&(MASTER["Subset"]==sub)]["macro_f1"].values
            hx =MASTER[(MASTER["Dataset"]=="hatexplain")&(MASTER["Model"]==m)&(MASTER["Subset"]==sub)]["macro_f1"].values
            if len(dav) and len(hx):
                diff=float(dav[0])-float(hx[0])
                lines.append(f"    {m:12}  Davidson={dav[0]:.4f}  HateXplain={hx[0]:.4f}  Gap={diff:+.4f}")
        lines.append("")

    lines += ["","RQ3: Feature representation or model architecture — primary driver?","",
              "  Full-dataset (Large) results:"]
    large = MASTER[MASTER["Subset"]=="large"]
    for _,row in large.sort_values(["Dataset","macro_f1"],ascending=[True,False]).iterrows():
        lines.append(f"    {row['Model']:12}  {row['Dataset']:10}  "
                     f"MacroF1={row['macro_f1']:.4f}  Params={int(row.get('n_params',0)):,}")
    lines += ["",
              "  Interpretation:",
              "  - Consistent DL>ML on HateXplain but not Davidson → architecture+contextual",
              "    embeddings advantage is dataset-conditional (implicit > explicit content).",
              "  - If gap holds on both datasets at large scale → joint effect of architecture",
              "    and feature representation.",
              "  - Competitive ML at small Davidson sizes → TF-IDF quality is key for",
              "    explicit-content, low-resource scenarios."]

    rq_text = "\n".join(lines)
    print(rq_text)
    with open(RESULTS_DIR/"research_question_answers.txt","w",encoding="utf-8") as f:
        f.write(rq_text)

print("\n✅ Step 16 complete — research_question_answers.txt saved")

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  EXPERIMENT COMPLETE")
print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*70}")
print("\nAll outputs saved to:")
for out in [
    "results/master_results.csv",
    "results/metrics/ml_results.csv",
    "results/metrics/bert_results.csv",
    "results/metrics/distilbert_results.csv",
    "results/metrics/mcnemar_results.csv",
    "results/metrics/efficiency_summary.csv",
    "results/confusion_matrices/all_confusion_matrices.png",
    "results/learning_curves/learning_curves_combined.png",
    "results/robustness/cross_dataset_gap.png",
    "results/research_question_answers.txt",
]:
    p = PROJECT_ROOT / out
    exists = "✓" if p.exists() else "✗ missing"
    print(f"  {exists}  {out}")
