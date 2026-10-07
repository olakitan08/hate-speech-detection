"""
Transformer fine-tuning: BERT and DistilBERT for hate speech detection.
Runs all 12 conditions (2 models × 2 datasets × 3 sizes).
MPS-safe manual training loop. Saves results after every condition.
"""

# ── MPS memory fix — must be BEFORE any torch import ─────────────────────────
import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONHASHSEED"] = "42"

import sys, random, time, json, pathlib, warnings, string, re
from collections import Counter
from copy import deepcopy
warnings.filterwarnings("ignore")

# Force line-buffered output so logs appear in real time
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed as transformers_set_seed,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
PROJECT = pathlib.Path(__file__).parent.resolve()
DATA_DIR     = PROJECT / "data"
MODELS_DIR   = PROJECT / "models"
RESULTS_DIR  = PROJECT / "results" / "metrics"
CKPT_DIR     = PROJECT / "models" / "checkpoints"

for d in [RESULTS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    transformers_set_seed(seed)

set_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 68)
print("  TRANSFORMER FINE-TUNING — BERT + DistilBERT")
print("=" * 68)
print(f"  Device  : {DEVICE}")
print(f"  PyTorch : {torch.__version__}")
print(f"  Start   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 68)

SUBSET_NAMES  = ["small", "medium", "large"]
SUBSET_EPOCHS = {"small": 3, "medium": 3, "large": 4}

# ─────────────────────────────────────────────────────────────────────────────
# RECREATE IDENTICAL SPLITS (same seed / proportions as the notebook)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Recreating splits from saved CSVs ───────────────────────────────────")

davidson_clean    = pd.read_csv(DATA_DIR / "davidson_clean.csv")
hatexplain_clean  = pd.read_csv(DATA_DIR / "hatexplain_clean.csv")

# Minimal transformer preprocessing (URL + mention removal only)
def preprocess_transformer(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|https\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()

DATASETS = {"davidson": davidson_clean, "hatexplain": hatexplain_clean}
SPLITS   = {}
TEST_SETS = {}

for ds_name, df_full in DATASETS.items():
    N      = len(df_full)
    test_n = int(round(N * 0.20))
    pool, test_df = train_test_split(
        df_full, test_size=test_n / N,
        stratify=df_full["label"], random_state=SEED
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
                pool, test_size=tn / len(pool),
                stratify=pool["label"], random_state=SEED
            )
            tp = tp.reset_index(drop=True)
        val_n    = max(2, int(round(len(tp) * 0.10)))
        tr_train, tr_val = train_test_split(
            tp, test_size=val_n / len(tp),
            stratify=tp["label"], random_state=SEED
        )
        SPLITS[ds_name][subset] = {
            "tr_train": tr_train.reset_index(drop=True),
            "tr_val":   tr_val.reset_index(drop=True),
            "test":     test_df,
        }
        print(f"  {ds_name:10} | {subset:6} | "
              f"train={len(tr_train):5}  val={len(tr_val):4}  test={len(test_df):5}  "
              f"hate={100*tp['label'].mean():.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# METRICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(y_true, y_pred, y_prob, metric_fn, n_iter=1000, seed=SEED):
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_iter):
        idx = rng.randint(0, n, n)
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx], y_prob[idx]))
        except Exception:
            pass
    return (np.percentile(scores, 2.5), np.percentile(scores, 97.5)) if scores else (0.0, 0.0)

f1_macro_fn = lambda yt, yp, _:    f1_score(yt, yp, average="macro", zero_division=0)
auc_fn      = lambda yt, _, yprob: roc_auc_score(yt, yprob) if len(np.unique(yt)) > 1 else 0.0


def compute_metrics(y_true, y_pred, y_prob, train_time, infer_time, n_params=0):
    acc  = accuracy_score(y_true, y_pred)
    mf1  = f1_score(y_true, y_pred, average="macro",     zero_division=0)
    wf1  = f1_score(y_true, y_pred, average="weighted",  zero_division=0)
    mp   = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    mr   = recall_score(y_true, y_pred,    average="macro",    zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    p_cls = precision_score(y_true, y_pred, average=None, zero_division=0)
    r_cls = recall_score(y_true, y_pred,    average=None, zero_division=0)
    f_cls = f1_score(y_true, y_pred,        average=None, zero_division=0)
    p0, p1   = (p_cls[0], p_cls[1]) if len(p_cls) >= 2 else (p_cls[0], 0.0)
    r0, r1   = (r_cls[0], r_cls[1]) if len(r_cls) >= 2 else (r_cls[0], 0.0)
    f0, f1_  = (f_cls[0], f_cls[1]) if len(f_cls) >= 2 else (f_cls[0], 0.0)
    f1_lo, f1_hi   = bootstrap_ci(y_true, y_pred, y_prob, f1_macro_fn)
    auc_lo, auc_hi = bootstrap_ci(y_true, y_pred, y_prob, auc_fn)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy":         acc,
        "macro_precision":  mp,
        "macro_recall":     mr,
        "macro_f1":         mf1,
        "macro_f1_ci_lo":   f1_lo,
        "macro_f1_ci_hi":   f1_hi,
        "weighted_f1":      wf1,
        "auc_roc":          auc,
        "auc_ci_lo":        auc_lo,
        "auc_ci_hi":        auc_hi,
        "hate_precision":   p1,
        "hate_recall":      r1,
        "hate_f1":          f1_,
        "nonhate_precision": p0,
        "nonhate_recall":   r0,
        "nonhate_f1":       f0,
        "train_time":       train_time,
        "infer_time":       infer_time,
        "n_params":         n_params,
        "confusion_matrix": cm,
    }

# ─────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
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

# ─────────────────────────────────────────────────────────────────────────────
# APPEND RESULT ROW TO CSV
# ─────────────────────────────────────────────────────────────────────────────
CSV_COLS = [
    "Model", "Dataset", "Subset", "Train_Size", "Test_Size",
    "accuracy", "macro_precision", "macro_recall",
    "macro_f1", "macro_f1_ci_lo", "macro_f1_ci_hi",
    "weighted_f1", "auc_roc", "auc_ci_lo", "auc_ci_hi",
    "hate_precision", "hate_recall", "hate_f1",
    "nonhate_precision", "nonhate_recall", "nonhate_f1",
    "train_time", "infer_time", "n_params",
]

def append_result(csv_path, row_dict):
    """Append one result row to a CSV, creating the file with headers if needed."""
    row = {c: row_dict.get(c, "") for c in CSV_COLS}
    df_row = pd.DataFrame([row])
    if csv_path.exists():
        df_row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(csv_path, mode="w", header=True, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def train_one_condition(model_hf, model_key, ds_name, subset,
                        csv_path, batch_size=8, device=DEVICE):
    """
    Fine-tune one transformer model on one condition.
    Skips if checkpoint + predictions already exist.
    Returns the metrics dict.
    """
    ckpt_path  = CKPT_DIR / f"{model_key}_{ds_name}_{subset}"
    pred_path  = RESULTS_DIR / f"{model_key}_{ds_name}_{subset}_predictions.npy"
    best_model_pt = ckpt_path / "best_model.pt"

    # ── Skip if already done ─────────────────────────────────────────────────
    if best_model_pt.exists() and pred_path.exists():
        print(f"\n  [SKIP] {model_key.upper()} | {ds_name} | {subset} — already complete")
        return None

    set_seed(SEED)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  {model_key.upper()} | {ds_name} | {subset}  [{time.strftime('%H:%M:%S')}]")
    print(f"{'='*65}", flush=True)

    split  = SPLITS[ds_name][subset]
    epochs = SUBSET_EPOCHS[subset]

    tr_texts  = [preprocess_transformer(t) for t in split["tr_train"]["text"].tolist()]
    val_texts = [preprocess_transformer(t) for t in split["tr_val"]["text"].tolist()]
    te_texts  = [preprocess_transformer(t) for t in split["test"]["text"].tolist()]
    y_tr  = split["tr_train"]["label"].values
    y_val = split["tr_val"]["label"].values
    y_te  = split["test"]["label"].values

    # ── Tokenise ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_hf)
    train_ds  = HateSpeechDataset(tr_texts, y_tr,  tokenizer)
    val_ds    = HateSpeechDataset(val_texts, y_val, tokenizer)
    test_ds   = HateSpeechDataset(te_texts,  y_te,  tokenizer)

    # ── DataLoaders — num_workers=0 and pin_memory=False for MPS ─────────────
    def make_loaders(bs):
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True,
                       num_workers=0, pin_memory=False),
            DataLoader(val_ds,   batch_size=bs, shuffle=False,
                       num_workers=0, pin_memory=False),
            DataLoader(test_ds,  batch_size=bs, shuffle=False,
                       num_workers=0, pin_memory=False),
        )

    tr_loader, val_loader, te_loader = make_loaders(batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        model_hf, num_labels=2, ignore_mismatched_sizes=True
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model    = model.to(device)
    print(f"  Parameters: {n_params:,}  |  Batch: {batch_size}  |  Epochs: {epochs}",
          flush=True)

    # ── Optimiser and linear-warmup scheduler ─────────────────────────────────
    optimizer    = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps  = len(tr_loader) * epochs
    warmup_steps = int(0.10 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0
    patience    = 2
    total_train_t = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        t_ep    = time.time()
        oom_this_epoch = False

        for batch_idx, batch in enumerate(tr_loader):
            optimizer.zero_grad()
            try:
                out = model(
                    input_ids      = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                    labels         = batch["labels"].to(device),
                )
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                ep_loss += loss.item()

                # ── flush MPS cache after every batch ────────────────────────
                if device.type == "mps":
                    torch.mps.empty_cache()

            except RuntimeError as e:
                err = str(e).lower()
                if "out of memory" in err or "mps" in err:
                    print(f"\n  OOM at batch {batch_idx} epoch {epoch} "
                          f"(batch_size={batch_size}) — reducing to "
                          f"{batch_size//2} and retrying condition", flush=True)
                    del model
                    if device.type == "mps":   torch.mps.empty_cache()
                    elif device.type == "cuda": torch.cuda.empty_cache()
                    new_bs = max(2, batch_size // 2)
                    return train_one_condition(
                        model_hf, model_key, ds_name, subset,
                        csv_path, batch_size=new_bs, device=device,
                    )
                raise

        ep_t = time.time() - t_ep
        total_train_t += ep_t

        # ── flush MPS cache after every epoch ────────────────────────────────
        if device.type == "mps":
            torch.mps.empty_cache()

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                out   = model(
                    input_ids      = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                )
                preds = out.logits.argmax(dim=-1).cpu().numpy()
                val_preds.extend(preds.tolist())
                if device.type == "mps":
                    torch.mps.empty_cache()

        val_f1   = f1_score(y_val, val_preds, average="macro", zero_division=0)
        avg_loss = ep_loss / max(len(tr_loader), 1)
        print(f"  Epoch {epoch}/{epochs}  "
              f"loss={avg_loss:.4f}  val_F1={val_f1:.4f}  "
              f"time={ep_t:.0f}s  [{time.strftime('%H:%M:%S')}]", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop — no improvement for {patience} epochs",
                      flush=True)
                break

    # ── Load best checkpoint ──────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Test inference ────────────────────────────────────────────────────────
    model.eval()
    te_preds, te_probs = [], []
    t_inf = time.time()
    with torch.no_grad():
        for batch in te_loader:
            out   = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
            )
            probs = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy()
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            te_preds.extend(preds.tolist())
            te_probs.extend(probs.tolist())
            if device.type == "mps":
                torch.mps.empty_cache()
    infer_time = time.time() - t_inf

    te_preds = np.array(te_preds)
    te_probs = np.array(te_probs)

    # ── Compute all metrics ───────────────────────────────────────────────────
    m = compute_metrics(y_te, te_preds, te_probs,
                        total_train_t, infer_time, n_params)

    print(f"\n  ★ TEST → MacroF1={m['macro_f1']:.4f}  "
          f"AUC={m['auc_roc']:.4f}  "
          f"HateF1={m['hate_f1']:.4f}  "
          f"train={total_train_t:.0f}s  "
          f"infer={infer_time:.2f}s", flush=True)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    torch.save(best_state, best_model_pt)

    # ── Save predictions ──────────────────────────────────────────────────────
    np.save(RESULTS_DIR / f"{model_key}_{ds_name}_{subset}_predictions.npy", te_preds)
    np.save(RESULTS_DIR / f"{model_key}_{ds_name}_{subset}_proba.npy",       te_probs)
    np.save(RESULTS_DIR / f"{model_key}_{ds_name}_{subset}_cm.npy",          m["confusion_matrix"])

    # ── Append metrics to CSV immediately ─────────────────────────────────────
    result_row = {
        "Model":       model_key.upper(),
        "Dataset":     ds_name,
        "Subset":      subset,
        "Train_Size":  len(y_tr),
        "Test_Size":   len(y_te),
        **{k: v for k, v in m.items() if k != "confusion_matrix"},
    }
    append_result(csv_path, result_row)
    print(f"  Saved → {csv_path.name}  predictions.npy  proba.npy  best_model.pt",
          flush=True)

    # ── Clean up GPU/MPS memory ───────────────────────────────────────────────
    del model
    if device.type == "mps":   torch.mps.empty_cache()
    elif device.type == "cuda": torch.cuda.empty_cache()

    return result_row

# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL 12 CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────
TRANSFORMER_CONFIGS = [
    ("bert-base-uncased",       "bert",       PROJECT / "results/metrics/bert_results.csv"),
    ("distilbert-base-uncased", "distilbert", PROJECT / "results/metrics/distilbert_results.csv"),
]

all_results = []
t_global_start = time.time()

for model_hf, model_key, csv_path in TRANSFORMER_CONFIGS:
    model_label = model_hf.upper()
    print(f"\n\n{'#'*68}")
    print(f"#  {model_label}")
    print(f"{'#'*68}")

    for ds_name in ["davidson", "hatexplain"]:
        for subset in SUBSET_NAMES:
            result = train_one_condition(
                model_hf   = model_hf,
                model_key  = model_key,
                ds_name    = ds_name,
                subset     = subset,
                csv_path   = csv_path,
                batch_size = 8,
                device     = DEVICE,
            )
            if result is not None:
                all_results.append(result)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
t_total = time.time() - t_global_start
print(f"\n\n{'='*68}")
print(f"  ALL 12 CONDITIONS COMPLETE  —  total time: {t_total/3600:.2f} h")
print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*68}")

# Load and print combined results
all_dfs = []
for _, _, csv_path in TRANSFORMER_CONFIGS:
    if csv_path.exists():
        all_dfs.append(pd.read_csv(csv_path))

if all_dfs:
    combined = pd.concat(all_dfs, ignore_index=True)
    print("\nFINAL RESULTS TABLE:")
    disp = combined[["Model","Dataset","Subset","Train_Size",
                      "macro_f1","auc_roc","hate_f1","train_time"]]
    print(disp.to_string(index=False))
    print(f"\nTotal rows saved: {len(combined)}")

# Verify all files
print("\nFile verification:")
for model_key in ["bert","distilbert"]:
    for ds in ["davidson","hatexplain"]:
        for sub in SUBSET_NAMES:
            p  = RESULTS_DIR / f"{model_key}_{ds}_{sub}_predictions.npy"
            ck = CKPT_DIR / f"{model_key}_{ds}_{sub}" / "best_model.pt"
            ok_p  = "✓" if p.exists()  else "✗ MISSING"
            ok_ck = "✓" if ck.exists() else "✗ MISSING"
            print(f"  {model_key:10} {ds:10} {sub:6} | pred={ok_p}  ckpt={ok_ck}")
