# train_safe.py  —  use group-aware split if possible, otherwise fall back to stratified split
import os, json, numpy as np, pandas as pd, random
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ===== CONFIG =====
FEAT_CSV  = "./features/featuresv1.csv"   # cols: path,label,source,feat
HPP_OUT   = "./firmware/model_params.hpp"
TEST_PROP = 0.30
SEED      = 42
# ==================

random.seed(SEED); np.random.seed(SEED)

def load_features(csv_path):
    df = pd.read_csv(csv_path)
    X = np.stack(df["feat"].map(lambda s: np.array(json.loads(s), dtype=np.float32)))
    labels = sorted(df["label"].unique().tolist())
    lab2id = {l:i for i,l in enumerate(labels)}
    y = df["label"].map(lab2id).values.astype(int)
    src = df["source"].astype(str).values
    return df, X, y, labels, src

def zfit(X): 
    mu = X.mean(axis=0); sigma = X.std(axis=0) + 1e-8
    return mu, sigma

def zapply(X, mu, sigma): 
    return (X - mu) / sigma

def can_do_group_split(y, sources, labels):
    """Check if each class has at least 2 unique sources."""
    ok = True
    per_cls = {}
    for i, lab in enumerate(labels):
        nsrc = len(set(sources[y == i]))
        per_cls[lab] = nsrc
        if nsrc < 2:
            ok = False
    return ok, per_cls

def group_split_indices(y, sources, labels, test_prop=0.30, seed=42):
    """Simple group-aware split: distribute sources per class."""
    rng = random.Random(seed)
    idx_by_lab_src = defaultdict(lambda: defaultdict(list))
    for i, (yi, s) in enumerate(zip(y, sources)):
        idx_by_lab_src[yi][s].append(i)

    tr_idx, te_idx = [], []
    for cls in range(len(labels)):
        src_map = idx_by_lab_src[cls]
        src_list = list(src_map.keys())
        n_src = len(src_list)
        k = max(1, int(round(n_src * test_prop)))
        if k >= n_src: k = n_src - 1
        rng.shuffle(src_list)
        test_srcs = set(src_list[:k])
        for s, idxs in src_map.items():
            (te_idx if s in test_srcs else tr_idx).extend(idxs)

    te_idx = np.array(sorted(set(te_idx)), dtype=int)
    tr_idx = np.array(sorted(set(tr_idx) - set(te_idx)), dtype=int)
    return tr_idx, te_idx

def export_hpp(W, labels, mu, sigma, out_path):
    C, Fp1 = W.shape; F = Fp1-1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("// Auto-generated model params\n#pragma once\nnamespace model {\n")
        f.write(f"static const int CLASSES = {C};\n")
        f.write(f"static const int FEATS   = {F};\n")
        f.write("static const float MU[FEATS] = {")
        f.write(",".join(f"{float(m):.6f}f" for m in mu)); f.write("};\n")
        f.write("static const float SIGMA[FEATS] = {")
        f.write(",".join(f"{float(s):.6f}f" for s in sigma)); f.write("};\n")
        f.write("static const float W[CLASSES][FEATS+1] = {\n")
        for c in range(C):
            row = ",".join([f"{float(W[c,0]):.6f}f"] + [f"{float(W[c,1+i]):.6f}f" for i in range(F)])
            f.write(f"  {{ {row} }},\n")
        f.write("};\n")
        f.write("static const char* LABELS[CLASSES] = {")
        f.write(",".join([f"\"{l}\"" for l in labels])); f.write("};\n}\n")
    print(f"[OK] wrote {out_path}")

def main():
    df, X, y, labels, sources = load_features(FEAT_CSV)
    all_lbl_idx = np.arange(len(labels))

    ok_group, per_cls_sources = can_do_group_split(y, sources, labels)
    print("Class -> #sources:", per_cls_sources)

    if ok_group:
        print("[INFO] Using GROUP-AWARE split (source-based).")
        tr_idx, te_idx = group_split_indices(y, sources, labels, test_prop=TEST_PROP, seed=SEED)
    else:
        print("[WARN] Not enough distinct sources per class; falling back to STRATIFIED split.")
        idx_all = np.arange(len(y))
        tr_idx, te_idx = train_test_split(
            idx_all, test_size=TEST_PROP, random_state=SEED, stratify=y
        )

    def stats(idx):
        return dict(Counter(df.loc[idx, "label"]))
    print("Train counts:", stats(tr_idx))
    print("Test  counts:", stats(te_idx))

    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]

    if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
        raise RuntimeError("Split failed: empty train/test. Check 'source' values or TEST_PROP.")

    mu, sigma = zfit(Xtr)
    Xtr = zapply(Xtr, mu, sigma)
    Xte = zapply(Xte, mu, sigma)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)

    ypr = clf.predict(Xte)
    print(classification_report(yte, ypr, labels=all_lbl_idx, target_names=labels, zero_division=0))
    print(confusion_matrix(yte, ypr, labels=all_lbl_idx))

    C = len(labels); F = X.shape[1]
    W = np.zeros((C, F+1), dtype=np.float32)
    W[:,0]  = clf.intercept_.astype(np.float32)
    W[:,1:] = clf.coef_.astype(np.float32)
    export_hpp(W, labels, mu, sigma, HPP_OUT)

if __name__ == "__main__":
    main()
