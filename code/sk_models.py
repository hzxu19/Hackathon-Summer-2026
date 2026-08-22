# scikit-learn ensemble members on the fold-wise feature matrices F_fold{f}.f64
# (fold f: neighbour-label block from the fold's training pool; fold 0: full pool, test model only)
# usage: python sk_models.py mlp rf et hgb svc lr
import os, sys, time, numpy as np

WORK = os.environ.get("PIPE_WORK", "work")
NTHREAD = os.environ.get("PIPE_NTHREAD", "5")
os.environ.setdefault("OMP_NUM_THREADS", NTHREAD)
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier

DAT = os.path.join(WORK, "features")
OUT = os.path.join(WORK, "scratch", "sk_out"); os.makedirs(OUT, exist_ok=True)
NJOBS = int(NTHREAD)
K = 60
P = len(open(f"{DAT}/F_colnames.txt").read().strip().split("\n"))
N = os.path.getsize(f"{DAT}/F_fold0.f64") // 8 // P
yf = np.loadtxt(f"{DAT}/y_fold.csv", delimiter=",", skiprows=1, usecols=(0, 1), dtype=int)
y, fold = yf[:, 0], yf[:, 1]
NTR = len(y)
which = sys.argv[1:] or ["mlp", "rf", "et", "hgb", "svc", "lr"]

def loadF(f):
    return np.fromfile(f"{DAT}/F_fold{f}.f64", dtype=np.float64).reshape(N, P)

def std_fit(X):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
    return mu, sd

def make(name):
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, early_stopping=True, max_iter=300,
                             n_iter_no_change=15, random_state=0)
    if name == "lr":
        return LogisticRegression(C=0.5, max_iter=1000)
    if name == "svc":
        return SVC(C=5.0, kernel="rbf", gamma="scale", probability=True, random_state=0)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, early_stopping=True,
                                              n_iter_no_change=20, validation_fraction=0.1, random_state=0,
                                              max_leaf_nodes=31, l2_regularization=1.0)
    if name == "rf":
        return RandomForestClassifier(n_estimators=600, max_features="sqrt", min_samples_leaf=1,
                                      n_jobs=NJOBS, random_state=0)
    if name == "et":
        return ExtraTreesClassifier(n_estimators=800, max_features=0.15, min_samples_leaf=1,
                                    n_jobs=NJOBS, random_state=0)
    raise ValueError(name)

def proba60(m, X):
    p = m.predict_proba(X); out = np.zeros((X.shape[0], K)); out[:, m.classes_] = p
    return out

for name in which:
    t0 = time.time(); oof = np.zeros((NTR, K)); scale = name in ("mlp", "lr", "svc")
    for f in range(1, 6):
        F = loadF(f); va = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
        Xtr, Xva = F[tr], F[va]
        if scale:
            mu, sd = std_fit(Xtr); Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
        m = make(name); m.fit(Xtr, y[tr]); oof[va] = proba60(m, Xva)
        print(f"[{name}] fold {f} acc={(oof[va].argmax(1) == y[va]).mean():.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
    np.savetxt(f"{OUT}/{name}_oof.csv", oof, delimiter=",", fmt="%.6g")
    F = loadF(0); Xtr, Xte = F[:NTR], F[NTR:]
    if scale:
        mu, sd = std_fit(Xtr); Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    m = make(name); m.fit(Xtr, y)
    np.savetxt(f"{OUT}/{name}_test.csv", proba60(m, Xte), delimiter=",", fmt="%.6g")
    print(f"[{name}] done ({(time.time()-t0)/60:.1f} min)", flush=True)
