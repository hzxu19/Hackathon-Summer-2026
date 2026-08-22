# bagged MLP member: average of 5 seeds per fold on the standardized F_fold{f}.f64 matrices
import os, time, numpy as np

WORK = os.environ.get("PIPE_WORK", "work")
NTHREAD = os.environ.get("PIPE_NTHREAD", "5")
os.environ.setdefault("OMP_NUM_THREADS", NTHREAD)
from sklearn.neural_network import MLPClassifier

DAT = os.path.join(WORK, "features")
OUT = os.path.join(WORK, "scratch", "sk_out"); os.makedirs(OUT, exist_ok=True)
K = 60
P = len(open(f"{DAT}/F_colnames.txt").read().strip().split("\n"))
N = os.path.getsize(f"{DAT}/F_fold0.f64") // 8 // P
yf = np.loadtxt(f"{DAT}/y_fold.csv", delimiter=",", skiprows=1, usecols=(0, 1), dtype=int)
y, fold = yf[:, 0], yf[:, 1]
NTR = len(y)
NSEED = 5

def loadF(f):
    return np.fromfile(f"{DAT}/F_fold{f}.f64", dtype=np.float64).reshape(N, P)

def std_fit(X):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-8] = 1.0
    return mu, sd

def proba60(m, X):
    p = m.predict_proba(X); out = np.zeros((X.shape[0], K)); out[:, m.classes_] = p
    return out

def bag(Xtr, ytr, Xq):
    pr = np.zeros((Xq.shape[0], K))
    for s in range(NSEED):
        m = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, early_stopping=True, max_iter=300,
                          n_iter_no_change=15, random_state=1000 + s, batch_size=128, learning_rate_init=1e-3)
        m.fit(Xtr, ytr); pr += proba60(m, Xq)
    return pr / NSEED

t0 = time.time(); oof = np.zeros((NTR, K))
for f in range(1, 6):
    F = loadF(f); va = np.where(fold == f)[0]; tr = np.where(fold != f)[0]
    mu, sd = std_fit(F[tr]); Xtr = (F[tr] - mu) / sd; Xva = (F[va] - mu) / sd
    oof[va] = bag(Xtr, y[tr], Xva)
    print(f"[mlpbag] fold {f} acc={(oof[va].argmax(1) == y[va]).mean():.4f} ({(time.time()-t0)/60:.1f} min)", flush=True)
np.savetxt(f"{OUT}/mlpbag_oof.csv", oof, delimiter=",", fmt="%.6g")
F = loadF(0); mu, sd = std_fit(F[:NTR]); Xtr = (F[:NTR] - mu) / sd; Xte = (F[NTR:] - mu) / sd
np.savetxt(f"{OUT}/mlpbag_test.csv", bag(Xtr, y, Xte), delimiter=",", fmt="%.6g")
print(f"[mlpbag] done ({(time.time()-t0)/60:.1f} min)", flush=True)
