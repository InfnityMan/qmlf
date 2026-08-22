import warnings; warnings.simplefilter("ignore")
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qmlf import QuantumKernel

X, y = make_classification(n_samples=80, n_features=6, n_informative=4, n_redundant=0,
                           class_sep=1.2, random_state=7)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

def run(mode, bw, norm):
    k = QuantumKernel(n_qubits=6, mode=mode, bandwidth=bw, normalize=norm).fit(Xtr)
    Ktr = k.compute_kernel_matrix(Xtr); Kte = k.compute_kernel_matrix(Xte, Xtr)
    off = Ktr[~np.eye(len(Ktr),dtype=bool)].mean()
    s = SVC(kernel="precomputed").fit(Ktr, ytr)
    return off, accuracy_score(yte, s.predict(Kte))

print(f"{'mode':<14}{'bandwidth':<11}{'normalize':<11}{'offdiag_mean':<14}{'test_acc'}")
for mode,norm in (("ZZ",None),("mahalanobis","maxabs")):
    for bw in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
        off, acc = run(mode,bw,norm)
        print(f"{mode:<14}{bw:<11}{str(norm):<11}{off:<14.5f}{acc:.4f}")
