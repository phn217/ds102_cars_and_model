import numpy as np

def confusion_matrix(y_true, y_pred, n_classes=None):
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm

def precision_recall_f1(y_true, y_pred, n_classes=None):
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)  
    recall = tp / np.maximum(cm.sum(axis=1), 1)     
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1

def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())

def evaluate(y_true, y_pred, n_classes=None, class_names=None):
    p, r, f = precision_recall_f1(y_true, y_pred, n_classes)
    res = {"accuracy": accuracy(y_true, y_pred)}
    if len(p) == 2:
        res.update(precision=p[1], recall=r[1], f1=f[1])
    else:
        res.update(precision=p.mean(), recall=r.mean(), f1=f.mean(), average="macro")
        names = class_names or [str(i) for i in range(len(p))]
        res["per_class"] = {n: dict(precision=p[i], recall=r[i], f1=f[i])
                            for i, n in enumerate(names)}
    return res

def print_report(res: dict):
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"{k:>10}: {res[k]:.4f}")
    if "per_class" in res:
        print("(macro-average; chi tiết từng lớp)")
        for n, m in res["per_class"].items():
            print(f"  {n:<12} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
