"""Métricas de classificação e incerteza."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

from src.config import parametros


def classificar(y: np.ndarray, p: np.ndarray, limiar: float = 0.5) -> dict:
    """Métricas para probabilidades `p` e decisão `p > limiar`."""
    d = (p > limiar).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, d, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y, d), "precision": precision_score(y, d, zero_division=0), "recall": recall_score(y, d, zero_division=0),
            "f1": f1_score(y, d, zero_division=0), "mcc": matthews_corrcoef(y, d),
            "auc": roc_auc_score(y, p) if np.ptp(p) > 0 else 0.5, "brier": brier_score_loss(y, p),
            "previu_alta_%": 100 * d.mean(), "VP": int(vp), "FP": int(fp), "FN": int(fn), "VN": int(vn)}


def _reamostras(datas: np.ndarray, n: int, bloco: int = 5):
    """Bootstrap em blocos de pregões inteiros: preserva a dependência entre ativos no mesmo dia e entre dias vizinhos."""
    g = np.random.default_rng(parametros()["semente"])
    unicas, inv = np.unique(datas, return_inverse=True)
    linhas = [np.flatnonzero(inv == k) for k in range(len(unicas))]
    for _ in range(n):
        ini = g.integers(0, len(unicas) - bloco + 1, size=int(np.ceil(len(unicas) / bloco)))
        dias = (ini[:, None] + np.arange(bloco)).ravel()[:len(unicas)]
        yield np.concatenate([linhas[k] for k in dias])


def intervalos(datas, y, p, limiar: float, n: int = 500) -> dict:
    """Intervalos de 95% para acurácia e AUC."""
    ac, au = [], []
    d = (p > limiar).astype(int)
    for i in _reamostras(datas, n):
        ac.append((d[i] == y[i]).mean())
        au.append(roc_auc_score(y[i], p[i]) if np.ptp(p) > 0 else 0.5)
    q = lambda v: tuple(np.round(np.percentile(v, [2.5, 97.5]), 4))
    return {"accuracy_ic": q(ac), "auc_ic": q(au)}


def diferenca(datas, y, p_a, p_b, l_a: float, l_b: float, n: int = 500) -> dict:
    """Diferença de acurácia e de AUC entre dois modelos, com intervalo de 95% no mesmo bootstrap."""
    ac, au = [], []
    a, b = (p_a > l_a) == y, (p_b > l_b) == y
    for i in _reamostras(datas, n):
        ac.append(a[i].mean() - b[i].mean())
        au.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    q = lambda v: tuple(np.round(np.percentile(v, [2.5, 97.5]), 4))
    return {"dif_accuracy": float(a.mean() - b.mean()), "dif_accuracy_ic": q(ac),
            "dif_auc": float(roc_auc_score(y, p_a) - roc_auc_score(y, p_b)), "dif_auc_ic": q(au)}


def tabela(resultados: dict[str, dict]) -> pd.DataFrame:
    """Dicionário {modelo: métricas} em tabela."""
    return pd.DataFrame(resultados).T
