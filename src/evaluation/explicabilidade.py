"""Importância de variáveis: permutação por grupo (os dois modelos) e SHAP (Random Forest)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import parametros
from src.features.ouro import GRUPOS


def permutacao_grupos(modelo, base: pd.DataFrame, mascara: np.ndarray, repeticoes: int = 5) -> pd.DataFrame:
    """Queda da AUC quando se embaralham, entre linhas, todas as variáveis de um grupo.

    Só as variáveis são embaralhadas, dentro do conjunto avaliado, para medir quanto o modelo depende delas.
    """
    g = np.random.default_rng(parametros()["semente"])
    y = base.alvo.to_numpy()[mascara]
    ref = roc_auc_score(y, modelo.prever(base)[mascara])
    idx = np.flatnonzero(mascara)
    linhas = []
    for nome, cols in GRUPOS.items():
        cols = [c for c in cols if c in modelo.prep.variaveis]
        if not cols:
            continue
        quedas = []
        for _ in range(repeticoes):
            b = base.copy()
            b.loc[b.index[idx], cols] = b[cols].to_numpy()[g.permutation(idx)]
            quedas.append(ref - roc_auc_score(y, modelo.prever(b)[mascara]))
        linhas.append({"grupo": nome, "queda_auc": np.mean(quedas), "desvio": np.std(quedas)})
    return pd.DataFrame(linhas).sort_values("queda_auc", ascending=False)


def shap_floresta(rf, base: pd.DataFrame, mascara: np.ndarray, amostra: int = 800) -> tuple[np.ndarray, pd.DataFrame]:
    """Valores SHAP da classe "alta" em uma amostra do conjunto avaliado."""
    import shap
    sub = base[mascara]
    sub = sub.sample(min(amostra, len(sub)), random_state=parametros()["semente"])
    x = pd.DataFrame(rf.prep.transformar(sub), columns=rf.prep.nomes)
    v = shap.TreeExplainer(rf.modelo).shap_values(x)
    v = v[1] if isinstance(v, list) else (v[:, :, 1] if v.ndim == 3 else v)
    return v, x
