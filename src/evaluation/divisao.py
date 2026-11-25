"""Divisão cronológica: treino, validação, teste e folds do TimeSeriesSplit."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import parametros


def mascaras(base: pd.DataFrame) -> dict[str, np.ndarray]:
    """Máscaras booleanas de treino, validação e teste, por data do sinal e só com alvo conhecido."""
    d = parametros()["divisao"]
    return {k: (base.data.between(*map(pd.Timestamp, d[k])) & base.alvo.notna()).to_numpy() for k in ("treino", "validacao", "teste")}


def folds(base: pd.DataFrame, treino: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """TimeSeriesSplit sobre DATAS ÚNICAS do treino.

    Como a base empilha vários ativos, dividir por linha colocaria o mesmo pregão dos dois lados.
    `gap` descarta pregões entre as duas partes (embargo).
    """
    p = parametros()["divisao"]
    datas = np.sort(base.loc[treino, "data"].unique())
    saida = []
    for a, b in TimeSeriesSplit(n_splits=p["n_folds"], gap=p["embargo_pregoes"]).split(datas):
        saida.append((treino & base.data.isin(datas[a]).to_numpy(), treino & base.data.isin(datas[b]).to_numpy()))
    return saida


def quadro(base: pd.DataFrame) -> pd.DataFrame:
    """Datas efetivas, número de pregões, observações e proporção de altas em cada bloco."""
    m = mascaras(base)
    linhas = []
    def linha(nome, mask):
        x = base[mask]
        linhas.append({"bloco": nome, "inicio": x.data.min().date(), "fim": x.data.max().date(), "pregoes": x.data.nunique(),
                       "observacoes": len(x), "altas_%": round(100 * x.alvo.mean(), 1)})
    for nome in ("treino", "validacao", "teste"):
        linha(nome, m[nome])
    for i, (a, b) in enumerate(folds(base, m["treino"]), 1):
        linha(f"fold {i} - ajuste", a)
        linha(f"fold {i} - validação", b)
    return pd.DataFrame(linhas)
