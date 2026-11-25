"""Fricções: matriz de custos, ajuste de limiar e simulação de carteira com custos de transação."""
import numpy as np
import pandas as pd

from src.config import parametros
from src.storage import lake


def custo_por_lado() -> float:
    """Custo de uma compra OU venda, em fração do valor negociado."""
    c = parametros()["custos"]
    return (c["tarifas_bps"] + c["spread_slippage_bps"]) / 1e4


def matriz_custos(ret_treino: np.ndarray) -> dict:
    """Custo médio, em % do capital, de cada erro. Estimado só no treino.

    Falso positivo: comprou e não subiu. Perde a queda média mais a ida e volta de custos.
    Falso negativo: ficou de fora e subiu. Deixa de ganhar a alta média, já descontados os custos.
    """
    ida_volta = 2 * custo_por_lado()
    r = ret_treino[~np.isnan(ret_treino)]
    return {"FP": float(-r[r <= 0].mean() + ida_volta), "FN": float(r[r > 0].mean() - ida_volta), "VP": 0.0, "VN": 0.0}


def custo_medio(y: np.ndarray, p: np.ndarray, limiar: float, matriz: dict) -> float:
    """Custo médio por decisão, segundo a matriz."""
    d = p > limiar
    return float((matriz["FP"] * (d & (y == 0)) + matriz["FN"] * (~d & (y == 1))).mean())


def curva_limiar(y: np.ndarray, p: np.ndarray, matriz: dict) -> pd.DataFrame:
    """Custo médio para cada limiar candidato (quantis das próprias probabilidades e o intervalo 0,40 a 0,65)."""
    grade = np.unique(np.round(np.concatenate([np.arange(0.40, 0.651, 0.005), np.quantile(p, np.linspace(0.02, 0.98, 49))]), 4))
    return pd.DataFrame({"limiar": grade, "custo": [custo_medio(y, p, t, matriz) for t in grade]})


def escolher_limiar(y: np.ndarray, p: np.ndarray, matriz: dict) -> float:
    """Limiar de menor custo na VALIDAÇÃO. Em empate, o mais próximo do limiar teórico FP/(FP+FN)."""
    c = curva_limiar(y, p, matriz)
    teorico = matriz["FP"] / (matriz["FP"] + matriz["FN"])
    melhores = c[c.custo <= c.custo.min() + 1e-12]
    return float(melhores.iloc[(melhores.limiar - teorico).abs().argmin()].limiar)


def simular(base: pd.DataFrame, compra: np.ndarray, retorno: str = "ret_exec", custo: float | None = None) -> pd.DataFrame:
    """Carteira comprada ou fora do mercado, com peso 1/N por ativo.

    Regra: sinal calculado depois do fechamento de t; execução na ABERTURA de t+1; posição mantida até a
    abertura de t+2 e renovada se o sinal continuar. O que não está investido rende CDI.
    O custo incide sobre cada mudança de peso. Não há venda a descoberto, logo não há aluguel.
    `retorno="ret_prox"` simula a execução impossível no próprio fechamento de t, só para comparação.
    """
    custo = custo_por_lado() if custo is None else custo
    d = base[["ticker", "data", "data_alvo", retorno]].copy()
    d["w"] = compra.astype(float)
    d = d.dropna(subset=[retorno])
    n = d.ticker.nunique()
    w = d.pivot(index="data", columns="ticker", values="w").fillna(0) / n
    r = d.pivot(index="data", columns="ticker", values=retorno).reindex_like(w).fillna(0)
    cdi = lake.ler("bronze", "cdi").assign(data=lambda x: pd.to_datetime(x.data)).set_index("data").cdi_dia
    dia_exec = d.groupby("data").data_alvo.first()
    cdi = cdi.reindex(dia_exec).ffill().fillna(0).to_numpy()
    giro = w.diff().abs().sum(axis=1)
    giro.iloc[0] = w.iloc[0].sum()
    liquido = (w * r).sum(axis=1) + (1 - w.sum(axis=1)) * cdi - giro * custo
    return pd.DataFrame({"data_execucao": dia_exec, "ret": liquido, "bruto": liquido + giro * custo, "giro": giro,
                         "exposicao": w.sum(axis=1), "cdi": cdi}, index=w.index)


def resumo(sim: pd.DataFrame) -> dict:
    """Retorno e risco anualizados (252 pregões), com excesso sobre o CDI, queda máxima e giro."""
    r, n = sim.ret, len(sim)
    acum = (1 + r).cumprod()
    exc = r - sim.cdi
    return {"retorno_total_%": 100 * (acum.iloc[-1] - 1), "retorno_aa_%": 100 * (acum.iloc[-1] ** (252 / n) - 1),
            "volatilidade_aa_%": 100 * r.std() * np.sqrt(252), "sharpe_vs_cdi": float(exc.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
            "queda_maxima_%": 100 * float((acum / np.maximum(acum.cummax(), 1) - 1).min()), "giro_diario_%": 100 * sim.giro.mean(),
            "exposicao_media_%": 100 * sim.exposicao.mean()}
