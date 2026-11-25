"""Prata de preços: calendário, limpeza e ajuste por proventos."""
import numpy as np
import pandas as pd

from src.config import parametros
from src.storage import lake


def calendario(bruto: pd.DataFrame) -> pd.DatetimeIndex:
    """Pregões válidos: dias em que o Ibovespa tem cotação e ao menos metade dos ativos negociou.

    Elimina os feriados que o Yahoo devolve como se fossem pregão.
    """
    ibov = set(bruto.loc[bruto.ticker == "^BVSP", "data"])
    acoes = bruto[bruto.ticker != "^BVSP"].copy()
    acoes["negociou"] = (acoes.volume > 0) | (acoes.maxima > acoes.minima)
    maioria = acoes.groupby("data").negociou.mean() >= 0.5
    validos = [d for d in maioria.index if maioria[d] and d in ibov]
    return pd.DatetimeIndex(sorted(pd.to_datetime(validos)))


def ajustar(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta retornos com proventos, sem usar o Adj Close da fonte.

    O Close do Yahoo já vem corrigido por desdobramentos, mas não por dividendos. O retorno total é
    (fechamento + dividendo do dia) / fechamento anterior - 1. O índice de retorno total acumula esses
    retornos a partir do primeiro dia, portanto não muda quando um novo provento é pago no futuro.
    """
    df = df.sort_values("data").copy()
    df["ret"] = (df.fechamento + df.dividendo) / df.fechamento.shift(1) - 1
    df["evento_societario"] = False
    for ev in parametros()["eventos_societarios"]:
        dia = (df.ticker == ev["ticker"]) & (df.data == pd.Timestamp(ev["data"]))
        df.loc[dia, "evento_societario"], df.loc[dia, "ret"] = True, 0.0
    df["ret_log"] = np.log1p(df.ret)
    df["indice_rt"] = (1 + df.ret.fillna(0)).cumprod() * 100
    df["ret_abertura"] = (df.abertura + df.dividendo) / df.abertura.shift(1) - 1
    df["ret_intradia"] = df.fechamento / df.abertura - 1
    return df


def construir() -> pd.DataFrame:
    """Bronze -> silver/precos, com uma linha por ativo e pregão válido."""
    bruto = lake.ler("bronze", "precos")
    pregoes = calendario(bruto)
    bruto["data"] = pd.to_datetime(bruto.data)
    partes = []
    for ticker, g in bruto[bruto.data.isin(pregoes)].groupby("ticker"):
        g = g.drop_duplicates("data").sort_values("data")
        g["volume"] = g.volume.where(g.volume > 0)
        g["sem_volume"] = g.volume.isna()
        coerente = (g.maxima + 1e-6 >= g[["abertura", "fechamento"]].max(axis=1)) & (g.minima - 1e-6 <= g[["abertura", "fechamento"]].min(axis=1))
        g["barra_coerente"] = coerente
        partes.append(ajustar(g))
    df = pd.concat(partes, ignore_index=True)
    df = df.drop(columns=["fechamento_ajustado_yahoo", "fonte", "coletado_em"])
    lake.sobrescrever("silver", "precos", df)
    return df
