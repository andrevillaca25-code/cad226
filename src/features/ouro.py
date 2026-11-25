"""Ouro: base integrada por ativo e pregão.

Regra única de tempo: a linha da data t só contém informação conhecida até o fechamento de t.
O alvo da linha t é a direção do retorno do pregão seguinte.
"""
import numpy as np
import pandas as pd

from src.config import parametros
from src.nlp import sentimento
from src.storage import lake

PRECO = ["r1", "r5", "r21", "vol21", "vol5_21", "volume_rel", "amplitude", "intradia", "rsi14", "dist_mm20", "dist_mm50", "dist_mm200"]
MERCADO = ["ibov_r1", "ibov_r5", "ibov_vol21", "rel1", "rel5"]
TEXTO = ["n_docs", "n_docs5", "n_docs21", "sent_dia", "sent5", "sent21", "tem_doc5", "pregoes_sem_doc"]
FUNDAMENTOS = ["roe_12m", "margem_liquida_12m", "alavancagem", "cresc_receita", "cresc_lucro", "dias_desde_balanco"]
NUMERICAS = PRECO + MERCADO + TEXTO + FUNDAMENTOS
GRUPOS = {"Preço e volume": PRECO, "Mercado": MERCADO, "Texto": TEXTO, "Fundamentos": FUNDAMENTOS}


def rsi(ret: pd.Series, n: int = 14) -> pd.Series:
    """Índice de força relativa, pela média simples das altas e baixas em n pregões."""
    alta = ret.clip(lower=0).rolling(n).mean()
    baixa = (-ret.clip(upper=0)).rolling(n).mean()
    return 100 * alta / (alta + baixa).replace(0, np.nan)


def variaveis_preco(g: pd.DataFrame) -> pd.DataFrame:
    """Indicadores técnicos de um ativo; toda janela termina em t."""
    g = g.sort_values("data").copy()
    lr, nivel = g.ret_log, g.indice_rt
    g["r1"], g["r5"], g["r21"] = lr, lr.rolling(5).sum(), lr.rolling(21).sum()
    g["vol21"] = lr.rolling(21).std() * np.sqrt(252)
    g["vol5_21"] = lr.rolling(5).std() / lr.rolling(21).std()
    g["volume_rel"] = np.log(g.volume / g.volume.rolling(21, min_periods=10).mean())
    g["amplitude"] = (g.maxima - g.minima) / g.fechamento
    g["intradia"] = g.ret_intradia
    g["rsi14"] = rsi(lr)
    for n in (20, 50, 200):
        g[f"dist_mm{n}"] = nivel / nivel.rolling(n).mean() - 1
    return g


def pregao_seguinte(datas: pd.Series, pregoes: pd.DatetimeIndex) -> pd.Series:
    """Primeiro pregão estritamente posterior a cada data (NaT se ainda não aconteceu)."""
    pos = pregoes.searchsorted(pd.DatetimeIndex(datas), side="right")
    alvo = pd.Series(pd.NaT, index=datas.index, dtype="datetime64[ns]")
    dentro = pos < len(pregoes)
    alvo[dentro] = pregoes[pos[dentro]]
    return alvo


def variaveis_texto(base: pd.DataFrame, pregoes: pd.DatetimeIndex, gravar: bool = True) -> pd.DataFrame:
    """Sentimento e contagem de documentos. Documento entregue em D conta a partir do pregão seguinte.

    Sem documento, o escore fica em zero e os indicadores (tem_doc5, pregoes_sem_doc) avisam o modelo
    de que se trata de ausência, não de um texto neutro.
    """
    t = lake.ler("silver", "textos").merge(sentimento.atualizar()[["hash_conteudo", "escore", "classe", "p_pos", "p_neg", "p_neu"]], on="hash_conteudo")
    t["data"] = pregao_seguinte(t.entregue_em, pregoes)
    if gravar:
        lake.sobrescrever("gold", "textos_classificados", t)
    dia = t.dropna(subset=["data"]).groupby(["ticker", "data"]).agg(n_docs=("escore", "size"), soma=("escore", "sum")).reset_index()
    dia["data"] = dia.data.astype(base.data.dtype)
    b = base[["ticker", "data"]].merge(dia, on=["ticker", "data"], how="left").fillna({"n_docs": 0, "soma": 0}).sort_values(["ticker", "data"])
    partes = []
    for _, g in b.groupby("ticker"):
        g = g.copy()
        g["sent_dia"] = (g.soma / g.n_docs.replace(0, np.nan)).fillna(0)
        for n in (5, 21):
            g[f"n_docs{n}"] = g.n_docs.rolling(n, min_periods=1).sum()
            g[f"sent{n}"] = (g.soma.rolling(n, min_periods=1).sum() / g[f"n_docs{n}"].replace(0, np.nan)).fillna(0)
        g["tem_doc5"] = (g.n_docs5 > 0).astype(float)
        ordem = pd.Series(np.arange(len(g)), index=g.index)
        g["pregoes_sem_doc"] = (ordem - ordem.where(g.n_docs > 0).ffill()).fillna(60).clip(upper=60)
        partes.append(g.drop(columns="soma"))
    return pd.concat(partes)


def variaveis_fundamentos(base: pd.DataFrame) -> pd.DataFrame:
    """Último balanço público antes de t (junção "as of", com desigualdade estrita).

    Uma reapresentação tardia de período antigo não substitui um balanço mais recente já conhecido.
    """
    f = lake.ler("silver", "fundamentos").sort_values(["ticker", "disponivel_em", "periodo"])
    f = f[f.periodo >= f.groupby("ticker").periodo.cummax()]
    cols = ["roe_12m", "margem_liquida_12m", "alavancagem", "cresc_receita", "cresc_lucro"]
    esq = base[["ticker", "data"]].sort_values("data").assign(chave=lambda d: d.data.astype("datetime64[ns]"))
    dir_ = f[["ticker", "disponivel_em", "periodo"] + cols].sort_values("disponivel_em").assign(chave=lambda d: d.disponivel_em.astype("datetime64[ns]"))
    m = pd.merge_asof(esq, dir_, on="chave", by="ticker", allow_exact_matches=False)
    m["dias_desde_balanco"] = (m.chave - m.disponivel_em).dt.days.clip(upper=200)
    m = m.rename(columns={"periodo": "balanco_periodo", "disponivel_em": "balanco_disponivel_em"})
    return m.drop(columns="chave")


def construir(precos: pd.DataFrame | None = None, gravar: bool = True) -> pd.DataFrame:
    """Prata -> gold/base: variáveis, alvo e retornos usados na simulação."""
    p = lake.ler("silver", "precos") if precos is None else precos
    p["data"] = pd.to_datetime(p.data)
    pregoes = pd.DatetimeIndex(sorted(p.loc[p.ticker == "^BVSP", "data"].unique()))
    ib = variaveis_preco(p[p.ticker == "^BVSP"].assign(volume=1.0))[["data", "r1", "r5", "vol21"]]
    ib.columns = ["data", "ibov_r1", "ibov_r5", "ibov_vol21"]
    acoes = p[p.ticker != "^BVSP"]
    base = pd.concat([variaveis_preco(g) for _, g in acoes.groupby("ticker")]).merge(ib, on="data", how="left")
    base["rel1"], base["rel5"] = base.r1 - base.ibov_r1, base.r5 - base.ibov_r5
    base = base.merge(variaveis_texto(base, pregoes, gravar), on=["ticker", "data"], how="left")
    base = base.merge(variaveis_fundamentos(base), on=["ticker", "data"], how="left")
    base = base.sort_values(["ticker", "data"])
    g = base.groupby("ticker")
    base["data_alvo"] = g.data.shift(-1)
    base["ret_prox"] = g.ret.shift(-1)
    base["ret_exec"] = g.ret_abertura.shift(-2)
    base["ret_intradia_prox"] = g.ret_intradia.shift(-1)
    evento_prox = g.evento_societario.shift(-1).fillna(False).astype(bool)
    base["alvo"] = np.where(base.ret_prox.isna() | evento_prox, np.nan, (base.ret_prox > 0).astype(float))
    base = base[base.dist_mm200.notna()].reset_index(drop=True)
    colunas = ["ticker", "data", "data_alvo", "alvo", "ret_prox", "ret_exec", "ret_intradia_prox", "fechamento", "volume",
               "balanco_periodo", "balanco_disponivel_em"] + NUMERICAS
    base = base[colunas]
    base[NUMERICAS] = base[NUMERICAS].replace([np.inf, -np.inf], np.nan)
    base["volume_rel"] = base.volume_rel.fillna(0.0)
    if gravar:
        lake.sobrescrever("gold", "base", base)
    return base


def base_modelagem() -> pd.DataFrame:
    """Base congelada usada na avaliação (linhas até a data de congelamento)."""
    b = lake.ler("gold", "base")
    b["data"] = pd.to_datetime(b.data)
    return b[b.data <= parametros()["dados"]["congelamento"]].sort_values(["ticker", "data"]).reset_index(drop=True)
