"""Preços diários (yfinance) e CDI (Banco Central). O yfinance não é fonte oficial."""
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.config import parametros
from src.ingestion.rede import baixar, preparar_certificados
from src.storage import controle, lake

COLUNAS = {"Open": "abertura", "High": "maxima", "Low": "minima", "Close": "fechamento",
           "Adj Close": "fechamento_ajustado_yahoo", "Volume": "volume", "Dividends": "dividendo",
           "Stock Splits": "desdobramento"}


def _pregao_encerrado(dia: pd.Timestamp) -> bool:
    """A barra do próprio dia só vale depois das 18h30 de Brasília."""
    agora = datetime.now(ZoneInfo("America/Sao_Paulo"))
    return dia.date() < agora.date() or (agora.hour, agora.minute) >= (18, 30)


def _consultar(simbolo: str, inicio: str) -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker(simbolo).history(start=inicio, auto_adjust=False, actions=True)
    if h.empty:
        raise ValueError(f"yfinance não devolveu dados para {simbolo}")
    h = h[[_pregao_encerrado(d) for d in h.index]].dropna(subset=["Close"])
    df = h.rename(columns=COLUNAS)
    for c in COLUNAS.values():
        if c not in df:
            df[c] = 0.0
    df = df[list(COLUNAS.values())].astype("float64")
    df.insert(0, "data", [str(d.date()) for d in h.index])
    return df.reset_index(drop=True)


def coletar(ticker: str) -> dict:
    """Atualiza bronze/precos de um ativo ou do índice (^BVSP).

    Busca só os últimos 10 dias além do que já existe. Se um fechamento já gravado vier diferente,
    houve desdobramento ou correção na fonte e a série inteira é baixada de novo.
    """
    preparar_certificados()
    simbolo = ticker if ticker.startswith("^") else f"{ticker}.SA"
    inicio_total = parametros()["dados"]["inicio_precos"]
    atual = pd.DataFrame()
    if lake.existe("bronze", "precos"):
        atual = lake.ler("bronze", "precos")
        atual = atual[atual.ticker == ticker]
    inicio = str((pd.Timestamp(atual.data.max()) - timedelta(days=10)).date()) if len(atual) else inicio_total
    try:
        df = _consultar(simbolo, inicio)
        if len(atual):
            m = atual.merge(df, on="data", suffixes=("_a", "_n"))
            if ((m.fechamento_a / m.fechamento_n - 1).abs() > 1e-3).any():
                df = _consultar(simbolo, inicio_total)
        df = df.assign(ticker=ticker, fonte="yfinance", coletado_em=controle.agora())
        r = lake.upsert("bronze", "precos", df, ["ticker", "data"])
        controle.registrar("yfinance", ticker, **r, ultima_data=df.data.max())
        return r
    except Exception as e:
        controle.registrar("yfinance", ticker, erro=f"{type(e).__name__}: {e}"[:300])
        return {"novos": 0, "atualizados": 0, "erro": str(e)}


def coletar_cdi() -> dict:
    """CDI diário (série 12 do SGS/Banco Central), remuneração do caixa na simulação."""
    try:
        ini = pd.Timestamp(parametros()["dados"]["inicio_base"])
        url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json"
               f"&dataInicial={ini:%d/%m/%Y}&dataFinal={date.today():%d/%m/%Y}")
        df = pd.DataFrame(baixar(url).json())
        df["data"] = pd.to_datetime(df["data"], dayfirst=True).dt.strftime("%Y-%m-%d")
        df["cdi_dia"] = df.valor.astype(float) / 100
        r = lake.upsert("bronze", "cdi", df[["data", "cdi_dia"]], ["data"])
        controle.registrar("BCB-SGS", "CDI", **r, ultima_data=df.data.max())
        return r
    except Exception as e:
        controle.registrar("BCB-SGS", "CDI", erro=f"{type(e).__name__}: {e}"[:300])
        return {"novos": 0, "atualizados": 0, "erro": str(e)}
