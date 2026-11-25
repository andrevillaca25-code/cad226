"""Função central do projeto: run_pipeline(ticker, period). Usada pelo notebook e pelo painel."""
import logging
from datetime import date

import numpy as np
import pandas as pd

from src.config import ativos, parametros, tickers
from src.features import ouro
from src.storage import controle, lake

log = logging.getLogger("cad226")
PERIODOS = {"6 meses": 126, "1 ano": 252, "2 anos": 504, "5 anos": 1260, "Tudo": None}
FONTES_OURO = ["precos", "cadastro", "cvm_contas", "cvm_indice", "cvm_ipe", "cvm_textos"]


def _hoje_ja_coletou(fonte: str, alvo: str) -> bool:
    u = controle.ultimo(fonte, alvo)
    return bool(u) and pd.Timestamp(u["momento"]).tz_convert("America/Sao_Paulo").date() == date.today()


def coletar(avisar=lambda m: None, forcar: bool = False) -> list[str]:
    """Consulta as fontes. Cada uma falha isoladamente; o que já está gravado nunca é apagado.

    Preços de todos os ativos são atualizados juntos porque o calendário de pregões depende da maioria deles.
    Sem `forcar`, cada fonte é consultada no máximo uma vez por dia.
    """
    from src.ingestion import cvm, documentos, noticias, precos
    avisos, etapas, inicio = [], [], pd.Timestamp.now(tz="UTC")
    for t in tickers() + ["^BVSP"]:
        if forcar or not _hoje_ja_coletou("yfinance", t):
            etapas.append((f"Preços {t}", lambda t=t: precos.coletar(t)))
    if forcar or not _hoje_ja_coletou("BCB-SGS", "CDI"):
        etapas.append(("CDI", precos.coletar_cdi))
    if forcar or not _hoje_ja_coletou("CVM-FCA", "cadastro"):
        etapas.append(("CVM: cadastro, balanços e documentos", cvm.atualizar))
        etapas.append(("CVM: textos novos", lambda: documentos.coletar(limite=60)))
    if forcar or not all(_hoje_ja_coletou("RSS", f) for f in parametros()["rede"]["feeds"]):
        etapas.append(("Notícias (RSS)", noticias.coletar))
    for nome, f in etapas:
        avisar(nome)
        try:
            f()
        except Exception as e:
            log.exception(nome)
            avisos.append(f"{nome}: {type(e).__name__}: {e}")
    c = lake.ler("bronze", "controle")
    falhas = c[(c.erro != "") & (pd.to_datetime(c.momento) >= inicio)]
    return avisos + [f"{x.fonte} {x.alvo}: {x.erro[:120]}" for x in falhas.itertuples()]


def _versoes_bronze() -> str:
    return "|".join(f"{t}:{lake.versao('bronze', t)}" for t in FONTES_OURO if lake.existe("bronze", t))


def transformar(avisar=lambda m: None, forcar: bool = False) -> bool:
    """Refaz Prata e Ouro somente se alguma tabela da Bronze mudou desde a última vez."""
    from src.nlp import sentimento
    from src.processing import fundamentos, precos, textos
    atual = _versoes_bronze()
    feito = controle.ultimo("transformacao", "ouro")
    if not forcar and feito and atual in feito["extra"] and lake.existe("gold", "base"):
        return False
    for nome, f in (("Prata: preços", precos.construir), ("Prata: balanços", fundamentos.construir),
                    ("Prata: textos", textos.construir), ("Sentimento e Ouro", ouro.construir)):
        avisar(nome)
        f()
    controle.registrar("transformacao", "ouro", len(lake.ler("gold", "base")), extra={"bronze": atual})
    return True


def pontuar_noticias() -> pd.DataFrame:
    """Sentimento das notícias de RSS (só para o painel)."""
    from src.nlp import sentimento
    if not lake.existe("silver", "noticias"):
        return pd.DataFrame()
    n = lake.ler("silver", "noticias")
    s = sentimento.atualizar("noticias", "sentimento_noticias")
    return n.merge(s[["hash_conteudo", "p_pos", "p_neg", "p_neu", "escore", "classe"]], on="hash_conteudo", how="left")


def run_pipeline(ticker: str, period: str = "2 anos", atualizar: bool = True, treinar: bool = False, avisar=lambda m: None) -> dict:
    """Executa o fluxo completo para um ativo e devolve tudo o que o painel precisa.

    ticker: código cadastrado em config/ativos.yaml.
    period: chave de PERIODOS.
    atualizar: consulta as fontes (no máximo uma vez por dia cada). False usa só o que está no lake.
    treinar: treina de novo se não houver modelo para a base congelada atual. Nunca treina sem pedido.
    """
    if ticker not in tickers():
        raise ValueError(f"Ativo {ticker} não está em config/ativos.yaml. Cadastrados: {', '.join(tickers())}")
    if period not in PERIODOS:
        raise ValueError(f"Período inválido: {period}. Use um de {list(PERIODOS)}")
    from src.models import experimento
    avisos = coletar(avisar) if atualizar else []
    refeito = transformar(avisar) if atualizar or not lake.existe("gold", "base") else False
    avisar("Modelos")
    if treinar and experimento.carregar() is None:
        avisar("Treinando modelos")
        experimento.executar(avisar)
    pacote = experimento.carregar(estrito=False)
    n = PERIODOS[period]
    corte = lambda d: d.sort_values("data").tail(n) if n else d.sort_values("data")
    precos_s = lake.ler("silver", "precos")
    precos_s["data"] = pd.to_datetime(precos_s.data)
    base = lake.ler("gold", "base")
    base["data"] = pd.to_datetime(base.data)
    base = base.sort_values(["ticker", "data"]).reset_index(drop=True)
    textos = lake.ler("gold", "textos_classificados")
    textos = textos[textos.ticker == ticker].sort_values("entregue_em", ascending=False)
    fund = lake.ler("silver", "fundamentos")
    saida = {"ticker": ticker, "periodo": period, "avisos": avisos, "refeito": refeito,
             "cadastro": lake.ler("bronze", "cadastro").set_index("ticker").loc[ticker].to_dict(),
             "precos": corte(precos_s[precos_s.ticker == ticker]), "ibov": corte(precos_s[precos_s.ticker == "^BVSP"]),
             "base": corte(base[base.ticker == ticker]), "textos": textos,
             "fundamentos": fund[fund.ticker == ticker].sort_values("periodo"), "controle": lake.ler("bronze", "controle")}
    try:
        noticias = pontuar_noticias() if atualizar else (lake.ler("silver", "noticias").merge(
            lake.ler("silver", "sentimento_noticias"), on="hash_conteudo", how="left") if lake.existe("silver", "sentimento_noticias") else pd.DataFrame())
        saida["noticias"] = noticias[noticias.ticker == ticker].sort_values("publicado_em", ascending=False) if len(noticias) else noticias
    except Exception as e:
        saida["noticias"] = pd.DataFrame()
        avisos.append(f"Notícias: {type(e).__name__}: {e}")
    if pacote is None:
        saida.update(resultados=None, previsoes=None, proxima=None)
        return saida
    modelos, resultados = pacote
    recente = base[base.data >= base.data.max() - pd.Timedelta(days=900)].reset_index(drop=True)
    prev = recente[["ticker", "data", "alvo", "ret_prox", "ret_exec"]].copy()
    for nome in ("Random Forest", "LSTM"):
        prev[nome] = modelos[nome].prever(recente)
    prev = prev[prev.ticker == ticker].dropna(subset=["Random Forest", "LSTM"])
    ult = prev.iloc[-1]
    saida.update(resultados=resultados, previsoes=corte(prev), proxima={
        "data_sinal": ult.data, **{m: {"probabilidade": float(ult[m]), "limiar": resultados["modelos"][m]["limiar"],
                                       "sinal": "alta" if ult[m] > resultados["modelos"][m]["limiar"] else "sem sinal"}
                                   for m in ("Random Forest", "LSTM")}})
    return saida


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    r = run_pipeline(sys.argv[1] if len(sys.argv) > 1 else "PETR4", treinar="--treinar" in sys.argv, avisar=print)
    print({k: (v.shape if hasattr(v, "shape") else v) for k, v in r.items() if k not in ("resultados", "cadastro")})
