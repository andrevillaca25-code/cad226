"""Painel Streamlit. Toda a lógica está em src/; aqui só há apresentação.

Rodar na raiz do projeto: streamlit run dashboard/app.py
"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import graficos as g  # noqa: E402
from src.config import parametros, tickers  # noqa: E402
from src.pipeline.run import PERIODOS, run_pipeline  # noqa: E402

st.set_page_config(page_title="CAD226 - Lakehouse, PLN e previsão", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def carregar(ticker: str, periodo: str, atualizar: bool, marca: int) -> dict:
    """Resultado do pipeline em cache por uma hora. `marca` muda quando o usuário pede nova coleta."""
    return run_pipeline(ticker, periodo, atualizar=atualizar)


def num(x, casas=2, pct=False, prefixo="") -> str:
    """Número no formato brasileiro."""
    if x is None or pd.isna(x):
        return "n/d"
    s = f"{x * 100 if pct else x:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{prefixo}{s}{'%' if pct else ''}"


with st.sidebar:
    st.header("Seleção")
    ticker = st.selectbox("Ativo", tickers())
    periodo = st.selectbox("Período", list(PERIODOS), index=2)
    st.session_state.setdefault("marca", 0)
    coletar = st.checkbox("Consultar as fontes ao selecionar", value=os.environ.get("CAD226_COLETAR", "1") == "1", key="coletar")
    if st.button("Recarregar", width="stretch"):
        st.session_state["marca"] += 1
    st.caption("Com a opção ligada, a seleção busca na CVM, no Yahoo, no Banco Central e nos portais só o que falta, no máximo uma vez por dia "
               "(a primeira vez leva alguns minutos). Desligada, usa o que já está no Lakehouse. Os modelos não são treinados de novo a cada tela.")

try:
    with st.spinner(f"Executando run_pipeline({ticker}, {periodo})..."):
        r = carregar(ticker, periodo, coletar, st.session_state["marca"])
except Exception as e:
    st.error(f"Não foi possível montar o painel de {ticker}: {e}")
    st.stop()

for a in r["avisos"]:
    st.warning(f"Fonte com problema (dados anteriores mantidos): {a}")

p, res = r["precos"], r["resultados"]
st.title(f"{ticker}: {r['cadastro']['nome']}")
st.caption(f"{r['cadastro']['setor']} | CNPJ {r['cadastro']['cnpj']} | código CVM {r['cadastro']['cd_cvm']} | preços até {p.data.max():%d/%m/%Y}")

ult, ant = p.iloc[-1], p.iloc[-2]
c = st.columns(5)
c[0].metric("Fechamento", num(ult.fechamento, prefixo="R$ "), num(ult.ret, pct=True))
c[1].metric("Retorno no período", num(p.indice_rt.iloc[-1] / p.indice_rt.iloc[0] - 1, 1, pct=True))
c[2].metric("Volatilidade 21 pregões (a.a.)", num(p.ret_log.tail(21).std() * 252 ** 0.5, 1, pct=True))
c[3].metric("Maior queda diária", num(p.ret.min(), 1, pct=True))
c[4].metric("Docs CVM, 21 pregões", num(r["base"].n_docs21.iloc[-1], 0))

abas = st.tabs(["Mercado", "Balanços", "Notícias e sentimento", "Modelos", "Previsão", "Custos", "Dados e fontes"])

with abas[0]:
    st.plotly_chart(g.precos(p, ticker), width="stretch")
    st.plotly_chart(g.acumulado(p, r["ibov"], ticker), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(g.volatilidade(p), width="stretch")
    b.plotly_chart(g.retornos(p), width="stretch")

with abas[1]:
    f = r["fundamentos"]
    if f.empty:
        st.info("Sem demonstrações consolidadas na CVM para este ativo.")
    else:
        u = f.dropna(subset=["lucro_liquido_12m"]).iloc[-1]
        c = st.columns(6)
        c[0].metric("Receita 12m (R$ bi)", num(u.receita_12m / 1000, 1))
        c[1].metric("Lucro 12m (R$ bi)", num(u.lucro_liquido_12m / 1000, 1))
        c[2].metric("EBITDA 12m (R$ bi)", num(u.ebitda_12m / 1000, 1))
        c[3].metric("Patrimônio (R$ bi)", num(u.patrimonio_liquido / 1000, 1))
        c[4].metric("Dívida bruta (R$ bi)", num(u.divida_bruta / 1000, 1))
        c[5].metric("ROE 12m", num(u.roe_12m, 1, pct=True))
        st.caption(f"Período {u.periodo:%d/%m/%Y}, público desde {u.disponivel_em:%d/%m/%Y} (versão {u.versao}). Bancos não têm EBITDA nem dívida comparável: aparecem como n/d.")
        a, b = st.columns(2)
        a.plotly_chart(g.balancos(f, "receita_12m", "Receita, 12 meses"), width="stretch")
        b.plotly_chart(g.balancos(f, "lucro_liquido_12m", "Lucro líquido, 12 meses"), width="stretch")
        with st.expander("Tabela completa"):
            st.dataframe(f.sort_values("periodo", ascending=False), width="stretch", hide_index=True)

with abas[2]:
    st.caption("O tom do texto não é previsão de retorno: um texto positivo pode já estar no preço. Modelo: " + parametros()["sentimento"]["modelo"])
    a, b = st.columns(2)
    a.plotly_chart(g.sentimento(r["base"]), width="stretch")
    b.plotly_chart(g.contagem_textos(r["base"]), width="stretch")
    st.subheader("Notícias recentes (RSS)")
    if len(r["noticias"]):
        n = r["noticias"][["publicado_em", "fonte", "titulo", "classe", "escore", "url"]]
        st.dataframe(n, width="stretch", hide_index=True, column_config={"url": st.column_config.LinkColumn("link"), "escore": st.column_config.NumberColumn(format="%.2f")})
    else:
        st.info("Nenhuma notícia sobre este ativo nos feeds coletados até agora.")
    st.subheader("Fatos relevantes e comunicados (CVM)")
    t = r["textos"][["entregue_em", "categoria", "assunto", "classe", "escore", "p_pos", "p_neg", "p_neu", "nivel_texto", "url"]].head(200)
    st.dataframe(t, width="stretch", hide_index=True, column_config={"url": st.column_config.LinkColumn("documento"), "entregue_em": st.column_config.DateColumn(format="DD/MM/YYYY"),
                 **{k: st.column_config.NumberColumn(format="%.2f") for k in ("escore", "p_pos", "p_neg", "p_neu")}})

with abas[3]:
    if res is None:
        st.info("Ainda não há modelo treinado para a base atual. Rode: python -m src.pipeline.run PETR4 --treinar")
    else:
        d = pd.DataFrame(res["divisao"]).head(3)
        st.caption("Avaliação única, fora da amostra, com os 10 ativos: " + "; ".join(f"{x.bloco} {x.inicio} a {x.fim}" for x in d.itertuples()))
        a, b = st.columns(2)
        a.plotly_chart(g.comparar(res, "accuracy", "Acurácia no teste, IC 95%", 0.5), width="stretch")
        b.plotly_chart(g.comparar(res, "auc", "AUC no teste, IC 95% (0,5 = acaso)", 0.5), width="stretch")
        cols = ["accuracy", "precision", "recall", "f1", "mcc", "auc", "brier", "previu_alta_%"]
        st.dataframe(pd.DataFrame({n: m["teste_0.5"] for n, m in res["modelos"].items()}).T[cols].astype(float).round(4), width="stretch")
        a, b = st.columns(2)
        for col, nome in ((a, "Random Forest"), (b, "LSTM")):
            col.plotly_chart(g.confusao(res["modelos"][nome]["teste_0.5"], f"{nome}: matriz de confusão (teste)"), width="stretch")
        with st.expander("Datas de cada divisão e de cada fold"):
            st.dataframe(pd.DataFrame(res["divisao"]), width="stretch", hide_index=True)

with abas[4]:
    if res is None:
        st.info("Sem modelo treinado.")
    else:
        x = r["proxima"]
        st.subheader(f"Sinal calculado com dados até {x['data_sinal']:%d/%m/%Y}")
        a, b = st.columns(2)
        for col, nome in ((a, "Random Forest"), (b, "LSTM")):
            col.metric(f"{nome}: probabilidade de alta no pregão seguinte", num(x[nome]["probabilidade"], 1, pct=True))
            col.caption(f"Decisão: {x[nome]['sinal']} (limiar {num(x[nome]['limiar'], 1, pct=True)})")
        st.warning("Uso acadêmico. Fora da amostra, veja na aba Modelos o quanto estes sinais se distinguem do acaso. Não é recomendação de investimento.")
        st.plotly_chart(g.probabilidades(r["previsoes"], {n: res["modelos"][n]["limiar"] for n in g.CORES}), width="stretch")

with abas[5]:
    if res is None:
        st.info("Sem modelo treinado.")
    else:
        m, cu = res["matriz_custos"], parametros()["custos"]
        c = st.columns(3)
        c[0].metric("Custo de um falso positivo", num(m["FP"], 2, pct=True))
        c[1].metric("Custo de um falso negativo", num(m["FN"], 2, pct=True))
        c[2].metric("Limiar teórico FP/(FP+FN)", num(res["limiar_teorico"], 3))
        st.caption(f"Hipóteses por lado: tarifas {num(cu['tarifas_bps'], 0)} bps e spread/slippage {num(cu['spread_slippage_bps'], 0)} bps. Erros estimados só no treino; limiar ajustado só na validação. "
                   "Simulação: sinal após o fechamento, execução na abertura seguinte, peso igual por ativo, caixa em CDI, sem venda a descoberto.")
        lin = {f"{n} | {k}": v for n in ("Random Forest", "LSTM") for k, v in res["modelos"][n]["simulacao"].items()}
        lin.update(res["referencias"])
        st.dataframe(pd.DataFrame(lin).T.astype(float).round(2), width="stretch")
        st.subheader("Retorno anual conforme o custo por lado")
        st.dataframe(pd.DataFrame({n: res["modelos"][n]["sensibilidade"] for n in ("Random Forest", "LSTM")}).astype(float).round(2).rename_axis("custo (bps)"), width="stretch")

with abas[6]:
    st.caption("Trilha de auditoria: cada coleta grava fonte, registros novos e atualizados, última data e erros.")
    k = r["controle"].sort_values("momento", ascending=False).drop_duplicates(["fonte", "alvo"])
    st.dataframe(k[["momento", "fonte", "alvo", "novos", "atualizados", "ultima_data", "erro"]], width="stretch", hide_index=True)
    if res:
        st.caption(f"Modelos: digital {res['digital']}, gerados em {res['gerado_em']}, versão {res['versao_ouro']} da tabela gold/base.")
