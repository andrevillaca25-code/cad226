"""Testes de integridade, qualidade e ausência de vazamento. Rodar com: python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.config import parametros, tickers
from src.evaluation import custos, divisao
from src.features import ouro
from src.models.lstm import posicoes_janela
from src.nlp.sentimento import dicionario
from src.processing import precos as prata_precos
from src.processing.textos import limpar
from src.storage import lake


@pytest.fixture(scope="module")
def base():
    return ouro.base_modelagem()


def test_cadastro_completo():
    c = lake.ler("bronze", "cadastro")
    assert set(c.ticker) == set(tickers())
    assert c.cnpj.is_unique and c.cd_cvm.notna().all()


def test_upsert_idempotente(tmp_path, monkeypatch):
    monkeypatch.setattr(lake, "caminho", lambda camada, tabela: str(tmp_path / tabela))
    df = pd.DataFrame({"k": ["a", "b"], "v": [1.0, 2.0]})
    assert lake.upsert("bronze", "t", df, ["k"]) == {"novos": 2, "atualizados": 0}
    assert lake.upsert("bronze", "t", df, ["k"]) == {"novos": 0, "atualizados": 0}
    assert lake.upsert("bronze", "t", df.assign(v=[1.0, 9.0]), ["k"]) == {"novos": 0, "atualizados": 1}
    assert lake.ler("bronze", "t", versao=0).v.tolist() == [1.0, 2.0]


def test_calendario_sem_feriados():
    p = lake.ler("silver", "precos")
    datas = set(pd.to_datetime(p.data).dt.strftime("%Y-%m-%d"))
    assert not datas & {"2017-09-07", "2017-12-25", "2018-01-25"}
    assert "2026-10-26" in datas, "o circuit breaker é evento real e precisa ficar"


def test_barras_coerentes_e_sem_duplicatas():
    p = lake.ler("silver", "precos")
    assert p.barra_coerente.all()
    assert not p.duplicated(["ticker", "data"]).any()


def test_evento_societario_neutralizado(base):
    p = lake.ler("silver", "precos")
    d = p[(p.ticker == "ITUB4") & (pd.to_datetime(p.data) == "2021-10-04")]
    assert d.ret.iloc[0] == 0 and d.evento_societario.iloc[0]
    assert base[(base.ticker == "ITUB4") & (base.data_alvo == "2021-10-04")].alvo.isna().all()


def test_retorno_total_nao_depende_de_proventos_futuros():
    p = lake.ler("bronze", "precos")
    g = p[p.ticker == "PETR4"].assign(data=lambda d: pd.to_datetime(d.data)).sort_values("data")
    inteiro, parcial = prata_precos.ajustar(g), prata_precos.ajustar(g.head(1000))
    np.testing.assert_allclose(inteiro.indice_rt.head(1000), parcial.indice_rt)


@pytest.mark.parametrize("ticker,periodo,tipo,codigo,item", [
    ("PETR4", "2026-09-30", "ITR", "3.11", "lucro_liquido_acum"), ("ITUB4", "2026-09-30", "ITR", "3.09", "lucro_liquido_acum"),
    ("BBAS3", "2026-09-30", "ITR", "3.11", "lucro_liquido_acum"), ("WEGE3", "2025-12-31", "DFP", "3.01", "receita_acum")])
def test_contas_conferem_com_o_arquivo_da_cvm(ticker, periodo, tipo, codigo, item):
    cnpj = lake.ler("bronze", "cadastro").set_index("ticker").cnpj[ticker]
    c = lake.ler("bronze", "cvm_contas")
    c = c[(c.CNPJ_CIA == cnpj) & (c.DT_REFER == periodo) & (c.tipo == tipo) & (c.demonstracao == "DRE_con") & (c.CD_CONTA == codigo)]
    c = c[c.DT_INI_EXERC == f"{periodo[:4]}-01-01"]
    f = lake.ler("silver", "fundamentos")
    f = f[(f.ticker == ticker) & (pd.to_datetime(f.periodo) == periodo) & (f.tipo == tipo)]
    assert f[item].iloc[0] == pytest.approx(float(c.VL_CONTA.iloc[0]) / 1000)


def test_doze_meses_confere_com_conta_manual():
    f = lake.ler("silver", "fundamentos")
    f["periodo"] = pd.to_datetime(f.periodo)
    f = f[f.ticker == "PETR4"].set_index("periodo")
    esperado = f.lucro_liquido_acum["2026-09-30"] + f.lucro_liquido_acum["2025-12-31"] - f.lucro_liquido_acum["2025-09-30"]
    assert f.lucro_liquido_12m["2026-09-30"] == pytest.approx(esperado, rel=0.02)


def test_balanco_so_depois_de_publico(base):
    b = base.dropna(subset=["balanco_disponivel_em"])
    assert (pd.to_datetime(b.balanco_disponivel_em) < b.data).all()


def test_documento_so_no_pregao_seguinte():
    t = lake.ler("gold", "textos_classificados").dropna(subset=["data"])
    assert (pd.to_datetime(t.data) > pd.to_datetime(t.entregue_em)).all()


def test_alterar_o_futuro_nao_muda_o_passado(base):
    """Teste de vazamento: preços posteriores ao corte são adulterados; nenhuma variável até o corte pode mudar."""
    corte = pd.Timestamp("2023-06-30")
    p = lake.ler("silver", "precos")
    p["data"] = pd.to_datetime(p.data)
    adulterado = p.copy()
    futuro = adulterado.data > corte
    for c in ("ret", "ret_log", "indice_rt", "volume", "maxima", "minima", "fechamento", "abertura", "ret_intradia", "ret_abertura"):
        adulterado.loc[futuro, c] = adulterado.loc[futuro, c] * 3 + 0.01
    nova = ouro.construir(precos=adulterado, gravar=False).set_index(["ticker", "data"]).sort_index()
    orig = base.set_index(["ticker", "data"]).sort_index()
    antes = orig.index[orig.index.get_level_values("data") <= corte]
    pd.testing.assert_frame_equal(nova.loc[antes, ouro.NUMERICAS], orig.loc[antes, ouro.NUMERICAS], check_exact=False, rtol=1e-9)


def test_alvo_e_o_retorno_do_pregao_seguinte(base):
    g = base[base.ticker == "VALE3"].sort_values("data")
    p = lake.ler("silver", "precos")
    ret = p[p.ticker == "VALE3"].assign(data=lambda d: pd.to_datetime(d.data)).set_index("data").ret
    x = g.dropna(subset=["alvo"]).iloc[100:400]
    np.testing.assert_array_equal(x.alvo.to_numpy(), (ret.reindex(x.data_alvo).to_numpy() > 0).astype(float))


def test_divisao_cronologica_sem_sobreposicao(base):
    m = divisao.mascaras(base)
    d = {k: base[v].data for k, v in m.items()}
    assert d["treino"].max() < d["validacao"].min() and d["validacao"].max() < d["teste"].min()
    embargo = parametros()["divisao"]["embargo_pregoes"]
    datas = np.sort(base[m["treino"]].data.unique())
    for a, v in divisao.folds(base, m["treino"]):
        fim, ini = base[a].data.max(), base[v].data.min()
        assert ((datas > np.datetime64(fim)) & (datas < np.datetime64(ini))).sum() >= embargo
        assert not set(base[a].data) & set(base[v].data)


def test_janela_lstm_nao_mistura_ativos_nem_olha_para_frente(base):
    j = parametros()["lstm"]["janela"]
    pos = posicoes_janela(base, j)
    linhas = np.flatnonzero(pos[:, 0] >= 0)
    g = np.random.default_rng(0).choice(linhas, 500, replace=False)
    for i in g:
        assert base.ticker.to_numpy()[pos[i]].tolist() == [base.ticker.iloc[i]] * j
        assert (base.data.to_numpy()[pos[i]] <= base.data.to_numpy()[i]).all()
    m = divisao.mascaras(base)
    i_t = np.flatnonzero(m["treino"] & (pos[:, 0] >= 0))
    assert base.data.to_numpy()[pos[i_t]].max() <= base[m["treino"]].data.max().to_datetime64()


def test_probabilidades_de_sentimento_somam_um():
    s = lake.ler("silver", "sentimento")
    np.testing.assert_allclose(s[["p_pos", "p_neg", "p_neu"]].sum(axis=1), 1, atol=1e-5)
    assert s.modelo.str.contains("@").all()


def test_dicionario_e_limpeza():
    assert dicionario(["lucro recorde", "prejuízo e queda", "assembleia"]).tolist() == [1.0, -1.0, 0.0]
    assert limpar("COMPANHIA ABERTA\nCNPJ 1\nA empresa X, nos termos da lei, informa que vendeu a usina").startswith("informa que vendeu")


def test_matriz_de_custos_e_limiar():
    m = custos.matriz_custos(np.array([0.02, -0.02, 0.01, -0.01]))
    assert m["FP"] > m["FN"] > 0
    y, p = np.array([0, 0, 1, 1]), np.array([0.2, 0.4, 0.6, 0.8])
    assert custos.custo_medio(y, p, custos.escolher_limiar(y, p, m), m) == 0


def test_simulacao_nunca_comprada_rende_cdi(base):
    m = divisao.mascaras(base)["teste"]
    s = custos.simular(base[m], np.zeros(m.sum(), bool))
    np.testing.assert_allclose(s.ret, s.cdi)
    assert s.giro.sum() == 0


def test_custo_maior_nunca_melhora_o_resultado(base):
    m = divisao.mascaras(base)["teste"]
    sinal = np.random.default_rng(1).random(m.sum()) > 0.5
    a, b = (custos.simular(base[m], sinal, custo=c).ret.sum() for c in (0.0, 0.002))
    assert a > b
