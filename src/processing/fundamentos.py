"""Prata de demonstrações: plano de contas, 12 meses móveis e data de disponibilidade."""
import unicodedata

import numpy as np
import pandas as pd

from src.storage import lake


def _normal(s: str) -> str:
    """Texto sem acento e em minúsculas, para comparar descrições de contas."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().strip()


def _regra(demonstracao: str, codigo: str, descricao: str, banco: bool) -> str | None:
    """Traduz uma linha da CVM para o nome padronizado, ou None.

    O lucro e o patrimônio são achados pela descrição porque o código muda entre empresas
    (lucro líquido é 3.11 na Petrobras e 3.09 no Itaú).
    """
    d = _normal(descricao)
    nivel2 = codigo.count(".") == 1
    if demonstracao == "DRE_con" and nivel2:
        if d.startswith(("lucro/prejuizo consolidado do periodo", "lucro ou prejuizo liquido consolidado do periodo")):
            return "lucro_liquido"
        if codigo == "3.01":
            return "receita"
        if codigo == "3.05" and not banco and "antes do resultado financeiro" in d:
            return "ebit"
    if demonstracao == "BPA_con" and codigo == "1":
        return "ativo_total"
    if demonstracao == "BPP_con":
        if nivel2 and d.startswith("patrimonio liquido"):
            return "patrimonio_liquido"
        if not banco and codigo in ("2.01.04", "2.02.01") and "emprestimos" in d:
            return "divida_cp" if codigo == "2.01.04" else "divida_lp"
    if demonstracao.startswith("DFC") and codigo == "6.01":
        return "fluxo_operacional"
    if demonstracao == "DVA_con" and not banco and codigo == "7.04.01" and "deprecia" in d:
        return "depreciacao"
    return None


FLUXOS = ["receita", "lucro_liquido", "ebit", "fluxo_operacional", "depreciacao"]
ESTOQUES = ["ativo_total", "patrimonio_liquido", "divida_cp", "divida_lp"]


def _valores(contas: pd.DataFrame, bancos: set[str]) -> pd.DataFrame:
    """Uma linha por documento, exercício (último/penúltimo) e item, em R$ milhões, sempre no acumulado do ano."""
    c = contas.copy()
    c["banco"] = c.CNPJ_CIA.isin(bancos)
    unicas = c[["demonstracao", "CD_CONTA", "DS_CONTA", "banco"]].drop_duplicates()
    unicas["item"] = [_regra(a, b, d, e) for a, b, d, e in unicas.itertuples(index=False)]
    c = c.merge(unicas.dropna(subset=["item"]), on=["demonstracao", "CD_CONTA", "DS_CONTA", "banco"])
    c["valor"] = pd.to_numeric(c.VL_CONTA, errors="coerce") * np.where(c.ESCALA_MOEDA.str.upper() == "MIL", 1e-3, 1e-6)
    c["exercicio"] = np.where(c.ORDEM_EXERC.map(_normal).str.startswith("pen"), "anterior", "atual")
    fluxo = c.item.isin(FLUXOS)
    inicio_ano = c.DT_INI_EXERC.str[5:] == "01-01"
    c = c[~fluxo | inicio_ano]
    return c.drop_duplicates(["tipo", "CNPJ_CIA", "DT_REFER", "VERSAO", "exercicio", "item"])


def construir() -> pd.DataFrame:
    """Bronze -> silver/fundamentos: indicadores por documento, com a data em que se tornaram públicos.

    A CVM mantém só a última versão de cada demonstração. Por isso `disponivel_em` é a data de
    recebimento DA VERSÃO que está no arquivo: se houve reapresentação, o dado só passa a valer depois dela.
    """
    cad = lake.ler("bronze", "cadastro")
    bancos = set(cad.loc[cad.setor.str.contains("Banco", case=False), "cnpj"])
    v = _valores(lake.ler("bronze", "cvm_contas"), bancos)
    larga = v.pivot_table(index=["tipo", "CNPJ_CIA", "DT_REFER", "VERSAO"], columns=["exercicio", "item"], values="valor", aggfunc="first")
    larga.columns = [f"{item}__{ex}" for ex, item in larga.columns]
    larga = larga.reset_index()
    idx = lake.ler("bronze", "cvm_indice")
    idx["n_versoes"] = idx.groupby(["tipo", "CNPJ_CIA", "DT_REFER"]).VERSAO.transform("nunique")
    larga = larga.merge(idx[["tipo", "CNPJ_CIA", "DT_REFER", "VERSAO", "DT_RECEB", "n_versoes", "LINK_DOC"]],
                        on=["tipo", "CNPJ_CIA", "DT_REFER", "VERSAO"], how="inner")
    larga = larga.merge(cad[["ticker", "cnpj"]], left_on="CNPJ_CIA", right_on="cnpj")
    for c in FLUXOS + ESTOQUES:
        for ex in ("atual", "anterior"):
            if f"{c}__{ex}" not in larga:
                larga[f"{c}__{ex}"] = np.nan
    larga["periodo"] = pd.to_datetime(larga.DT_REFER)
    larga["disponivel_em"] = pd.to_datetime(larga.DT_RECEB)
    anual = larga[larga.tipo == "DFP"].set_index(["ticker", larga[larga.tipo == "DFP"].periodo.dt.year])
    linhas = []
    for r in larga.itertuples(index=False):
        d = {"ticker": r.ticker, "tipo": r.tipo, "periodo": r.periodo, "versao": r.VERSAO, "disponivel_em": r.disponivel_em,
             "reapresentado": r.n_versoes > 1, "documento": r.LINK_DOC}
        for c in ESTOQUES:
            d[c] = getattr(r, f"{c}__atual")
        chave = (r.ticker, r.periodo.year - 1)
        for c in FLUXOS:
            ytd, ytd_ant = getattr(r, f"{c}__atual"), getattr(r, f"{c}__anterior")
            d[f"{c}_acum"] = ytd
            if r.tipo == "DFP":
                d[f"{c}_12m"], d[f"{c}_12m_ant"] = ytd, ytd_ant
                continue
            d[f"{c}_12m"] = d[f"{c}_12m_ant"] = np.nan
            if chave in anual.index:
                ant = anual.loc[chave]
                if ant.disponivel_em <= r.disponivel_em:
                    d[f"{c}_12m"] = ytd + getattr(ant, f"{c}__atual") - ytd_ant
        linhas.append(d)
    df = pd.DataFrame(linhas).sort_values(["ticker", "periodo", "disponivel_em"])
    df["divida_bruta"] = df.divida_cp.fillna(0) + df.divida_lp.fillna(0)
    df.loc[df.divida_cp.isna() & df.divida_lp.isna(), "divida_bruta"] = np.nan
    df["ebitda_12m"] = df.ebit_12m - df.depreciacao_12m
    df["roe_12m"] = df.lucro_liquido_12m / df.patrimonio_liquido
    df["margem_liquida_12m"] = df.lucro_liquido_12m / df.receita_12m
    df["alavancagem"] = 1 - df.patrimonio_liquido / df.ativo_total
    base = df[["ticker", "periodo", "receita_12m", "lucro_liquido_12m", "disponivel_em"]].rename(
        columns={"receita_12m": "r_ant", "lucro_liquido_12m": "l_ant", "disponivel_em": "disp_ant"})
    base["periodo"] = base.periodo + pd.offsets.QuarterEnd(4)
    df = df.merge(base, on=["ticker", "periodo"], how="left")
    ok = df.disp_ant <= df.disponivel_em
    df["cresc_receita"] = np.where(ok, df.receita_12m / df.r_ant - 1, np.nan)
    df["cresc_lucro"] = np.where(ok, (df.lucro_liquido_12m - df.l_ant) / df.l_ant.abs(), np.nan)
    df["cresc_lucro"] = df.cresc_lucro.clip(-3, 3)
    df = df.drop(columns=["r_ant", "l_ant", "disp_ant", "receita_12m_ant", "lucro_liquido_12m_ant", "ebit_12m_ant",
                          "fluxo_operacional_12m_ant", "depreciacao_12m_ant"])
    lake.sobrescrever("silver", "fundamentos", df)
    return df
