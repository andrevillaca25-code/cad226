"""Prata de textos: limpeza, deduplicação e data de disponibilidade."""
import hashlib
import re

import numpy as np
import pandas as pd

from src.config import parametros
from src.storage import lake

_CABECALHO = re.compile(r"(companhia aberta|cnpj|nire|\bri@|relações com investidores|\btel\b|www\.|e-mail:|fato relevante|comunicado ao mercado)", re.I)
_RESSALVA = re.compile(r"Este documento pode conter previsões.*?aqui contidas\.", re.I | re.S)
_INGLES = re.compile(r"(MATERIAL FACT|PUBLICLY HELD COMPANY|PUBLICLY LISTED COMPANY|NOTICE TO THE MARKET)", re.I)
_INICIO = re.compile(r"\b(comunica|comunicam|informa|informam|informar|anuncia|anunciam|esclarece|divulga|torna público|vem a público)\b", re.I)


def limpar(texto: str) -> str:
    """Deixa só a parte informativa em português.

    Remove a ressalva legal da Petrobras, a versão em inglês e as linhas de cabeçalho; depois começa no
    primeiro verbo de anúncio ("informa", "comunica"...), pulando o preâmbulo jurídico.
    """
    t = _RESSALVA.sub(" ", texto or "")
    m = _INGLES.search(t)
    if m and m.start() > 200:
        t = t[:m.start()]
    linhas = [l.strip() for l in t.splitlines()]
    linhas = [l for l in linhas if l and not (len(l) < 110 and _CABECALHO.search(l))]
    t = re.sub(r"\s+", " ", " ".join(linhas)).strip()
    m = _INICIO.search(t[:1200])
    return t[m.start():] if m else t


def construir() -> pd.DataFrame:
    """Bronze -> silver/textos: um registro por documento, com texto limpo e chave de conteúdo.

    Duplicata é o mesmo conteúdo, da mesma empresa, no mesmo dia; ou um texto integral já publicado antes.
    Assuntos genéricos repetidos em dias diferentes ("Notícia divulgada na mídia") são documentos distintos.

    Sem hora na CVM, o documento entregue no dia D só vale a partir do pregão seguinte a D.
    """
    ipe = lake.ler("bronze", "cvm_ipe")
    cad = lake.ler("bronze", "cadastro")
    ipe["doc_id"] = ipe.Protocolo_Entrega + "_" + ipe.Versao
    df = ipe.merge(cad[["ticker", "cnpj"]], left_on="CNPJ_Companhia", right_on="cnpj")
    if lake.existe("bronze", "cvm_textos"):
        t = lake.ler("bronze", "cvm_textos")[["doc_id", "texto", "limitacao"]]
        df = df.merge(t, on="doc_id", how="left")
    else:
        df = df.assign(texto="", limitacao="texto nao coletado")
    df["limitacao"] = df.limitacao.fillna("texto nao coletado")
    df["assunto"] = df.Assunto.fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    usa_texto = df.Categoria.isin(parametros()["dados"]["texto_integral"])
    df["corpo"] = df.texto.fillna("").where(usa_texto, "").map(limpar)
    df.loc[~usa_texto, "limitacao"] = "so assunto (categoria)"
    df["conteudo"] = (df.assunto + ". " + df.corpo).str.strip(". ").str.slice(0, 2500)
    df["nivel_texto"] = np.where(df.corpo.str.len() >= 80, "integral", np.where(df.assunto != "", "assunto", "vazio"))
    df["entregue_em"] = pd.to_datetime(df.Data_Entrega)
    df["reapresentacao"] = ~df.Tipo_Apresentacao.str.startswith("AP")
    df["hash_conteudo"] = [hashlib.sha1(c.encode()).hexdigest() for c in df.conteudo]
    df = df[df.nivel_texto != "vazio"].sort_values(["entregue_em", "doc_id"])
    df = df.drop_duplicates(["ticker", "hash_conteudo", "entregue_em"], keep="first")
    repetido = df.duplicated(["ticker", "hash_conteudo"], keep="first") & (df.reapresentacao | (df.nivel_texto == "integral"))
    df = df[~repetido]
    df = df.rename(columns={"Categoria": "categoria", "Link_Download": "url"})
    df["fonte"] = "CVM/IPE"
    df = df[["doc_id", "ticker", "categoria", "assunto", "conteudo", "nivel_texto", "limitacao", "entregue_em",
             "reapresentacao", "hash_conteudo", "fonte", "url"]].reset_index(drop=True)
    lake.sobrescrever("silver", "textos", df)
    return df
