"""Dados abertos da CVM: cadastro (FCA), demonstrações (DFP e ITR) e índice de documentos (IPE)."""
import hashlib
import io
import zipfile
from datetime import date

import pandas as pd

from src.config import RAIZ, ativos, parametros, tickers
from src.ingestion.rede import baixar
from src.storage import controle, lake

BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA"
DEMONSTRACOES = ["DRE_con", "BPA_con", "BPP_con", "DFC_MI_con", "DFC_MD_con", "DVA_con"]
PASTA_ZIPS = RAIZ / "data" / "bronze" / "_zips"


def _arquivo(url: str, nome: str, conferir: bool) -> bytes:
    """Conteúdo de um arquivo da CVM. Usa a cópia local; se `conferir`, atualiza no máximo uma vez por dia."""
    local = PASTA_ZIPS / nome
    PASTA_ZIPS.mkdir(parents=True, exist_ok=True)
    ja_hoje = local.exists() and date.fromtimestamp(local.stat().st_mtime) == date.today()
    if not local.exists() or (conferir and not ja_hoje):
        local.write_bytes(baixar(url, pausa=parametros()["rede"]["pausa_cvm_s"]).content)
    return local.read_bytes()


def _zip(tipo: str, ano: int) -> tuple[zipfile.ZipFile, str]:
    """Zip anual e seu hash. Só o ano corrente e o anterior são conferidos de novo."""
    nome = f"{tipo.lower()}_cia_aberta_{ano}.zip"
    conteudo = _arquivo(f"{BASE}/DOC/{tipo}/DADOS/{nome}", nome, conferir=ano >= date.today().year - 1)
    return zipfile.ZipFile(io.BytesIO(conteudo)), hashlib.sha256(conteudo).hexdigest()


def _csv(z: zipfile.ZipFile, nome: str) -> pd.DataFrame:
    """Lê um CSV do zip inteiro como texto, sem interpretar nada (é a Bronze)."""
    return pd.read_csv(z.open(nome), sep=";", encoding="latin-1", dtype=str, keep_default_na=False, quoting=3)


def _ja_processado(fonte: str, ano: int, h: str, cnpjs: list[str]) -> bool:
    """A última carga usou o mesmo arquivo (hash) e já incluía todas as empresas."""
    anterior = controle.ultimo(fonte, str(ano))
    return bool(anterior) and h in anterior["extra"] and all(c in anterior["extra"] for c in cnpjs)


def _ano_encerrado(fonte: str, ano: int, cnpjs: list[str]) -> bool:
    """Ano antigo já carregado para todas as empresas: nem é preciso baixar o arquivo."""
    return ano < date.today().year - 1 and _ja_processado(fonte, ano, "", cnpjs)


def cadastro() -> pd.DataFrame:
    """Descobre CNPJ, código CVM, razão social e setor de cada ticker (FCA + cadastro da CVM)."""
    ano = date.today().year
    z, h = _zip("FCA", ano)
    v = _csv(z, f"fca_cia_aberta_valor_mobiliario_{ano}.csv")
    v["ticker"] = v.Codigo_Negociacao.str.strip().str.upper()
    v = v[v.ticker.isin(tickers())].sort_values("Versao").drop_duplicates("ticker", keep="last")
    bruto = _arquivo(f"{BASE}/CAD/DADOS/cad_cia_aberta.csv", "cad_cia_aberta.csv", conferir=True)
    c = pd.read_csv(io.BytesIO(bruto), sep=";", encoding="latin-1", dtype=str, quoting=3)
    c = c[c.SIT == "ATIVO"].drop_duplicates("CNPJ_CIA")[["CNPJ_CIA", "CD_CVM", "DENOM_SOCIAL", "SETOR_ATIV"]]
    df = v[["ticker", "CNPJ_Companhia"]].merge(c, left_on="CNPJ_Companhia", right_on="CNPJ_CIA")
    df = df.rename(columns={"CNPJ_CIA": "cnpj", "CD_CVM": "cd_cvm", "DENOM_SOCIAL": "nome", "SETOR_ATIV": "setor"})
    df = df[["ticker", "cnpj", "cd_cvm", "nome", "setor"]]
    df["tickers_antigos"] = df.ticker.map({a["ticker"]: ",".join(a.get("tickers_antigos", [])) for a in ativos()})
    df["hash_fca"] = h
    faltam = sorted(set(tickers()) - set(df.ticker))
    r = lake.upsert("bronze", "cadastro", df, ["ticker"])
    controle.registrar("CVM-FCA", "cadastro", **r, erro=f"sem CNPJ: {faltam}" if faltam else None)
    return df


def demonstracoes(tipo: str, ano: int, cnpjs: list[str]) -> None:
    """Grava na Bronze o índice (todas as versões) e as demonstrações consolidadas de DFP ou ITR.

    Só entram linhas das empresas cadastradas; cada campo é mantido como publicado.
    """
    if _ano_encerrado(f"CVM-{tipo}", ano, cnpjs):
        return
    z, h = _zip(tipo, ano)
    if _ja_processado(f"CVM-{tipo}", ano, h, cnpjs):
        return
    pre = f"{tipo.lower()}_cia_aberta"
    idx = _csv(z, f"{pre}_{ano}.csv")
    idx = idx[idx.CNPJ_CIA.isin(cnpjs)].assign(tipo=tipo, hash_zip=h)
    r = lake.upsert("bronze", "cvm_indice", idx, ["tipo", "CNPJ_CIA", "DT_REFER", "VERSAO"])
    partes = []
    for dem in DEMONSTRACOES:
        nome = f"{pre}_{dem}_{ano}.csv"
        if nome in z.namelist():
            d = _csv(z, nome)
            d = d[d.CNPJ_CIA.isin(cnpjs)].assign(tipo=tipo, demonstracao=dem, hash_zip=h)
            if "DT_INI_EXERC" not in d:
                d["DT_INI_EXERC"] = ""
            partes.append(d)
    contas = pd.concat(partes, ignore_index=True)
    chaves = ["tipo", "demonstracao", "CNPJ_CIA", "DT_REFER", "VERSAO", "ORDEM_EXERC",
              "DT_INI_EXERC", "DT_FIM_EXERC", "CD_CONTA", "DS_CONTA"]
    r2 = lake.upsert("bronze", "cvm_contas", contas, chaves)
    controle.registrar(f"CVM-{tipo}", str(ano), r["novos"] + r2["novos"], r["atualizados"] + r2["atualizados"],
                       idx.DT_RECEB.max() if len(idx) else None, extra={"hash": h, "cnpjs": cnpjs})


def indice_ipe(ano: int, cnpjs: list[str]) -> None:
    """Grava na Bronze o índice de fatos relevantes e comunicados ao mercado."""
    if _ano_encerrado("CVM-IPE", ano, cnpjs):
        return
    z, h = _zip("IPE", ano)
    if _ja_processado("CVM-IPE", ano, h, cnpjs):
        return
    d = _csv(z, f"ipe_cia_aberta_{ano}.csv")
    d = d[d.CNPJ_Companhia.isin(cnpjs) & d.Categoria.isin(parametros()["dados"]["categorias_ipe"])]
    d = d.assign(hash_zip=h)
    r = lake.upsert("bronze", "cvm_ipe", d, ["Protocolo_Entrega", "Versao"])
    controle.registrar("CVM-IPE", str(ano), **r, ultima_data=d.Data_Entrega.max() if len(d) else None,
                       extra={"hash": h, "cnpjs": cnpjs})


def atualizar() -> pd.DataFrame:
    """Atualiza cadastro, demonstrações e índice de documentos. Falha de um arquivo não interrompe os demais."""
    cad = cadastro()
    cnpjs = sorted(cad.cnpj)
    p, hoje = parametros()["dados"], date.today().year
    tarefas = [(lambda a=a: demonstracoes("DFP", a, cnpjs), "CVM-DFP", a) for a in range(p["primeiro_ano_cvm"], hoje)]
    tarefas += [(lambda a=a: demonstracoes("ITR", a, cnpjs), "CVM-ITR", a) for a in range(p["primeiro_ano_cvm"] + 1, hoje + 1)]
    tarefas += [(lambda a=a: indice_ipe(a, cnpjs), "CVM-IPE", a) for a in range(p["primeiro_ano_ipe"], hoje + 1)]
    for f, fonte, ano in tarefas:
        try:
            f()
        except Exception as e:
            controle.registrar(fonte, str(ano), erro=f"{type(e).__name__}: {e}"[:300])
    return cad
