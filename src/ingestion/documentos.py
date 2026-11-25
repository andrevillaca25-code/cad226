"""Texto integral dos fatos relevantes e comunicados (PDFs no sistema RAD da CVM)."""
import hashlib
import io
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pypdf import PdfReader

from src.config import RAIZ, parametros
from src.ingestion.rede import baixar, preparar_certificados
from src.storage import controle, lake

logging.getLogger("pypdf").setLevel(logging.CRITICAL)
PASTA = RAIZ / "data" / "bronze" / "_documentos"
MAX_ERROS = 5


def extrair_texto(conteudo: bytes, paginas: int = 3) -> tuple[str, str]:
    """Texto das primeiras páginas e o motivo da limitação, se houver."""
    if not conteudo.startswith(b"%PDF"):
        return "", "arquivo nao e PDF"
    try:
        leitor = PdfReader(io.BytesIO(conteudo))
        texto = "\n".join((p.extract_text() or "") for p in leitor.pages[:paginas])
    except Exception as e:
        return "", f"falha de leitura: {type(e).__name__}"
    texto = texto.encode("utf-8", "ignore").decode("utf-8").replace("\x00", "")
    return texto[:12000], "" if len(texto.strip()) >= 80 else "sem camada de texto"


def _um_documento(doc_id: str, url: str) -> dict | None:
    """Baixa (ou relê) um PDF e devolve o registro da Bronze; None se a rede falhou."""
    arq = PASTA / f"{doc_id}.pdf"
    try:
        if arq.exists():
            conteudo = arq.read_bytes()
        else:
            conteudo = baixar(url, parametros()["rede"]["pausa_documentos_s"], 2).content
            arq.write_bytes(conteudo)
    except ConnectionError:
        return None
    texto, limitacao = extrair_texto(conteudo)
    return {"doc_id": doc_id, "texto": texto, "limitacao": limitacao, "url": url,
            "sha256": hashlib.sha256(conteudo).hexdigest(), "coletado_em": controle.agora()}


def coletar(limite: int | None = None, lote: int = 90, conexoes: int = 1) -> dict:
    """Baixa os PDFs ainda não coletados (categorias em dados.texto_integral) e grava em bronze/cvm_textos.

    Documentos já processados não são baixados de novo. Uma conexão por vez, com pausa, e parada após
    poucas falhas: em 25/11/2026 o servidor começou a recusar conexões durante a carga inicial.
    """
    PASTA.mkdir(parents=True, exist_ok=True)
    preparar_certificados()
    ipe = lake.ler("bronze", "cvm_ipe")
    ipe["doc_id"] = ipe.Protocolo_Entrega + "_" + ipe.Versao
    feitos = set(lake.ler("bronze", "cvm_textos").doc_id) if lake.existe("bronze", "cvm_textos") else set()
    fila = ipe[~ipe.doc_id.isin(feitos) & ipe.Link_Download.str.startswith("http")
               & ipe.Categoria.isin(parametros()["dados"]["texto_integral"])]
    fila = fila.sort_values("Data_Entrega", ascending=False).head(limite or len(fila))
    total = {"novos": 0, "erros": 0}
    with ThreadPoolExecutor(conexoes) as executor:
        for i in range(0, len(fila), lote):
            bloco = fila.iloc[i:i + lote]
            linhas = list(executor.map(_um_documento, bloco.doc_id, bloco.Link_Download))
            bons = [x for x in linhas if x]
            total["erros"] += len(linhas) - len(bons)
            if bons:
                total["novos"] += lake.upsert("bronze", "cvm_textos", pd.DataFrame(bons), ["doc_id"])["novos"]
            print(f"{min(i + lote, len(fila))}/{len(fila)} textos={total['novos']} erros={total['erros']}", flush=True)
            if total["erros"] > MAX_ERROS:
                controle.registrar("CVM-RAD", "textos", total["novos"], erro="interrompido por excesso de falhas")
                return total
    controle.registrar("CVM-RAD", "textos", total["novos"], extra=total)
    return total


if __name__ == "__main__":
    print(coletar(int(sys.argv[1]) if len(sys.argv) > 1 else None))
