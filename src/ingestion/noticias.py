"""Notícias recentes por RSS. Servem ao painel; não têm histórico suficiente para o treino."""
import hashlib
import re
from datetime import datetime, timezone

import feedparser
import pandas as pd

from src.config import ativos, parametros
from src.ingestion.rede import baixar
from src.storage import controle, lake


def _sem_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def _empresas(texto: str) -> list[str]:
    """Tickers citados, segundo os termos do cadastro. Palavras inteiras, respeitando maiúsculas."""
    return [a["ticker"] for a in ativos()
            if any(re.search(rf"(?<!\w){re.escape(t.strip())}(?!\w)", texto) for t in a["termos"])]


def coletar() -> dict:
    """Lê os feeds permitidos, guarda o que é público (título e resumo) e associa às empresas."""
    total = {"novos": 0, "atualizados": 0}
    for fonte, url in parametros()["rede"]["feeds"].items():
        try:
            f = feedparser.parse(baixar(url, pausa=parametros()["rede"]["pausa_cvm_s"]).content)
            linhas = []
            for e in f.entries:
                titulo, resumo = _sem_html(e.get("title")), _sem_html(e.get("summary"))[:800]
                quando = e.get("published_parsed") or e.get("updated_parsed")
                publicado = datetime(*quando[:6], tzinfo=timezone.utc).isoformat() if quando else ""
                linhas.append({"noticia_id": hashlib.sha1(e.get("link", titulo).encode()).hexdigest(), "titulo": titulo,
                               "conteudo": f"{titulo}. {resumo}", "publicado_em": publicado, "coletado_em": controle.agora(),
                               "fonte": fonte, "url": e.get("link", ""), "tickers": ",".join(_empresas(f"{titulo} {resumo}")),
                               "tipo_documento": "noticia", "limitacao": "titulo e resumo do RSS"})
            r = lake.upsert("bronze", "noticias", pd.DataFrame(linhas), ["noticia_id"])
            total = {k: total[k] + r[k] for k in total}
            controle.registrar("RSS", fonte, **r, ultima_data=max((l["publicado_em"] for l in linhas), default=None))
        except Exception as e:
            controle.registrar("RSS", fonte, erro=f"{type(e).__name__}: {e}"[:300])
    if lake.existe("bronze", "noticias"):
        b = lake.ler("bronze", "noticias").drop_duplicates("url")
        b = b[b.tickers != ""].assign(ticker=lambda d: d.tickers.str.split(",")).explode("ticker")
        if len(b):
            b["hash_conteudo"] = [hashlib.sha1(c.encode()).hexdigest() for c in b.conteudo]
            lake.sobrescrever("silver", "noticias", b.drop(columns="tickers").reset_index(drop=True))
    return total
