"""Controle de atualizações: o que foi coletado, quando, de onde e com qual resultado."""
import json
import logging
from datetime import datetime, timezone

import pandas as pd

from src.storage import lake

log = logging.getLogger("cad226")


def agora() -> str:
    """Instante atual em UTC, formato ISO."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def registrar(fonte: str, alvo: str, novos: int = 0, atualizados: int = 0,
              ultima_data: str | None = None, erro: str | None = None, extra: dict | None = None) -> None:
    """Acrescenta uma linha à tabela bronze/controle."""
    linha = pd.DataFrame([{
        "momento": datetime.now(timezone.utc).isoformat(), "fonte": fonte, "alvo": alvo,
        "novos": int(novos), "atualizados": int(atualizados), "ultima_data": ultima_data or "",
        "erro": erro or "", "extra": json.dumps(extra or {}, ensure_ascii=False)}])
    lake.upsert("bronze", "controle", linha, ["momento", "fonte", "alvo"])
    (log.error if erro else log.info)("%s | %s | novos=%s atualizados=%s %s", fonte, alvo, novos, atualizados, erro or "")


def ultimo(fonte: str, alvo: str) -> dict | None:
    """Último registro bem-sucedido de uma fonte, ou None."""
    if not lake.existe("bronze", "controle"):
        return None
    c = lake.ler("bronze", "controle")
    c = c[(c.fonte == fonte) & (c.alvo == alvo) & (c.erro == "")].sort_values("momento")
    return None if c.empty else c.iloc[-1].to_dict()
