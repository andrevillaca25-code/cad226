"""Leitura da configuração e caminhos do projeto."""
from functools import lru_cache
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]


@lru_cache
def parametros() -> dict:
    """Devolve o conteúdo de config/parametros.yaml."""
    return yaml.safe_load((RAIZ / "config" / "parametros.yaml").read_text(encoding="utf-8"))


@lru_cache
def ativos() -> list[dict]:
    """Devolve a lista de ativos de config/ativos.yaml."""
    return yaml.safe_load((RAIZ / "config" / "ativos.yaml").read_text(encoding="utf-8"))["ativos"]


def tickers() -> list[str]:
    """Códigos de negociação cadastrados."""
    return [a["ticker"] for a in ativos()]
