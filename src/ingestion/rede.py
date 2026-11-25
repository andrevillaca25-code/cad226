"""Acesso HTTP identificado, com pausa entre chamadas e novas tentativas."""
import os
import ssl
import sys
import tempfile
import threading
import time
from pathlib import Path

import certifi
import requests

from src.config import parametros

_ultima: dict[str, float] = {}
_trava = threading.Lock()


def _aguardar_vez(servidor: str, pausa: float) -> None:
    """Reserva o próximo horário livre do servidor; vale também entre threads."""
    with _trava:
        vez = max(time.monotonic(), _ultima.get(servidor, -1e9) + pausa)
        _ultima[servidor] = vez
    time.sleep(max(0.0, vez - time.monotonic()))


def preparar_certificados() -> None:
    """No Windows, acrescenta ao certifi as raízes em que o sistema já confia.

    Necessário quando um antivírus inspeciona TLS. A verificação de certificados continua ligada.
    """
    if sys.platform != "win32" or os.environ.get("CURL_CA_BUNDLE"):
        return
    raizes = [ssl.DER_cert_to_PEM_cert(c) for c, tipo, _ in ssl.enum_certificates("ROOT") if tipo == "x509_asn"]
    destino = Path(tempfile.gettempdir()) / "cad226_ca.pem"
    destino.write_text(Path(certifi.where()).read_text(encoding="utf-8") + "\n" + "\n".join(raizes), encoding="utf-8")
    for var in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        os.environ[var] = str(destino)


def baixar(url: str, pausa: float = 1.0, tentativas: int = 3) -> requests.Response:
    """GET com identificação honesta, respeitando `pausa` segundos entre chamadas ao mesmo servidor."""
    preparar_certificados()
    servidor = url.split("/")[2]
    erro: Exception | None = None
    for n in range(tentativas):
        _aguardar_vez(servidor, pausa)
        try:
            r = requests.get(url, timeout=90, headers={"User-Agent": parametros()["rede"]["user_agent"]})
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            erro = e
            if getattr(e.response, "status_code", None) in (403, 404):
                break
            time.sleep(2 * (n + 1))
    raise ConnectionError(f"{url}: {erro}")
