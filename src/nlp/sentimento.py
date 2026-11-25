"""Sentimento de textos financeiros em português.

Método principal: FinBERT-PT-BR (BERTimbau treinado em notícias financeiras brasileiras).
Método de conferência: dicionário curto de termos financeiros, transparente e auditável.
Sentimento é o tom do texto. Não é previsão de retorno.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

from src.config import parametros
from src.storage import controle, lake

POSITIVOS = """aumento alta cresce crescimento recorde lucro lucros ganho ganhos eleva elevacao supera superou expansao
aprova aprovacao conclusao conclui concluiu acordo parceria contrato vitoria favoravel recompra dividendos jcp melhora
upgrade elevou reducao_da_divida premio conquista inaugura investimento sinergias""".split()
NEGATIVOS = """queda cai recuo prejuizo perda perdas reducao reduz rebaixamento rebaixa renuncia multa acao_judicial processo
investigacao acidente rompimento paralisacao suspensao suspende interdicao greve impairment baixa_contabil atraso inadimplencia
desfavoravel condenacao fraude vazamento cancelamento rescisao arbitragem incendio embargo denuncia""".split()


def _normal(s: str) -> str:
    return unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()


def dicionario(textos: list[str]) -> np.ndarray:
    """Escore em [-1, 1]: (positivos - negativos) / (positivos + negativos), ou 0 sem termos."""
    def conta(t: str, termos: list[str]) -> int:
        return sum(len(re.findall(rf"\b{w.replace('_', ' ')}\b", t)) for w in termos)
    saida = []
    for t in map(_normal, textos):
        p, n = conta(t, POSITIVOS), conta(t, NEGATIVOS)
        saida.append((p - n) / (p + n) if p + n else 0.0)
    return np.array(saida)


class FinBertPtBr:
    """Classificador de três classes. O modelo só é carregado quando há texto novo para pontuar."""

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from src.ingestion.rede import preparar_certificados
        preparar_certificados()
        cfg = parametros()["sentimento"]
        self.nome = f"{cfg['modelo']}@{cfg['revisao'][:8]}"
        self.max_tokens, self.lote = cfg["max_tokens"], cfg["lote"]
        self.torch = torch
        self.disp = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(cfg["modelo"], revision=cfg["revisao"])
        self.modelo = AutoModelForSequenceClassification.from_pretrained(cfg["modelo"], revision=cfg["revisao"]).to(self.disp).eval()
        self.rotulos = [self.modelo.config.id2label[i] for i in range(3)]

    def pontuar(self, textos: list[str]) -> pd.DataFrame:
        """Probabilidades por texto. Usa softmax explícito: o config do modelo pede multi-rótulo,
        o que faria o pipeline padrão aplicar sigmoide e as probabilidades não somariam 1."""
        ordem = np.argsort([len(t) for t in textos])
        probs = np.zeros((len(textos), 3))
        for i in range(0, len(textos), self.lote):
            idx = ordem[i:i + self.lote]
            e = self.tok([textos[j] for j in idx], truncation=True, max_length=self.max_tokens, padding=True, return_tensors="pt").to(self.disp)
            with self.torch.no_grad():
                probs[idx] = self.torch.softmax(self.modelo(**e).logits, dim=1).cpu().numpy()
        df = pd.DataFrame(probs, columns=self.rotulos)
        df = df.rename(columns={"POSITIVE": "p_pos", "NEGATIVE": "p_neg", "NEUTRAL": "p_neu"})
        df["escore"] = df.p_pos - df.p_neg
        df["classe"] = df[["p_pos", "p_neg", "p_neu"]].idxmax(axis=1).map({"p_pos": "positivo", "p_neg": "negativo", "p_neu": "neutro"})
        return df


def atualizar(tabela_textos: str = "textos", destino: str = "sentimento", so_integral: bool = False) -> pd.DataFrame:
    """Pontua só os conteúdos ainda sem escore nesta versão do modelo e devolve a tabela completa."""
    cfg = parametros()["sentimento"]
    versao = f"{cfg['modelo']}@{cfg['revisao'][:8]}"
    t = lake.ler("silver", tabela_textos)
    if so_integral:
        t = t[t.nivel_texto == "integral"]
    t = t[["hash_conteudo", "conteudo"]].drop_duplicates("hash_conteudo")
    feito = lake.ler("silver", destino) if lake.existe("silver", destino) else pd.DataFrame(columns=["hash_conteudo", "modelo"])
    novos = t[~t.hash_conteudo.isin(feito.loc[feito.modelo == versao, "hash_conteudo"])]
    if len(novos):
        modelo, total = FinBertPtBr(), {"novos": 0, "atualizados": 0}
        for i in range(0, len(novos), 400):
            bloco = novos.iloc[i:i + 400]
            r = modelo.pontuar(bloco.conteudo.tolist())
            r.insert(0, "hash_conteudo", bloco.hash_conteudo.to_numpy())
            r["dicionario"] = dicionario(bloco.conteudo.tolist())
            r["modelo"] = versao
            res = lake.upsert("silver", destino, r, ["hash_conteudo", "modelo"])
            total = {k: total[k] + res[k] for k in total}
            print(f"sentimento: {min(i + 400, len(novos))}/{len(novos)}", flush=True)
        controle.registrar("FinBERT-PT-BR", destino, **total, extra={"modelo": versao})
    s = lake.ler("silver", destino)
    return s[s.modelo == versao]
