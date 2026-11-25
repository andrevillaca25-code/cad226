"""LSTM em PyTorch: lê os últimos 20 pregões do ativo e estima a probabilidade de alta no seguinte."""
import copy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss
from torch import nn

from src.config import parametros
from src.models.base import Preparador


class Rede(nn.Module):
    """Uma camada LSTM; o estado do último pregão passa por dropout e vira um único número (logit)."""

    def __init__(self, n_variaveis: int, ocultas: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_variaveis, ocultas, batch_first=True)
        self.saida = nn.Sequential(nn.Dropout(dropout), nn.Linear(ocultas, 1))

    def forward(self, x):
        estados, _ = self.lstm(x)
        return self.saida(estados[:, -1]).squeeze(1)


def posicoes_janela(base: pd.DataFrame, janela: int) -> np.ndarray:
    """Para cada linha, os índices das `janela` linhas do MESMO ativo que terminam nela (-1 se faltar histórico).

    `base` precisa estar ordenada por ativo e data. A janela só olha para trás.
    """
    n = len(base)
    inicio = base.groupby("ticker").cumcount().to_numpy()
    pos = np.arange(n)[:, None] + np.arange(-janela + 1, 1)[None, :]
    pos[inicio < janela - 1] = -1
    return pos


class LSTM:
    """Média de redes idênticas com sementes diferentes, para reduzir o acaso da inicialização."""
    nome = "LSTM"
    grade = [{"ocultas": h, "dropout": d} for h in (16, 32) for d in (0.2, 0.5)]

    def __init__(self, variaveis: list[str], ocultas: int = 32, dropout: float = 0.3, epocas: int | None = None):
        self.cfg = parametros()["lstm"]
        self.prep, self.ocultas, self.dropout, self.epocas = Preparador(variaveis), ocultas, dropout, epocas
        self.redes: list[Rede] = []

    def _treinar(self, x, pos, y, i_treino, semente, i_val=None):
        torch.manual_seed(semente)
        g = np.random.default_rng(semente)
        rede = Rede(x.shape[1], self.ocultas, self.dropout)
        otim = torch.optim.Adam(rede.parameters(), lr=self.cfg["taxa"], weight_decay=1e-4)
        perda = nn.BCEWithLogitsLoss()
        melhor, melhor_ep, melhor_estado, sem_melhora = np.inf, 1, None, 0
        for ep in range(1, (self.epocas or self.cfg["epocas_max"]) + 1):
            rede.train()
            ordem = g.permutation(i_treino)
            for k in range(0, len(ordem), self.cfg["lote"]):
                b = ordem[k:k + self.cfg["lote"]]
                otim.zero_grad()
                perda(rede(x[pos[b]]), y[b]).backward()
                nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
                otim.step()
            if i_val is not None:
                p = self._inferir([rede], x, pos, i_val)
                v = log_loss(y[i_val].numpy(), np.clip(p, 1e-6, 1 - 1e-6))
                if v < melhor - 1e-5:
                    melhor, melhor_ep, melhor_estado, sem_melhora = v, ep, copy.deepcopy(rede.state_dict()), 0
                else:
                    sem_melhora += 1
                    if sem_melhora >= self.cfg["paciencia"]:
                        break
        if melhor_estado:
            rede.load_state_dict(melhor_estado)
        return rede, melhor_ep

    @staticmethod
    def _inferir(redes, x, pos, idx, lote=4096) -> np.ndarray:
        soma = np.zeros(len(idx))
        with torch.no_grad():
            for r in redes:
                r.eval()
                soma += np.concatenate([torch.sigmoid(r(x[pos[idx[k:k + lote]]])).numpy() for k in range(0, len(idx), lote)])
        return soma / len(redes)

    def _tensores(self, base: pd.DataFrame):
        x = torch.from_numpy(self.prep.transformar(base))
        pos = posicoes_janela(base, self.cfg["janela"])
        y = torch.from_numpy(base.alvo.fillna(0).to_numpy("float32"))
        return x, pos, y

    def ajustar(self, base: pd.DataFrame, mascara: np.ndarray, validacao: np.ndarray | None = None) -> "LSTM":
        """Com `validacao` (um fold do treino), para quando a perda deixa de cair. Sem ela, usa `epocas` fixas.

        A ordem das JANELAS dentro do treino é sorteada a cada época (só a ordem de apresentação);
        a ordem do tempo dentro de cada janela e entre treino e validação nunca muda.
        """
        self.prep.ajustar(base[mascara])
        x, pos, y = self._tensores(base)
        i_t = np.flatnonzero(mascara & (pos[:, 0] >= 0))
        i_v = None if validacao is None else np.flatnonzero(validacao & (pos[:, 0] >= 0))
        sementes = self.cfg["sementes"][:1] if i_v is not None else self.cfg["sementes"]
        treinadas = [self._treinar(x, pos, y, i_t, s, i_v) for s in sementes]
        self.redes, self.melhor_epoca = [r for r, _ in treinadas], int(np.median([e for _, e in treinadas]))
        return self

    def prever(self, base: pd.DataFrame) -> np.ndarray:
        x, pos, _ = self._tensores(base)
        p = np.full(len(base), np.nan)
        i = np.flatnonzero(pos[:, 0] >= 0)
        p[i] = self._inferir(self.redes, x, pos, i)
        return p
