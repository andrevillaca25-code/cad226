"""Interface comum dos modelos e preparação das variáveis.

Para incluir um novo modelo basta criar uma classe com `ajustar` e `prever` e registrá-la em
src/models/catalogo.py.
"""
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class Modelo(Protocol):
    """Contrato mínimo. `base` é sempre a tabela inteira; a máscara diz quais linhas podem ser aprendidas."""
    nome: str

    def ajustar(self, base: pd.DataFrame, mascara: np.ndarray) -> "Modelo": ...

    def prever(self, base: pd.DataFrame) -> np.ndarray: ...


class Preparador:
    """Mediana e escala aprendidas SOMENTE nas linhas de treino; depois acrescenta o ativo em colunas 0/1."""

    def __init__(self, variaveis: list[str], escalar: bool = True):
        self.variaveis, self.escalar = variaveis, escalar

    def ajustar(self, treino: pd.DataFrame) -> "Preparador":
        self.tickers = sorted(treino.ticker.unique())
        self.imputador = SimpleImputer(strategy="median", keep_empty_features=True).fit(treino[self.variaveis])
        self.escala = StandardScaler().fit(self.imputador.transform(treino[self.variaveis]))
        return self

    def transformar(self, base: pd.DataFrame) -> np.ndarray:
        x = self.imputador.transform(base[self.variaveis])
        if self.escalar:
            x = np.clip(self.escala.transform(x), -5, 5)
        ativo = np.column_stack([(base.ticker == t).astype(float) for t in self.tickers])
        return np.hstack([x, ativo]).astype("float32")

    @property
    def nomes(self) -> list[str]:
        return self.variaveis + [f"ativo_{t}" for t in self.tickers]
