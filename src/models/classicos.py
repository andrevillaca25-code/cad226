"""Modelos clássicos: Random Forest (principal), regressão logística e classe majoritária (linhas de base)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import parametros
from src.models.base import Preparador


class RandomForest:
    """Floresta aleatória. Média de muitas árvores rasas, cada uma vendo amostras e variáveis sorteadas."""
    nome = "Random Forest"
    grade = [{"max_depth": d, "min_samples_leaf": f} for d in (3, 6, 10) for f in (50, 200)]

    def __init__(self, variaveis: list[str], max_depth: int = 6, min_samples_leaf: int = 100, n_estimators: int = 400):
        self.prep = Preparador(variaveis, escalar=False)
        self.modelo = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                             max_features="sqrt", n_jobs=-1, random_state=parametros()["semente"])

    def ajustar(self, base: pd.DataFrame, mascara: np.ndarray) -> "RandomForest":
        t = base[mascara]
        self.modelo.fit(self.prep.ajustar(t).transformar(t), t.alvo.astype(int))
        return self

    def prever(self, base: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict_proba(self.prep.transformar(base))[:, 1]


class Logistica:
    """Regressão logística com penalização L2: o modelo mais simples que usa as mesmas variáveis."""
    nome = "Regressão logística"

    def __init__(self, variaveis: list[str], C: float = 0.01):
        self.prep, self.modelo = Preparador(variaveis), LogisticRegression(C=C, max_iter=2000)

    def ajustar(self, base: pd.DataFrame, mascara: np.ndarray) -> "Logistica":
        t = base[mascara]
        self.modelo.fit(self.prep.ajustar(t).transformar(t), t.alvo.astype(int))
        return self

    def prever(self, base: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict_proba(self.prep.transformar(base))[:, 1]


class Majoritaria:
    """Sempre a proporção de altas do treino. Quem não a supera não tem poder preditivo."""
    nome = "Classe majoritária"

    def ajustar(self, base: pd.DataFrame, mascara: np.ndarray) -> "Majoritaria":
        self.p = float(base.loc[mascara, "alvo"].mean())
        return self

    def prever(self, base: pd.DataFrame) -> np.ndarray:
        return np.full(len(base), self.p)
