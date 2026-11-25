"""Seleção de hiperparâmetros, treino final, limiares e avaliação única no teste."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from src.config import RAIZ, parametros
from src.evaluation import custos, divisao, metricas
from src.features.ouro import NUMERICAS, TEXTO, base_modelagem
from src.models.classicos import Logistica, Majoritaria, RandomForest
from src.models.lstm import LSTM
from src.storage import controle, lake

ARQ_MODELOS = RAIZ / "models" / "modelos.joblib"
ARQ_RESULTADOS = RAIZ / "models" / "resultados.json"


def impressao_digital(base: pd.DataFrame) -> str:
    """Hash da base congelada e dos parâmetros. Se não mudou, não há por que treinar de novo."""
    h = hashlib.sha256(pd.util.hash_pandas_object(base[["ticker", "data", "alvo"] + NUMERICAS].round(5), index=False).values.tobytes())
    h.update(json.dumps({k: parametros()[k] for k in ("semente", "dados", "divisao", "lstm", "custos", "sentimento")}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def selecionar(classe, base: pd.DataFrame, treino: np.ndarray, variaveis: list[str]) -> tuple[dict, pd.DataFrame]:
    """Testa a grade com TimeSeriesSplit DENTRO do treino. Escolhe pela menor perda logarítmica média."""
    linhas = []
    for hp in classe.grade:
        for k, (a, v) in enumerate(divisao.folds(base, treino), 1):
            m = classe(variaveis, **hp)
            if classe is LSTM:
                m.ajustar(base, a, v)
            else:
                m.ajustar(base, a)
            p, y = m.prever(base)[v], base.alvo.to_numpy()[v]
            ok = ~np.isnan(p)
            linhas.append({**hp, "fold": k, "perda": log_loss(y[ok], p[ok]), "auc": roc_auc_score(y[ok], p[ok]),
                           "epoca": getattr(m, "melhor_epoca", 0)})
    r = pd.DataFrame(linhas)
    chaves = list(classe.grade[0])
    media = r.groupby(chaves)[["perda", "auc", "epoca"]].agg({"perda": "mean", "auc": "mean", "epoca": "median"}).reset_index()
    melhor = media.sort_values("perda").iloc[0]
    hp = {c: (int(melhor[c]) if float(melhor[c]).is_integer() and c != "dropout" else float(melhor[c])) for c in chaves}
    if classe is LSTM:
        hp["epocas"] = max(1, int(round(melhor.epoca)))
    return hp, r


def executar(avisar=print) -> dict:
    """Roda o experimento completo e grava modelos e resultados. O teste é usado uma única vez, no final."""
    base = base_modelagem()
    m = divisao.mascaras(base)
    y = base.alvo.to_numpy()
    matriz = custos.matriz_custos(base.ret_prox.to_numpy()[m["treino"]])
    sem_texto = [v for v in NUMERICAS if v not in TEXTO]
    planos = [("Random Forest", RandomForest, NUMERICAS), ("LSTM", LSTM, NUMERICAS),
              ("Random Forest sem texto", RandomForest, sem_texto), ("LSTM sem texto", LSTM, sem_texto)]
    modelos, res, cv = {}, {}, {}
    for nome, classe, vars_ in planos:
        if nome.endswith("sem texto"):
            hp = res[nome.replace(" sem texto", "")]["hiperparametros"]
        else:
            hp, tab = selecionar(classe, base, m["treino"], vars_)
            cv[nome] = tab.to_dict("records")
        avisar(f"{nome}: {hp}")
        modelos[nome] = classe(vars_, **hp).ajustar(base, m["treino"])
        res[nome] = {"hiperparametros": hp}
    modelos["Regressão logística"] = Logistica(NUMERICAS).ajustar(base, m["treino"])
    modelos["Classe majoritária"] = Majoritaria().ajustar(base, m["treino"])
    previsoes = base[["ticker", "data", "data_alvo", "alvo", "ret_prox", "ret_exec", "ret_intradia_prox"]].copy()
    for nome, mod in modelos.items():
        previsoes[nome] = mod.prever(base)
        pv, yv = previsoes[nome].to_numpy()[m["validacao"]], y[m["validacao"]]
        pt, yt = previsoes[nome].to_numpy()[m["teste"]], y[m["teste"]]
        fixo = nome == "Classe majoritária"
        limiar = 0.5 if fixo else custos.escolher_limiar(yv, pv, matriz)
        datas = base.data.to_numpy()[m["teste"]]
        r = res.setdefault(nome, {})
        r["limiar"] = limiar
        r["validacao_0.5"], r["validacao"] = metricas.classificar(yv, pv), metricas.classificar(yv, pv, limiar)
        r["teste_0.5"], r["teste"] = metricas.classificar(yt, pt), metricas.classificar(yt, pt, limiar)
        r["teste_0.5"]["custo"], r["teste"]["custo"] = (custos.custo_medio(yt, pt, l, matriz) for l in (0.5, limiar))
        r["validacao_0.5"]["custo"], r["validacao"]["custo"] = (custos.custo_medio(yv, pv, l, matriz) for l in (0.5, limiar))
        r["teste_0.5"].update(metricas.intervalos(datas, yt, pt, 0.5))
        r["simulacao"], r["sensibilidade"] = {}, {}
        bt = base[m["teste"]]
        for rot, l in (("limiar 0,5", 0.5), ("limiar ajustado", limiar)):
            r["simulacao"][rot] = custos.resumo(custos.simular(bt, pt > l))
        r["simulacao"]["fechamento (impossível)"] = custos.resumo(custos.simular(bt, pt > 0.5, "ret_prox"))
        sem_evento = ~bt.data.isin(pd.to_datetime(["2026-10-22", "2026-10-23"]))  # sinais cuja posição atravessa a abertura de 26/10
        r["simulacao"]["limiar 0,5 sem 26/10/2026"] = custos.resumo(custos.simular(bt[sem_evento], (pt > 0.5)[sem_evento.to_numpy()]))
        for bps in parametros()["custos"]["sensibilidade_bps"]:
            r["sensibilidade"][str(bps)] = custos.resumo(custos.simular(bt, pt > 0.5, custo=bps / 1e4))["retorno_aa_%"]
        avisar(f"{nome}: AUC validação {r['validacao']['auc']:.4f} limiar {limiar:.3f}")
    datas = base.data.to_numpy()[m["teste"]]
    yt = y[m["teste"]]
    col = lambda n: previsoes[n].to_numpy()[m["teste"]]
    comparacoes = {f"{a} x {b}": metricas.diferenca(datas, yt, col(a), col(b), 0.5, 0.5) for a, b in
                   [("LSTM", "Random Forest"), ("Random Forest", "Random Forest sem texto"), ("LSTM", "LSTM sem texto"),
                    ("Random Forest", "Regressão logística")]}
    bt = base[m["teste"]]
    referencias = {"Comprar e manter os 10 ativos": custos.resumo(custos.simular(bt, np.ones(len(bt), bool))),
                   "CDI": custos.resumo(custos.simular(bt, np.zeros(len(bt), bool)))}
    digital = impressao_digital(base)
    saida = {"digital": digital, "gerado_em": controle.agora(), "versao_ouro": lake.versao("gold", "base"), "matriz_custos": matriz,
             "limiar_teorico": matriz["FP"] / (matriz["FP"] + matriz["FN"]), "divisao": divisao.quadro(base).astype(str).to_dict("records"),
             "cv": cv, "modelos": res, "comparacoes": comparacoes, "referencias": referencias}
    ARQ_MODELOS.parent.mkdir(exist_ok=True)
    joblib.dump({"digital": digital, "modelos": modelos}, ARQ_MODELOS, compress=3)
    ARQ_RESULTADOS.write_text(json.dumps(saida, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    lake.sobrescrever("gold", "previsoes", previsoes)
    controle.registrar("treino", "modelos", len(previsoes), extra={"digital": digital})
    return saida


def carregar(estrito: bool = True) -> tuple[dict, dict] | None:
    """Modelos e resultados gravados, ou None.

    `estrito`: só devolve se a base congelada e os parâmetros são exatamente os do treino.
    O painel usa `estrito=False`, porque a coleta ao vivo pode corrigir dados antigos (reapresentações).
    """
    if not (ARQ_MODELOS.exists() and ARQ_RESULTADOS.exists()):
        return None
    pacote = joblib.load(ARQ_MODELOS)
    if estrito and pacote["digital"] != impressao_digital(base_modelagem()):
        return None
    return pacote["modelos"], json.loads(ARQ_RESULTADOS.read_text(encoding="utf-8"))
