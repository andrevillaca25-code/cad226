"""Gráficos Plotly compartilhados pelo notebook e pelo painel."""
import pandas as pd
import plotly.graph_objects as go

AZUL, LARANJA, CINZA, TINTA, GRADE = "#1c5cab", "#eb6834", "#898781", "#0b0b0b", "#e1e0d9"
CORES = {"Random Forest": AZUL, "LSTM": LARANJA}


def _layout(fig: go.Figure, titulo: str, y: str = "", altura: int = 340, pct: bool = False) -> go.Figure:
    fig.update_layout(title={"text": titulo, "font": {"size": 15}}, template="plotly_white", height=altura, hovermode="x unified", separators=",.",
                      margin={"l": 10, "r": 10, "t": 48, "b": 10}, legend={"orientation": "h", "y": -0.18}, font={"color": TINTA})
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7")
    fig.update_yaxes(gridcolor=GRADE, title=y, zerolinecolor="#c3c2b7", tickformat=".0%" if pct else None)
    return fig


def precos(p: pd.DataFrame, ticker: str) -> go.Figure:
    """Fechamento e médias móveis de 20 e 50 pregões."""
    f = go.Figure()
    f.add_scatter(x=p.data, y=p.fechamento, name="Fechamento", line={"color": AZUL, "width": 2}, hovertemplate="R$ %{y:.2f}")
    for n, cor, traco in ((20, "#6da7ec", "dot"), (50, CINZA, "dash")):
        f.add_scatter(x=p.data, y=p.fechamento.rolling(n).mean(), name=f"Média {n} pregões", line={"color": cor, "width": 1.5, "dash": traco}, hovertemplate="R$ %{y:.2f}")
    return _layout(f, f"{ticker}: preço de fechamento", "R$")


def acumulado(p: pd.DataFrame, ibov: pd.DataFrame, ticker: str) -> go.Figure:
    """Retorno acumulado com dividendos, contra o Ibovespa."""
    f = go.Figure()
    for d, nome, cor in ((p, ticker, AZUL), (ibov, "Ibovespa", CINZA)):
        f.add_scatter(x=d.data, y=d.indice_rt / d.indice_rt.iloc[0] - 1, name=nome, line={"color": cor, "width": 2}, hovertemplate="%{y:.1%}")
    return _layout(f, "Retorno acumulado no período (com dividendos)", pct=True)


def volatilidade(p: pd.DataFrame) -> go.Figure:
    """Volatilidade de 21 pregões, anualizada."""
    f = go.Figure()
    f.add_scatter(x=p.data, y=p.ret_log.rolling(21).std() * 252 ** 0.5, name="Volatilidade", line={"color": AZUL, "width": 2}, hovertemplate="%{y:.1%}", showlegend=False)
    return _layout(f, "Volatilidade de 21 pregões, anualizada", pct=True, altura=280)


def retornos(p: pd.DataFrame) -> go.Figure:
    """Distribuição dos retornos diários."""
    f = go.Figure(go.Histogram(x=p.ret, nbinsx=60, marker={"color": AZUL, "line": {"color": "white", "width": 1}}, hovertemplate="%{x:.1%}: %{y} pregões<extra></extra>"))
    _layout(f, "Distribuição dos retornos diários", "pregões", 280)
    f.update_xaxes(tickformat=".0%")
    f.update_layout(hovermode="closest")
    return f


def balancos(f_: pd.DataFrame, coluna: str, titulo: str) -> go.Figure:
    """Barras de um item do balanço ao longo do tempo (R$ bilhões)."""
    d = f_.dropna(subset=[coluna]).tail(16)
    f = go.Figure(go.Bar(x=d.periodo, y=d[coluna] / 1000, marker={"color": AZUL}, hovertemplate="R$ %{y:.1f} bi<extra></extra>"))
    return _layout(f, titulo, "R$ bilhões", 280)


def sentimento(base: pd.DataFrame) -> go.Figure:
    """Tom médio dos documentos dos últimos 21 pregões e número de documentos."""
    f = go.Figure()
    f.add_scatter(x=base.data, y=base.sent21, name="Tom médio", line={"color": AZUL, "width": 2}, hovertemplate="%{y:.2f}", showlegend=False)
    f.add_hline(y=0, line={"color": "#c3c2b7", "width": 1})
    return _layout(f, "Tom dos documentos, 21 pregões (-1 a +1)", altura=280)


def contagem_textos(base: pd.DataFrame) -> go.Figure:
    """Documentos por mês."""
    m = base.set_index("data").n_docs.resample("MS").sum()
    f = go.Figure(go.Bar(x=m.index, y=m.values, marker={"color": AZUL}, hovertemplate="%{x|%b %Y}: %{y:.0f}<extra></extra>"))
    f = _layout(f, "Fatos relevantes e comunicados por mês", "documentos", 280)
    return f.update_layout(hovermode="closest")


def probabilidades(prev: pd.DataFrame, limiares: dict) -> go.Figure:
    """Probabilidade de alta por modelo e o limiar de decisão de cada um."""
    f = go.Figure()
    for nome, cor in CORES.items():
        f.add_scatter(x=prev.data, y=prev[nome], name=nome, line={"color": cor, "width": 2}, hovertemplate="%{y:.1%}")
        f.add_hline(y=limiares[nome], line={"color": cor, "width": 1, "dash": "dot"})
    return _layout(f, "Probabilidade estimada de alta no pregão seguinte (pontilhado: limiar)", pct=True)


def confusao(m: dict, titulo: str) -> go.Figure:
    """Matriz de confusão."""
    z = [[m["VN"], m["FP"]], [m["FN"], m["VP"]]]
    f = go.Figure(go.Heatmap(z=z, x=["previu queda", "previu alta"], y=["caiu", "subiu"], text=z, texttemplate="%{text}", showscale=False,
                             colorscale=[[0, "#eaf2fd"], [1, "#86b6ef"]], hoverinfo="skip"))
    f.update_layout(title={"text": titulo, "font": {"size": 14}}, height=260, margin={"l": 10, "r": 10, "t": 44, "b": 10}, template="plotly_white")
    return f


def comparar(resultados: dict, metrica: str, titulo: str, referencia: float | None = None) -> go.Figure:
    """Métrica de teste com intervalo de 95% em todos os modelos."""
    nomes = [n for n in resultados["modelos"] if n != "Classe majoritária"][::-1]
    f = go.Figure()
    for n in nomes:
        r = resultados["modelos"][n]["teste_0.5"]
        v, (a, b) = r[metrica], r[f"{metrica}_ic"]
        cor = next((c for k, c in CORES.items() if n.startswith(k)), CINZA)
        f.add_scatter(x=[v], y=[n], mode="markers", marker={"color": cor, "size": 10}, showlegend=False,
                      error_x={"type": "data", "symmetric": False, "array": [b - v], "arrayminus": [v - a], "color": cor},
                      hovertemplate=f"%{{x:.3f}} (IC 95%: {a:.3f} a {b:.3f})<extra>{n}</extra>")
    if referencia is not None:
        f.add_vline(x=referencia, line={"color": CINZA, "dash": "dot"})
    f = _layout(f, titulo, altura=280)
    f.update_layout(hovermode="closest")
    f.update_xaxes(tickformat=".2f")
    return f
