# CAD226 - Data Lakehouse, Text Mining e Dashboard

Trabalho Final de Finanças Computacionais e Big Data (UFMG, 2026/2), Tema 1.
André Villaça, Lucas Soares e Matheus França.

O projeto coleta preços, balanços e documentos oficiais de 10 empresas da B3, organiza tudo em um Lakehouse (Bronze, Prata e Ouro, em Delta Lake), mede o tom dos textos com FinBERT-PT-BR e compara Random Forest e LSTM na previsão da direção do retorno do pregão seguinte.

## Entregas
| Entrega | Onde |
|---|---|
| Notebook no Colab | [abrir no Colab](https://colab.research.google.com/github/andrevillaca25-code/cad226/blob/main/notebooks/CAD226_Trabalho_Final.ipynb) |
| Relatório executivo | [`relatorio/CAD226_Relatorio_Executivo.pdf`](relatorio/CAD226_Relatorio_Executivo.pdf) |
| Apresentação | [`apresentacao/CAD226_Apresentacao.pdf`](apresentacao/CAD226_Apresentacao.pdf) (também em `.pptx`) |
| Painel interativo | [`dashboard/app.py`](dashboard/app.py), aberto pela seção 12 do notebook |

## Rodar no Colab
Abra o notebook no Colab e use *Ambiente de execução > Executar tudo*. Ele baixa este repositório, instala as dependências e refaz Prata, Ouro, modelos e avaliação a partir da camada Bronze, congelada em 20/11/2026 para que o resultado seja reproduzível. Nenhuma senha é necessária.

## Rodar no computador
```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/Mac: .venv/bin/pip
.venv/Scripts/python -m pytest -q                  # testes de integridade e de vazamento
.venv/Scripts/python -m streamlit run dashboard/app.py
.venv/Scripts/python -m src.pipeline.run PETR4     # coleta, transforma e prevê
```

## Estrutura
| Pasta | Conteúdo |
|---|---|
| `config/` | ativos e parâmetros (datas, custos, fontes) |
| `src/ingestion/` | coleta: yfinance, CVM (FCA, DFP, ITR, IPE), PDFs, RSS e CDI |
| `src/storage/` | Delta Lake e controle de atualizações |
| `src/processing/` | camada Prata |
| `src/nlp/` | sentimento |
| `src/features/` | camada Ouro: base integrada e alvo |
| `src/models/` | modelos e experimento |
| `src/evaluation/` | divisão temporal, métricas, custos e explicabilidade |
| `src/pipeline/` | `run_pipeline(ticker, period)` |
| `dashboard/` | painel Streamlit |
| `tests/` | testes de integridade, qualidade e vazamento |
| `data/`, `models/` | Lakehouse congelado e modelos treinados |

## Incluir uma empresa
Acrescente o ticker em `config/ativos.yaml` e rode `python -m src.pipeline.run TICKER --treinar`. CNPJ, código CVM e setor são descobertos na CVM.

## Fontes e limites
- Preços: yfinance (não oficial). Balanços e documentos: dados abertos da CVM. CDI: Banco Central. Notícias: RSS de InfoMoney, Money Times e Agência Brasil.
- O GDELT foi testado e descartado: 9 de 15 consultas recusadas, cobertura da Petrobras caindo de 262 para 24 artigos por dia entre 2017 e 2026, e títulos sem letras acentuadas.
- O Google Notícias não é usado, porque o robots.txt proíbe.
- Uso acadêmico. Não é recomendação de investimento.
