"""Camadas Bronze, Prata e Ouro em Delta Lake (Parquet + log de transações ACID)."""
import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

from src.config import RAIZ

CAMADAS = {"bronze", "silver", "gold"}


def caminho(camada: str, tabela: str) -> str:
    """Caminho físico de uma tabela do lake."""
    if camada not in CAMADAS:
        raise ValueError(f"camada desconhecida: {camada}")
    return str(RAIZ / "data" / camada / tabela)


def existe(camada: str, tabela: str) -> bool:
    """Informa se a tabela já foi criada."""
    return DeltaTable.is_deltatable(caminho(camada, tabela))


def ler(camada: str, tabela: str, versao: int | None = None) -> pd.DataFrame:
    """Lê a tabela inteira; `versao` recupera um estado antigo (time travel)."""
    return DeltaTable(caminho(camada, tabela), version=versao).to_pandas()


def versao(camada: str, tabela: str) -> int:
    """Número da versão atual da tabela."""
    return DeltaTable(caminho(camada, tabela)).version()


def historico(camada: str, tabela: str) -> pd.DataFrame:
    """Histórico de transações da tabela, usado como trilha de auditoria."""
    return pd.DataFrame(DeltaTable(caminho(camada, tabela)).history())


def _arrow(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(df, preserve_index=False).replace_schema_metadata(None)


def sobrescrever(camada: str, tabela: str, df: pd.DataFrame) -> int:
    """Grava uma nova versão completa (a versão anterior continua acessível)."""
    write_deltalake(caminho(camada, tabela), _arrow(df), mode="overwrite", schema_mode="overwrite")
    return len(df)


def upsert(camada: str, tabela: str, df: pd.DataFrame, chaves: list[str]) -> dict:
    """Insere linhas novas e atualiza as que já existem, segundo `chaves`.

    Repetir a mesma carga não duplica nada (idempotência). Linhas cujo conteúdo é idêntico
    ao já gravado são ignoradas e não geram nova versão da tabela.
    """
    df = df.drop_duplicates(chaves, keep="last").reset_index(drop=True)
    if df.empty:
        return {"novos": 0, "atualizados": 0}
    if not existe(camada, tabela):
        write_deltalake(caminho(camada, tabela), _arrow(df))
        return {"novos": len(df), "atualizados": 0}
    atual = ler(camada, tabela)
    comuns = [c for c in df.columns if c in atual.columns and c != "coletado_em"]
    junto = df[comuns].merge(atual[comuns].drop_duplicates(), how="left", indicator=True)
    mudou = df[(junto["_merge"] == "left_only").to_numpy()]
    if mudou.empty:
        return {"novos": 0, "atualizados": 0}
    condicao = " and ".join(f't."{c}" = s."{c}"' for c in chaves)
    r = (DeltaTable(caminho(camada, tabela))
         .merge(_arrow(mudou), predicate=condicao, source_alias="s", target_alias="t")
         .when_matched_update_all().when_not_matched_insert_all().execute())
    return {"novos": r["num_target_rows_inserted"], "atualizados": r["num_target_rows_updated"]}
