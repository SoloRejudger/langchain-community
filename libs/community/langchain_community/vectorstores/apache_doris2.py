from __future__ import annotations

import json
import logging
import re
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain_community.vectorstores.utils import maximal_marginal_relevance

logger = logging.getLogger(__name__)

Metadata = Mapping[str, Union[str, int, float, bool]]


class ApacheDoris2Settings(BaseSettings):
    """Apache Doris settings for the ANN-based ApacheDoris2 vector store."""

    host: str = "localhost"
    port: int = 9030
    http_port: int = 8030
    username: str = "root"
    password: str = ""

    database: str = "default"
    table: str = "langchain"

    column_map: Dict[str, str] = {
        "id": "id",
        "document": "document",
        "embedding": "embedding",
        "metadata": "metadata",
    }

    create_index: bool = True
    index_type: str = "hnsw"
    metric_type: str = "l2_distance"
    dim: Optional[int] = None
    quantizer: Optional[str] = None
    pq_m: Optional[int] = None
    pq_nbits: Optional[int] = None
    max_degree: int = 32
    ef_construction: int = 40
    nlist: int = 1024

    num_buckets: Optional[int] = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="apache_doris2_",
        extra="ignore",
    )


class ApacheDoris2(VectorStore):
    """Apache Doris ANN-backed vector store built on top of ``doris_vector_search``.

    This class preserves the high-level LangChain VectorStore API, while using
    Doris ANN index DDL and approximate distance functions for retrieval.
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        config: Optional[ApacheDoris2Settings] = None,
        **kwargs: Any,
    ) -> None:
        """Create ApacheDoris2 with ANN-backed Doris search.

        Args:
            embedding: Embedding model.
            config: ApacheDoris2 settings.
            **kwargs: Reserved for API compatibility.
        """
        del kwargs
        super().__init__()

        self._embedding = embedding
        self.config = config or ApacheDoris2Settings()
        self._validate_config()

        self._metric_type = self._normalize_metric_type(self.config.metric_type)
        self._dim = self.config.dim or len(self._embedding.embed_query("test"))

        try:
            from doris_vector_search import (
                AuthOptions,
                DorisVectorClient,
                IndexOptions,
                LoadOptions,
                TableOptions,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import doris_vector_search package. "
                "Please install it with `pip install doris-vector-search`."
            ) from exc

        self._IndexOptions = IndexOptions
        self._LoadOptions = LoadOptions
        self._TableOptions = TableOptions

        auth = AuthOptions(
            host=self.config.host,
            query_port=self.config.port,
            http_port=self.config.http_port,
            user=self.config.username,
            password=self.config.password,
        )
        self._client = DorisVectorClient(
            database=self.config.database,
            auth_options=auth,
        )
        self._table = None

        self._ensure_table_exists()

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def metadata_column(self) -> str:
        return self.config.column_map["metadata"]

    def _validate_config(self) -> None:
        required = {"id", "document", "embedding", "metadata"}
        missing = required - set(self.config.column_map)
        if missing:
            raise ValueError(f"Missing required column_map keys: {sorted(missing)}")

    def _normalize_metric_type(self, metric_type: str) -> str:
        metric_alias = {
            "l2": "l2_distance",
            "euclidean": "l2_distance",
            "l2_distance": "l2_distance",
            "inner_product": "inner_product",
            "dot": "inner_product",
            "cosine": "inner_product",
            "angular": "inner_product",
        }
        normalized = metric_alias.get(metric_type.lower())
        if not normalized:
            supported = sorted(set(metric_alias.keys()))
            raise ValueError(
                f"Unsupported metric_type '{metric_type}'. Supported: {supported}"
            )
        return normalized

    def _build_index_options(self) -> Any:
        return self._IndexOptions(
            index_type=self.config.index_type,
            metric_type=self._metric_type,
            dim=self._dim,
            quantizer=self.config.quantizer,
            pq_m=self.config.pq_m,
            pq_nbits=self.config.pq_nbits,
            max_degree=self.config.max_degree,
            ef_construction=self.config.ef_construction,
            nlist=self.config.nlist,
        )

    def _resolve_num_buckets(self) -> int:
        if self.config.num_buckets is not None:
            return self.config.num_buckets
        return self._client._get_alive_be_count()

    def _table_exists(self) -> bool:
        cursor = self._client.connection.cursor()
        try:
            cursor.execute(f"SHOW TABLES LIKE '{self.config.table}'")
            return cursor.fetchone() is not None
        finally:
            cursor.close()

    def _table_has_ann_index(self) -> bool:
        cursor = self._client.connection.cursor()
        try:
            cursor.execute(f"SHOW CREATE TABLE `{self.config.table}`")
            row = cursor.fetchone()
            if not row:
                return False
            create_sql = row[-1]
            if isinstance(create_sql, (bytes, bytearray)):
                create_sql = create_sql.decode("utf-8")
            return "USING ANN" in str(create_sql).upper()
        finally:
            cursor.close()

    def _ensure_table_exists(self) -> None:
        if self._table_exists():
            self._table = self._client.open_table(self.config.table)
            if self.config.create_index and not self._table_has_ann_index():
                self._table.add_index(self._build_index_options())
            return

        columns = {
            self.config.column_map["id"]: "VARCHAR(64)",
            self.config.column_map["document"]: "TEXT",
            self.config.column_map["embedding"]: "ARRAY<FLOAT>",
            self.config.column_map["metadata"]: "TEXT",
        }
        vector_options = self._build_index_options() if self.config.create_index else None

        table_options = self._TableOptions(
            table_name=self.config.table,
            columns=columns,
            key_column=self.config.column_map["id"],
            vector_column=self.config.column_map["embedding"],
            vector_options=vector_options,
            num_buckets=self._resolve_num_buckets(),
        )

        ddl = self._client.ddl_compiler.compile_create_table(table_options)
        cursor = self._client.connection.cursor()
        try:
            cursor.execute(ddl)
        finally:
            cursor.close()

        self._table = self._client.open_table(self.config.table)

    def _get_table(self) -> Any:
        if self._table is None:
            self._table = self._client.open_table(self.config.table)
        return self._table

    def _parse_metadata(self, raw_metadata: Any) -> Dict[str, Any]:
        if raw_metadata is None:
            return {}
        if isinstance(raw_metadata, dict):
            return raw_metadata
        if isinstance(raw_metadata, (bytes, bytearray)):
            raw_metadata = raw_metadata.decode("utf-8")
        if isinstance(raw_metadata, str):
            raw_metadata = raw_metadata.strip()
            if not raw_metadata:
                return {}
            try:
                parsed = json.loads(raw_metadata)
                if isinstance(parsed, dict):
                    return parsed
                return {"value": parsed}
            except json.JSONDecodeError:
                return {"value": raw_metadata}
        return {"value": raw_metadata}

    def _apply_where(self, query: Any, where_str: Optional[str]) -> Any:
        if not where_str:
            return query

        conditions = [
            condition.strip()
            for condition in re.split(r"\s+AND\s+", where_str, flags=re.IGNORECASE)
            if condition.strip()
        ]
        for condition in conditions:
            query = query.where(condition)
        return query

    def _query_rows_by_vector(
        self,
        embedding: List[float],
        k: int,
        where_str: Optional[str],
        *,
        include_distance: bool,
        selected_columns: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        query = self._get_table().search(
            embedding,
            vector_column=self.config.column_map["embedding"],
            metric_type=self._metric_type,
            include_distance=include_distance,
        )
        query = self._apply_where(query, where_str)

        if selected_columns:
            query = query.select(selected_columns)

        return query.limit(k).to_list()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        batch_size: int = 32,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        del kwargs
        texts_list = list(texts)
        if not texts_list:
            return []

        if metadatas is not None and len(metadatas) != len(texts_list):
            raise ValueError(
                "The number of metadatas must match the number of texts. "
                f"Got {len(metadatas)} metadatas and {len(texts_list)} texts."
            )

        ids_list = list(ids) if ids is not None else [
            sha1(text.encode("utf-8")).hexdigest() for text in texts_list
        ]
        if len(ids_list) != len(texts_list):
            raise ValueError(
                "The number of ids must match the number of texts. "
                f"Got {len(ids_list)} ids and {len(texts_list)} texts."
            )

        vectors = self._embedding.embed_documents(texts_list)

        id_col = self.config.column_map["id"]
        document_col = self.config.column_map["document"]
        embedding_col = self.config.column_map["embedding"]
        metadata_col = self.config.column_map["metadata"]

        rows = []
        for idx, text in enumerate(texts_list):
            if len(vectors[idx]) != self._dim:
                raise ValueError(
                    "Embedding dim mismatch at row "
                    f"{idx}: expected {self._dim}, got {len(vectors[idx])}"
                )
            metadata = metadatas[idx] if metadatas is not None else {}
            rows.append(
                {
                    id_col: ids_list[idx],
                    document_col: text,
                    embedding_col: vectors[idx],
                    metadata_col: json.dumps(metadata),
                }
            )

        load_options = self._LoadOptions(format="arrow", batch_size=batch_size)
        self._get_table().add(rows, load_options=load_options)
        return ids_list

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[Any, Any]]] = None,
        *,
        ids: Optional[List[str]] = None,
        config: Optional[ApacheDoris2Settings] = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> "ApacheDoris2":
        ctx = cls(embedding=embedding, config=config, **kwargs)
        ctx.add_texts(texts, ids=ids, batch_size=batch_size, metadatas=metadatas)
        return ctx

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        del kwargs
        return self.similarity_search_by_vector(
            self._embedding.embed_query(query),
            k=k,
            where_str=where_str,
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Document]:
        del kwargs
        document_col = self.config.column_map["document"]
        metadata_col = self.config.column_map["metadata"]

        rows = self._query_rows_by_vector(
            embedding,
            k,
            where_str,
            include_distance=False,
            selected_columns=[document_col, metadata_col],
        )

        return [
            Document(
                page_content=row.get(document_col, ""),
                metadata=self._parse_metadata(row.get(metadata_col)),
            )
            for row in rows
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        del kwargs
        embedding = self._embedding.embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding,
            k=k,
            where_str=where_str,
        )

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        where_str: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        document_col = self.config.column_map["document"]
        metadata_col = self.config.column_map["metadata"]

        rows = self._query_rows_by_vector(
            embedding,
            k,
            where_str,
            include_distance=True,
            selected_columns=[document_col, metadata_col],
        )

        results = []
        for row in rows:
            results.append(
                (
                    Document(
                        page_content=row.get(document_col, ""),
                        metadata=self._parse_metadata(row.get(metadata_col)),
                    ),
                    float(row.get("distance", 0.0)),
                )
            )
        return results

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        where_str: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        del kwargs
        return self.similarity_search_with_score(query, k=k, where_str=where_str)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        del kwargs
        id_col = self.config.column_map["id"]
        document_col = self.config.column_map["document"]
        embedding_col = self.config.column_map["embedding"]
        metadata_col = self.config.column_map["metadata"]

        rows = self._query_rows_by_vector(
            embedding,
            fetch_k,
            None,
            include_distance=True,
            selected_columns=[id_col, document_col, metadata_col, embedding_col],
        )
        if not rows:
            return []

        candidate_embeddings = [row.get(embedding_col, []) for row in rows]
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            candidate_embeddings,
            k=k,
            lambda_mult=lambda_mult,
        )

        selected = []
        for idx, row in enumerate(rows):
            if idx in mmr_selected:
                selected.append(
                    Document(
                        page_content=row.get(document_col, ""),
                        metadata=self._parse_metadata(row.get(metadata_col)),
                    )
                )
        return selected

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        del filter
        del where_document
        del kwargs

        embedding = self._embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )

    def drop(self) -> None:
        self._client.drop_table(self.config.table)
        self._table = None

    def close(self) -> None:
        self._client.close()

    def __repr__(self) -> str:
        return (
            f"{self.config.database}.{self.config.table} @ "
            f"{self.config.host}:{self.config.port}"
        )
