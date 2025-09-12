# Module to manage a Chroma vector store using LangChain wrappers.

import os
from typing import List, Optional, Any
from pydantic import SecretStr

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class ChromaVectorStore:
    """
    Thin wrapper around LangChain's Chroma usage.
    - Initialize with a persist_directory
    - add_texts to append chunks
    - similarity_search to retrieve chunks
    """
    def __init__(self, collection_name: str = "transcripts", persist_directory: str = "data/chroma_langchain_db", api_key: Optional[str] = None, embedding_model: str = "text-embedding-3-small"):
        self.collection_name = collection_name
        self.persist_directory = os.path.abspath(persist_directory)
        self.api_key = api_key
        secret_api = SecretStr(self.api_key) if self.api_key is not None else None
        self._emb = OpenAIEmbeddings(api_key=secret_api, model=embedding_model) if OpenAIEmbeddings is not None else None
        self._store = Chroma(collection_name=collection_name, embedding_function=self._emb, persist_directory=self.persist_directory) if Chroma is not None else None
        self._initialize_store()

    def _initialize_store(self) -> None:
        """Ensure the persist directory exists and that the Chroma store is available."""
        os.makedirs(self.persist_directory, exist_ok=True)

        if self._store is None:
            raise RuntimeError("Chroma or OpenAIEmbeddings is not available; cannot initialize vector store.")
        
        persist_fn = getattr(self._store, "persist", None)
        if callable(persist_fn):
            persist_fn()

    def add_texts(self, texts: List[str]) -> List[str]:
        """Add texts to the collection."""
        if self._store is None:
            raise RuntimeError("Chroma vector store is not initialized; ensure Chroma and embeddings were created successfully.")
        return self._store.add_texts(texts=texts)

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Any]:
        """Return top-k similar chunks for the query."""
        
        if self._store is None:
            raise RuntimeError("Chroma vector store is not initialized; ensure Chroma and embeddings were created successfully.")
        return self._store.similarity_search(query, k=k, **kwargs)

    def query(self, collection_name: Optional[str] = "transcripts", query: str = "", top_k: int = 5, **kwargs) -> Any:
        """
        Perform a similarity search and return results in a dict format.
        """
        results = self.similarity_search(collection=collection_name, query=query, k=top_k, **kwargs)
        return {"results": results}
