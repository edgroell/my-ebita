# Module to manage a Chroma vector store using LangChain wrappers.

import os
from typing import List, Optional, Any, Dict
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
    def __init__(
        self, 
        collection_name: str = "transcripts",
        persist_directory: str = "data/chroma_langchain_db",
        api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Ensure API key is a string, not a callable
        openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if callable(openai_api_key):
            # If it's a callable, call it immediately to get the string
            openai_api_key = openai_api_key()
        
        # Create embeddings with explicit sync client
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            # Explicitly disable async
            async_client=None
        )
        
        # Initialize Chroma with the embeddings
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def _initialize_store(self) -> None:
        """Ensure the persist directory exists and that the Chroma store is available."""
        os.makedirs(self.persist_directory, exist_ok=True)

        if self.vectorstore is None:
            raise RuntimeError("Chroma or OpenAIEmbeddings is not available; cannot initialize vector store.")
        
        persist_fn = getattr(self.vectorstore, "persist", None)
        if callable(persist_fn):
            persist_fn()

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """
        Add texts with optional per-text metadata (e.g., transcript_id, chunk_index).
        """
        if self.vectorstore is None:
            raise RuntimeError("Chroma vector store is not initialized; cannot add texts.")
        return self.vectorstore.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 5, **kwargs) -> List[Any]:
        """Return top-k similar chunks for the query."""
        
        if self.vectorstore is None:
            raise RuntimeError("Chroma vector store is not initialized; ensure Chroma and embeddings were created successfully.")
        return self.vectorstore.similarity_search(query, k=k, **kwargs)

    def query(self, collection_name: Optional[str] = "transcripts", query: str = "", top_k: int = 5, **kwargs) -> Any:
        """
        Perform a similarity search and return results in a dict format.
        """
        results = self.similarity_search(collection=collection_name, query=query, k=top_k, **kwargs)
        return {"results": results}
