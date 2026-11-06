# Module to manage a Chroma vector store using LangChain wrappers.

import os
from pathlib import Path
from typing import List, Optional, Any, Dict

import httpx
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from pydantic import SecretStr
from tokenizers import Tokenizer


MAX_EMBEDDING_CHUNK_TOKENS = 6000


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
        embedding_model: str = "mistral-embed"
    ):
        self.collection_name = collection_name
        project_root = Path(__file__).resolve().parents[1]
        persist_path = Path(persist_directory)
        if not persist_path.is_absolute():
            persist_path = project_root / persist_path
        self.persist_directory = str(persist_path)
        
        mistral_api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not mistral_api_key:
            raise ValueError(
                "Mistral API key is required for embeddings. "
                "Set MISTRAL_API_KEY or pass api_key."
            )
        
        self.embeddings = MistralAIEmbeddings(
            model=embedding_model,
            api_key=SecretStr(mistral_api_key),
        )
        
        # Initialize Chroma with the embeddings
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def _split_text(self, text: str) -> List[str]:
        """Split oversized text into chunks below Mistral's per-input limit."""
        tokenizer = self.embeddings.tokenizer
        if isinstance(tokenizer, Tokenizer):
            token_ids = tokenizer.encode(text).ids
            return [
                tokenizer.decode(
                    token_ids[start:start + MAX_EMBEDDING_CHUNK_TOKENS],
                    skip_special_tokens=True,
                )
                for start in range(0, len(token_ids), MAX_EMBEDDING_CHUNK_TOKENS)
            ] or [text]

        # LangChain uses a character-based fallback if the model tokenizer
        # cannot be loaded; keep chunks conservative in that case as well.
        return [
            text[start:start + MAX_EMBEDDING_CHUNK_TOKENS]
            for start in range(0, len(text), MAX_EMBEDDING_CHUNK_TOKENS)
        ] or [text]

    def _initialize_store(self) -> None:
        """Ensure the persist directory exists and that the Chroma store is available."""
        os.makedirs(self.persist_directory, exist_ok=True)

        if self.vectorstore is None:
            raise RuntimeError("Chroma or MistralAIEmbeddings is not available; cannot initialize vector store.")
        
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
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("The number of metadata entries must match the number of texts.")

        chunked_texts: List[str] = []
        chunked_metadatas: Optional[List[Dict[str, Any]]] = [] if metadatas is not None else None
        for index, text in enumerate(texts):
            chunks = self._split_text(text)
            chunked_texts.extend(chunks)
            if chunked_metadatas is not None and metadatas is not None:
                chunked_metadatas.extend([metadatas[index]] * len(chunks))

        try:
            return self.vectorstore.add_texts(
                texts=chunked_texts,
                metadatas=chunked_metadatas,
            )
        except httpx.HTTPStatusError as error:
            response_body = error.response.text
            raise RuntimeError(
                f"Mistral embeddings request failed with HTTP "
                f"{error.response.status_code}: {response_body}"
            ) from error

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
