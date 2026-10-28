"""Wires the components together (used by the app, tests and the evaluation script)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from config import Settings

from .document_manager import DocumentManager
from .embeddings import Embedder, get_embedder
from .llm_service import GeminiService, LLMBackend
from .rag_pipeline import RAGPipeline
from .retriever import Retriever
from .study_tools import StudyTools
from .vector_store import VectorStore


@dataclass
class Services:
    settings: Settings
    store: VectorStore
    embedder: Embedder
    manager: DocumentManager
    retriever: Retriever
    pipeline: RAGPipeline
    study: StudyTools
    llm: LLMBackend


def build_services(
    settings: Settings, llm: Optional[LLMBackend] = None, embedder: Optional[Embedder] = None, store: Optional[VectorStore] = None
) -> Services:
    embedder = embedder or get_embedder(settings.embedding_backend, settings.embedding_model)
    store = store or VectorStore()
    llm = llm or GeminiService(settings)
    manager = DocumentManager(settings, store, embedder)
    retriever = Retriever(store, embedder, settings)
    return Services(
        settings=settings,
        store=store,
        embedder=embedder,
        manager=manager,
        retriever=retriever,
        pipeline=RAGPipeline(settings, retriever, llm),
        study=StudyTools(settings, store, llm),
        llm=llm,
    )
