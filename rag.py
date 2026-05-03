"""
rag.py
======
RAG-движок (Retrieval-Augmented Generation).

Логика:
    текст -> SentenceTransformer (эмбеддинг) -> вектор
                                                    |
                                                    v
                          FAISS (хранение + поиск ближайших соседей)
                                                    |
                                                    v
                                       top-k самых релевантных чанков

Используется в двух сценариях:
  * INGEST: build_index(...) + save(...)
  * RUNTIME (Streamlit): load(...) + search(query)

ВАЖНО: этот модуль НЕ зависит от выбранного LLM (Gemini/Claude/OpenAI).
RAG отвечает только за поиск релевантного контекста; генерацию делает
отдельный клиент (gemini_client.py).
"""

from pathlib import Path
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import config


class RAGEngine:
    """Минималистичный движок RAG поверх FAISS."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        # Модель грузится ОДИН раз при создании объекта.
        # Streamlit оборачивает создание в @st.cache_resource — см. app.py.
        # При первом запуске модель скачается с HuggingFace (~120 МБ).
        self.model = SentenceTransformer(model_name)
        self.index: faiss.Index | None = None
        # Параллельный список текстов чанков:
        # FAISS возвращает индексы — по ним достаём оригинальный текст.
        self.chunks: list[str] = []

    # -------------------------------------------------------------------
    # Чанкование текста
    # -------------------------------------------------------------------
    @staticmethod
    def chunk_text(text: str,
                   chunk_size: int = config.CHUNK_SIZE,
                   overlap:    int = config.CHUNK_OVERLAP) -> list[str]:
        """
        Делим длинный текст на пересекающиеся куски.
        Перекрытие нужно, чтобы важная мысль на границе чанка
        не оказалась "разрезанной" пополам.
        """
        text = text.strip()
        if not text:
            return []
        chunks, start = [], 0
        step = max(1, chunk_size - overlap)
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += step
        return chunks

    # -------------------------------------------------------------------
    # Построение индекса
    # -------------------------------------------------------------------
    def build_index(self, documents: list[str]) -> None:
        """Из списка документов строим FAISS-индекс."""
        # 1. Режем все документы на чанки
        all_chunks: list[str] = []
        for doc in documents:
            all_chunks.extend(self.chunk_text(doc))
        if not all_chunks:
            raise ValueError("Нет ни одного непустого чанка для индексации.")
        self.chunks = all_chunks

        # 2. Векторизуем чанки одной батчевой операцией (быстрее)
        embeddings = self.model.encode(
            all_chunks,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        # 3. Нормализуем векторы — это превращает скалярное произведение
        #    (IndexFlatIP) в косинусное сходство.
        faiss.normalize_L2(embeddings)

        # 4. Создаём точный (Flat) индекс. Для MVP его достаточно;
        #    для миллионов векторов имеет смысл IVF/HNSW.
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    # -------------------------------------------------------------------
    # Поиск
    # -------------------------------------------------------------------
    def search(self, query: str, top_k: int = config.TOP_K) -> list[str]:
        """Возвращаем top-k наиболее релевантных чанков для запроса."""
        if self.index is None or not self.chunks:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, top_k)
        return [self.chunks[i] for i in idxs[0] if 0 <= i < len(self.chunks)]

    # -------------------------------------------------------------------
    # Персистентность (сохранение / загрузка с диска)
    # -------------------------------------------------------------------
    def save(self, path: Path = config.FAISS_INDEX_PATH) -> None:
        """Сохраняем индекс и список чанков на диск."""
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self, path: Path = config.FAISS_INDEX_PATH) -> bool:
        """
        Загружаем ранее построенный индекс.
        Возвращаем False, если файлов нет (значит, надо запустить ingest.py).
        """
        idx_file    = path / "index.faiss"
        chunks_file = path / "chunks.pkl"
        if not idx_file.exists() or not chunks_file.exists():
            return False
        self.index = faiss.read_index(str(idx_file))
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)
        return True
