"""
ingest.py
=========
CLI-скрипт: загружает все .txt из data/knowledge_base/ в FAISS-индекс
и сохраняет на диск (storage/faiss_index/).

Запускается ОДИН раз перед стартом приложения, а также после
добавления/удаления документов в базе знаний.

Использование:
    python ingest.py
"""

from rag import RAGEngine
import config


def load_documents() -> list[str]:
    """Читает все .txt-файлы из директории базы знаний."""
    docs: list[str] = []
    txt_files = sorted(config.DATA_DIR.glob("*.txt"))
    if not txt_files:
        print(f"⚠ В {config.DATA_DIR} нет .txt файлов. "
              "Положите туда хотя бы один документ и запустите ingest.py снова.")
        return docs
    for fp in txt_files:
        text = fp.read_text(encoding="utf-8")
        docs.append(text)
        print(f"  ✔ загружен {fp.name} ({len(text)} символов)")
    return docs


def main() -> None:
    print(f"Чтение документов из {config.DATA_DIR}…")
    docs = load_documents()
    if not docs:
        return

    print(f"Индексация ({len(docs)} документов)…")
    rag = RAGEngine()
    rag.build_index(docs)
    rag.save()

    print(f"✅ Готово. Чанков в индексе: {len(rag.chunks)}")
    print(f"   Сохранено в {config.FAISS_INDEX_PATH}")


if __name__ == "__main__":
    main()
