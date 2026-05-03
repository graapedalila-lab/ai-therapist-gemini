"""
config.py
=========
Единая точка конфигурации приложения "AI-терапевт".

Поддерживает два сценария хранения секретов:
  1) ЛОКАЛЬНО       — файл .env (через python-dotenv)
  2) STREAMLIT CLOUD — секреты, заданные в UI приложения (st.secrets)

Это позволяет использовать ОДИН и тот же код и на ноутбуке, и в облаке.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Локальный режим: подхватываем .env
load_dotenv()


def _get_secret(key: str, default: str = "") -> str:
    """
    Возвращает значение секрета.

    Порядок поиска:
      1. st.secrets[key] — если запущены под Streamlit Cloud;
      2. os.environ[key] — если задан в .env / переменных окружения;
      3. default.
    """
    # 1) Streamlit secrets (Streamlit Cloud)
    try:
        import streamlit as st  # noqa
        try:
            value = st.secrets.get(key) if hasattr(st, "secrets") else None
            if value:
                return value
        except Exception:
            pass
    except ImportError:
        pass

    # 2) Fallback — переменные окружения / .env
    return os.getenv(key, default)


# === Пути проекта ===
BASE_DIR         = Path(__file__).resolve().parent
DATA_DIR         = BASE_DIR / "data" / "knowledge_base"
STORAGE_DIR      = BASE_DIR / "storage"
FAISS_INDEX_PATH = STORAGE_DIR / "faiss_index"
DB_PATH          = STORAGE_DIR / "memory.db"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# === Google Gemini API ===
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")

# === Модели ===
# gemini-2.5-flash — быстрая и бесплатная модель Google.
# Альтернативы (есть в бесплатном тарифе):
#   "gemini-2.5-pro"   — мощнее, но медленнее и с меньшим RPD-лимитом
#   "gemini-2.0-flash" — стабильное предыдущее поколение
GEMINI_MODEL = "gemini-2.5-flash"

# Эмбеддинги: компактная мультиязычная модель (~120 МБ),
# хорошо работает с русским. Запускается локально, бесплатно.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# === Параметры RAG ===
CHUNK_SIZE    = 500   # размер чанка в символах
CHUNK_OVERLAP = 50    # перекрытие между чанками
TOP_K         = 3     # сколько релевантных фрагментов передавать в промпт

# === Параметры памяти ===
RECENT_MESSAGES_LIMIT = 10   # сколько последних сообщений тянуть в контекст

# === Системный промпт (роль ассистента) ===
SYSTEM_PROMPT = """Ты — эмпатичный психологический ассистент.
Ты не ставишь диагнозы.
Ты используешь техники когнитивно-поведенческой терапии (CBT).
Ты поддерживаешь пользователя, задаёшь уточняющие вопросы.
Если пользователь сообщает о серьёзном состоянии (мысли о суициде, самоповреждении,
насилии или сильном кризисе) — мягко рекомендуешь обратиться к специалисту
и приводишь телефон экстренной психологической помощи.

Правила работы:
1. Отвечай тепло, без осуждения, на языке пользователя.
2. Применяй техники КПТ: помогай выявить автоматические мысли,
   распознать когнитивные искажения, переформулировать убеждения.
3. Сначала задавай уточняющие вопросы — не давай советов вслепую.
4. Если в промпте есть блок "База знаний" — опирайся на него,
   это методические материалы по психологии.
5. Если есть блок "История диалога" — учитывай контекст прошлых сообщений.
6. Никогда не ставь диагнозы и не назначай медикаменты.
"""
