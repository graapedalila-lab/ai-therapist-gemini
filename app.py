"""
app.py
======
Streamlit-интерфейс приложения "AI-терапевт" на базе Google Gemini.

АРХИТЕКТУРА ОДНОГО ЗАПРОСА:

    User → Streamlit UI → Backend (Python)
                            │
                            ├── 1. memory.add_message()         → SQLite
                            ├── 2. rag.search(query)            → FAISS
                            ├── 3. memory.get_recent_messages() ← SQLite
                            ├── 4. build_messages(history + RAG-ctx + user)
                            ├── 5. gemini.chat()                → Google AI
                            └── 6. memory.add_message(answer)   → SQLite

ОСОБЕННОСТЬ ДЕПЛОЯ В ОБЛАКО:
  При первом запуске в облаке FAISS-индекса на диске НЕТ — `get_rag()`
  обнаруживает это и автоматически строит индекс из data/knowledge_base/.
  Дальше индекс кэшируется на время жизни инстанса (st.cache_resource).
"""

import streamlit as st

import config
from rag import RAGEngine
from memory import MemoryStore
from gemini_client import GeminiClient
from ingest import load_documents


# =========================================================================
#   Кэшируемые ресурсы
#   @st.cache_resource — Streamlit хранит объект между ререндерами,
#   чтобы НЕ перезагружать модель эмбеддингов при каждом сообщении.
# =========================================================================

@st.cache_resource(show_spinner="Подготовка базы знаний… (≈30 сек при первом запуске)")
def get_rag() -> RAGEngine:
    rag = RAGEngine()
    if rag.load():
        return rag

    # Индекса нет на диске — строим автоматически.
    # Это критично для облачного деплоя, где storage/ пустой при старте.
    docs = load_documents()
    if not docs:
        st.error(
            "⚠ В data/knowledge_base/ нет .txt файлов. "
            "Добавьте материалы и перезапустите приложение."
        )
        st.stop()
    rag.build_index(docs)
    rag.save()
    return rag


@st.cache_resource
def get_memory() -> MemoryStore:
    return MemoryStore()


@st.cache_resource
def get_llm() -> GeminiClient:
    return GeminiClient()


# =========================================================================
#   Сборка промпта
# =========================================================================

def build_messages(user_id:    str,
                   user_input: str,
                   memory:     MemoryStore,
                   rag:        RAGEngine) -> list[dict]:
    """
    Собирает список messages для LLM:
      1) Короткая память — последние N сообщений из SQLite.
      2) Текущий вопрос с прикреплённым контекстом из RAG.

    RAG-контекст вставляется ВНУТРЬ user-сообщения (а не в system),
    чтобы он обновлялся под каждый запрос, а системный промпт оставался
    стабильным.
    """
    history: list[dict] = memory.get_recent_messages(user_id)

    context_chunks = rag.search(user_input)
    context_block  = "\n\n---\n\n".join(context_chunks) if context_chunks else ""

    if context_block:
        augmented_user = (
            "[База знаний — используй при необходимости как справочный материал]\n"
            f"{context_block}\n\n"
            "[Сообщение пользователя]\n"
            f"{user_input}"
        )
    else:
        augmented_user = user_input

    return history + [{"role": "user", "content": augmented_user}]


# =========================================================================
#   UI
# =========================================================================

st.set_page_config(
    page_title="AI-терапевт",
    page_icon="🧠",
    layout="centered",
    # На мобильных боковая панель прячется, чтобы не мешать чату.
    initial_sidebar_state="collapsed",
)

st.title("🧠 AI-терапевт")
st.caption(
    "Эмпатичный ассистент с элементами когнитивно-поведенческой терапии. "
    "Не заменяет работу со специалистом."
)

# --- Сайдбар (на iPhone открывается тапом по «☰» в левом верхнем углу) ---
with st.sidebar:
    st.header("⚙ Настройки")
    user_id = st.text_input(
        "ID пользователя",
        value="default_user",
        help="Под этим ID хранится история диалога в SQLite.",
    )
    st.caption(f"Модель: `{config.GEMINI_MODEL}`")
    st.caption(f"Эмбеддинги: `{config.EMBEDDING_MODEL.split('/')[-1]}`")

    if st.button("🗑 Очистить историю", use_container_width=True):
        get_memory().clear_user(user_id)
        st.success("История очищена.")
        st.rerun()

    with st.expander("ℹ Как добавить свои тексты"):
        st.markdown(
            "1. Положите `.txt` файлы в `data/knowledge_base/`\n"
            "2. Локально: `python ingest.py` и перезапуск\n"
            "3. На Streamlit Cloud: закоммитьте файлы в репозиторий — "
            "индекс пересоберётся автоматически"
        )

# --- Инициализация ресурсов ---
rag    = get_rag()
memory = get_memory()
llm    = get_llm()

# --- Отрисовка истории из БД (даёт «продолжение разговора» после релоада) ---
for msg in memory.get_recent_messages(user_id, limit=100):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Поле ввода ---
if user_input := st.chat_input("Расскажи, что тебя беспокоит…"):

    with st.chat_message("user"):
        st.markdown(user_input)
    memory.add_message(user_id, "user", user_input)

    with st.chat_message("assistant"):
        with st.spinner("Думаю…"):
            try:
                messages = build_messages(user_id, user_input, memory, rag)
                answer = llm.chat(
                    system_prompt=config.SYSTEM_PROMPT,
                    messages=messages,
                )
            except Exception as e:
                answer = f"⚠ Ошибка при обращении к Gemini: `{e}`"
        st.markdown(answer)

    memory.add_message(user_id, "assistant", answer)
