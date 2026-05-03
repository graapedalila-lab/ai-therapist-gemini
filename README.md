# 🧠 AI-терапевт — MVP на Google Gemini

Веб-приложение «AI-терапевт» с элементами психологической поддержки.
Использует **Google Gemini API** (бесплатный), **FAISS** (RAG) и **SQLite** (память).

> ⚠ Приложение носит образовательный характер и **не заменяет** консультацию специалиста.

---

## 📁 Структура проекта

```
ai_therapist/
├── app.py                  # Streamlit-интерфейс (точка входа)
├── config.py               # Все настройки + системный промпт + чтение секретов
├── gemini_client.py        # Обёртка над Google GenAI SDK
├── rag.py                  # Эмбеддинги + FAISS (поиск)
├── memory.py               # Долговременная память на SQLite
├── ingest.py               # Скрипт индексации .txt → FAISS
├── data/
│   └── knowledge_base/
│       ├── cbt_basics.txt
│       ├── automatic_thoughts.txt
│       └── relaxation_techniques.txt
├── storage/                # (создаётся автоматически)
├── .streamlit/
│   └── secrets.toml.example
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 🏛 Архитектура

```
   ┌─────────────┐
   │    User     │
   └──────┬──────┘
          │ (Safari на iPhone / браузер на ПК)
          ▼
   ┌─────────────────┐
   │  Streamlit UI   │  app.py
   └──────┬──────────┘
          ▼
   ┌─────────────────────────────────────────────────────┐
   │  Backend (Python)                                   │
   │  1) memory.add_message(...)         ← SQLite       │
   │  2) rag.search(query)               ← FAISS        │
   │  3) memory.get_recent_messages(...) ← SQLite       │
   │  4) build_messages()                                │
   │  5) gemini.chat()                   → Google AI    │
   │  6) memory.add_message(answer)      → SQLite       │
   └─────────────────────────────────────────────────────┘
```

---

## 🔑 Шаг 1. Получить бесплатный ключ Gemini

1. Открой <https://aistudio.google.com/apikey>
2. Войди через свой Google-аккаунт.
3. Нажми **«Create API key»** → **«Create API key in new project»**.
4. Скопируй ключ (начинается на `AIza...`).

**Бесплатный лимит:** ~1500 запросов в день — этого хватит на демо и защиту.
Карта **не нужна**.

---

## 📱 Шаг 2. Деплой на iPhone через Streamlit Cloud

### 2.1. Залей проект на GitHub

Создай **public** репозиторий и загрузи всё содержимое архива в его корень
(не во вложенную папку!). Файлы `.env` и `.streamlit/secrets.toml` сами не
попадут — `.gitignore` уже настроен.

### 2.2. Задеплой на Streamlit Cloud

1. Зайди на <https://share.streamlit.io>, авторизуйся через GitHub.
2. **Create app** → выбери репозиторий.
3. Заполни поля:
   - **Repository:** `твой_логин/имя_репозитория`
   - **Branch:** `main` (проверь название ветки на странице репо)
   - **Main file path:** `app.py` *(если файлы в корне; если в подпапке — `подпапка/app.py`)*
4. Нажми **Advanced settings** → в поле **Secrets** вставь:
   ```toml
   GEMINI_API_KEY = "AIza...твой_ключ"
   ```
   Кавычки обязательны.
5. Жми **Deploy**.

Сборка длится 3–5 минут. При первом открытии приложение ещё ~30 секунд
будет строить FAISS-индекс — это нормально.

### 2.3. Открой на iPhone

1. Скопируй URL вида `https://...streamlit.app`.
2. Открой в **Safari** (не в Chrome).
3. Кнопка **«Поделиться»** (квадрат со стрелкой) → **«На экран Домой»** —
   появится иконка-приложение.

---

## 💻 Локальный запуск (опционально)

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                # впиши GEMINI_API_KEY
python ingest.py                    # построить FAISS-индекс
streamlit run app.py
```

---

## 📚 Как добавить свои тексты в базу знаний

1. Положи `.txt` (UTF-8) в `data/knowledge_base/`.
2. **Локально:** `python ingest.py` → перезапусти Streamlit.
3. **На Streamlit Cloud:** закоммить файлы в GitHub → приложение
   автоматически перестроит индекс при старте.

---

## 🔐 Безопасность ключа

| Где запускается | Где лежит ключ | Как читается |
| --- | --- | --- |
| Локально | `.env` (gitignored) | `python-dotenv` → `os.getenv` |
| Streamlit Cloud | App → Settings → Secrets | `st.secrets["GEMINI_API_KEY"]` |

Логика чтения — в `config._get_secret()`: сначала пытаемся `st.secrets`,
при неудаче падаем на `os.getenv`. Один и тот же код работает в обоих режимах.

---

## ❓ Вопросы для защиты

| Вопрос | Где в коде ответ |
| --- | --- |
| Какая LLM используется? | `config.GEMINI_MODEL` (gemini-2.5-flash) |
| Почему Gemini, а не Claude/GPT? | Бесплатный API, мультиязычность, скорость |
| Как реализован RAG? | `rag.py` — `build_index`, `search` |
| Какие эмбеддинги? | `paraphrase-multilingual-MiniLM-L12-v2` (мультиязычная, локальная) |
| Зачем L2-нормализация? | Превращает скалярное произведение в `IndexFlatIP` в косинусное сходство |
| Как устроена память? | `memory.py` — таблица `messages` в SQLite, индекс `(user_id, created_at)` |
| Как формируется промпт? | `app.py` → `build_messages`: history + (RAG-context + user_input) |
| Почему RAG-контекст в user, а не в system? | Чтобы он обновлялся под каждый запрос, system оставался стабильным |
| Где системный промпт? | `config.SYSTEM_PROMPT` |
| Как защищён ключ? | `.env` локально / `st.secrets` в облаке, оба — в `.gitignore` |
| Как мигрировать на Claude/GPT? | Заменить только `gemini_client.py` — остальные модули не зависят от LLM |
| Как масштабировать на много документов? | Заменить `IndexFlatIP` на `IndexIVFFlat` или HNSW в `rag.py` |
