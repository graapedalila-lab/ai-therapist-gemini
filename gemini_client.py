"""
gemini_client.py
================
Тонкая обёртка над официальным SDK `google-genai`.

Зачем выделять отдельный класс:
  * вся работа с API сосредоточена в одном месте;
  * легко подменить модель или замокать клиент в тестах;
  * остальные модули (app.py) не зависят напрямую от SDK Google.

Особенности Gemini API (отличия от Anthropic Claude):
  * роли называются "user" и "model" (а не "assistant");
  * системный промпт передаётся через config.system_instruction,
    а не как отдельное поле в messages;
  * сообщения упаковываются в объекты Content с полем parts.
"""

from google import genai
from google.genai import types

import config


class GeminiClient:
    def __init__(self,
                 api_key: str = config.GEMINI_API_KEY,
                 model:   str = config.GEMINI_MODEL):
        if not api_key:
            raise RuntimeError(
                "Не найден GEMINI_API_KEY. "
                "Получить ключ можно бесплатно: https://aistudio.google.com/apikey\n"
                "Затем создайте .env по образцу .env.example "
                "(или добавьте секрет в Streamlit Cloud)."
            )
        # Клиент — единая точка входа в Gemini API.
        self.client = genai.Client(api_key=api_key)
        self.model  = model

    # -------------------------------------------------------------------
    # Преобразование истории в формат, который понимает Gemini SDK.
    #
    # Наш внутренний формат (как у OpenAI/Anthropic):
    #     [{"role": "user"|"assistant", "content": "..."}]
    #
    # Формат Gemini:
    #     [Content(role="user"|"model", parts=[Part(text="...")])]
    #
    # Поэтому "assistant" → "model".
    # -------------------------------------------------------------------
    @staticmethod
    def _to_gemini_contents(messages: list[dict]) -> list[types.Content]:
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=m["content"])],
                )
            )
        return contents

    def chat(self,
             system_prompt: str,
             messages:      list[dict],
             max_tokens:    int = 1024) -> str:
        """
        Отправляет запрос в Gemini и возвращает текст ответа.

        :param system_prompt: системные инструкции (роль ассистента).
        :param messages:      история диалога формата
                              [{"role": "user"|"assistant", "content": "..."}].
        :param max_tokens:    максимальная длина ответа (в токенах).
        """
        response = self.client.models.generate_content(
            model    = self.model,
            contents = self._to_gemini_contents(messages),
            config   = types.GenerateContentConfig(
                system_instruction = system_prompt,
                max_output_tokens  = max_tokens,
                # temperature чуть ниже дефолта = ответы стабильнее,
                # но всё ещё человечные.
                temperature        = 0.7,
            ),
        )
        # response.text — удобный шорткат, который Gemini SDK даёт
        # для извлечения объединённого текста из всех частей ответа.
        return response.text or ""
