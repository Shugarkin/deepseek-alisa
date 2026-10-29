import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

SYSTEM_PROMPT = (
    "Ты — Иннокентий, голосовой поисковик внутри умной колонки. "
    "Отвечай кратко (1-3 предложения), простым разговорным языком, "
    "без markdown, без списков, без ссылок — только суть. "
    "Если тебя спрашивают, кто ты — отвечай, что ты Иннокентий, "
    "поисковый помощник. Не называй себя Алисой и не упоминай DeepSeek. "
    "Если в предоставленной информации нет ответа — честно скажи об этом."
)

EXIT_WORDS = {"хватит", "стоп", "пока", "выход", "закончить", "отмена"}

RESET_PHRASES = {
    "создай новый контекст",
    "новый контекст",
    "сбрось контекст",
    "сбросить контекст",
    "очисти контекст",
    "очистить контекст",
    "забудь всё",
    "забудь все",
    "начни заново",
    "начать заново",
    "очисти историю",
    "очистить историю",
}

# session_id -> list[{"role": ..., "content": ...}]
CONVERSATIONS: dict[str, list[dict]] = {}

MAX_HISTORY_MESSAGES = 10  # храним последние 10 реплик (5 пар вопрос-ответ)


def is_exit(text: str) -> bool:
    return text.strip().lower() in EXIT_WORDS


def is_reset(text: str) -> bool:
    return text.strip().lower() in RESET_PHRASES


def build_context(results: list) -> str:
    parts = []
    for r in results[:3]:
        summary = r.get("summary", "").strip()
        if summary:
            parts.append(f"- {r.get('title', '')}: {summary}")
    return "\n".join(parts)


def get_history(session_id: str) -> list[dict]:
    return CONVERSATIONS.get(session_id, [])


def trim_history(history: list[dict]) -> list[dict]:
    if len(history) > MAX_HISTORY_MESSAGES:
        return history[-MAX_HISTORY_MESSAGES:]
    return history


def save_history(session_id: str, history: list[dict]):
    CONVERSATIONS[session_id] = trim_history(history)


@app.post("/")
async def main(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "bad json"}, status_code=400)

    session = body.get("session", {})
    session_id = session.get("session_id", "default")
    version = body.get("version", "1.0")
    user_text = (body.get("request", {}).get("original_utterance") or "").strip()

    # Старт навыка
    if not user_text:
        return {
            "version": version,
            "session": session,
            "response": {
                "end_session": False,
                "text": "Привет! Меня зовут Иннокентий. Спроси что-нибудь, и я поищу в интернете.",
            },
        }

    # Выход
    if is_exit(user_text):
        CONVERSATIONS.pop(session_id, None)
        return {
            "version": version,
            "session": session,
            "response": {
                "end_session": True,
                "text": "Ок, до связи!",
            },
        }

    # Сброс контекста
    if is_reset(user_text):
        CONVERSATIONS.pop(session_id, None)
        return {
            "version": version,
            "session": session,
            "response": {
                "end_session": False,
                "text": "Готово, начал с чистого листа. О чём поговорим?",
            },
        }

    history = get_history(session_id)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(4.0, connect=2.0)) as client:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}

            # Шаг 1: поиск
            search_resp = await client.post(
                f"{DEEPSEEK_BASE}/v1/tools/search",
                headers=headers,
                json={"query": user_text, "count": 3, "need_content": False},
            )
            search_resp.raise_for_status()
            results = search_resp.json().get("results", [])
            context = build_context(results)

            if context:
                user_content = (
                    f"Вопрос пользователя: {user_text}\n\n"
                    f"Найденная информация из интернета:\n{context}\n\n"
                    f"Ответь на вопрос на основе этой информации."
                )
            else:
                user_content = (
                    f"Вопрос пользователя: {user_text}\n\n"
                    f"Поиск не дал результатов. Ответь, что не удалось найти информацию."
                )

            # Шаг 2: модель с учётом истории
            messages = (
                [{"role": "system", "content": SYSTEM_PROMPT}]
                + history
                + [{"role": "user", "content": user_content}]
            )

            chat_resp = await client.post(
                f"{DEEPSEEK_BASE}/v1/chat/completions",
                headers=headers,
                json={
                    "model": "deepseek-v4-flash",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
            )
            chat_resp.raise_for_status()
            answer = chat_resp.json()["choices"][0]["message"]["content"].strip()

            # Сохраняем историю: чистый вопрос пользователя + ответ модели
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": answer})
            save_history(session_id, history)

    except httpx.TimeoutException:
        answer = "Извини, поиск затянулся. Попробуй ещё раз."
    except Exception as e:
        print(f"DeepSeek error: {e}")
        answer = "Что-то пошло не так с поиском. Попробуй позже."

    if len(answer) > 900:
        answer = answer[:900].rsplit(" ", 1)[0] + "..."

    return {
        "version": version,
        "session": session,
        "response": {
            "end_session": False,
            "text": answer,
        },
    }