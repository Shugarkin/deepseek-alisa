import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

SYSTEM_PROMPT = (
    "Ты — голосовой поисковик внутри умной колонки Алиса. "
    "Отвечай кратко (1-3 предложения), простым разговорным языком, "
    "без markdown, без списков, без ссылок — только суть. "
    "Если в предоставленной информации нет ответа — честно скажи об этом."
)

EXIT_WORDS = {"хватит", "стоп", "пока", "выход", "закончить", "отмена"}


def is_exit(text: str) -> bool:
    return text.strip().lower() in EXIT_WORDS


def build_context(results: list) -> str:
    """Собирает summary из результатов поиска в текст для модели."""
    parts = []
    for r in results[:5]:  # берём не больше 5 результатов
        summary = r.get("summary", "").strip()
        if summary:
            parts.append(f"- {r.get('title', '')}: {summary}")
    return "\n".join(parts)


@app.post("/")
async def main(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "bad json"}, status_code=400)

    session = body.get("session", {})
    version = body.get("version", "1.0")
    user_text = (body.get("request", {}).get("original_utterance") or "").strip()

    # Пустой запрос = старт навыка
    if not user_text:
        return {
            "version": version,
            "session": session,
            "response": {
                "end_session": False,
                "text": "Привет! Спроси что-нибудь, и я поищу в интернете.",
            },
        }

    # Команда выхода
    if is_exit(user_text):
        return {
            "version": version,
            "session": session,
            "response": {
                "end_session": True,
                "text": "Ок, до связи!",
            },
        }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0)) as client:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}

            # Шаг 1: поиск в интернете через DeepSeek Search API
            search_resp = await client.post(
                f"{DEEPSEEK_BASE}/v1/tools/search",
                headers=headers,
                json={
                    "query": user_text,
                    "count": 5,
                    "need_content": False,
                },
            )
            search_resp.raise_for_status()
            results = search_resp.json().get("results", [])
            context = build_context(results)

            # Шаг 2: формируем ответ на основе найденного
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

            chat_resp = await client.post(
                f"{DEEPSEEK_BASE}/v1/chat/completions",
                headers=headers,
                json={
                    "model": "deepseek-v4-flash",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
            )
            chat_resp.raise_for_status()
            answer = chat_resp.json()["choices"][0]["message"]["content"].strip()

    except httpx.TimeoutException:
        answer = "Извини, поиск затянулся. Попробуй ещё раз."
    except Exception as e:
        print(f"DeepSeek error: {e}")
        answer = "Что-то пошло не так с поиском. Попробуй позже."

    # Обрезка под лимит Алисы (1024 символа на text)
    if len(answer) > 900:
        answer = answer[:900].rsplit(" ", 1)[0] + "..."

    return {
        "version": version,
        "session": session,
        "response": {
            "end_session": False,
            "text": f"Одну секунду, ищу информацию в интернете. {answer}",
        },
    }