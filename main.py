import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

SYSTEM_PROMPT = (
    "Ты — голосовой поисковик внутри умной колонки Алиса. "
    "Отвечай кратко (1-3 предложения), простым разговорным языком, "
    "без markdown, без списков, без ссылок — только суть. "
    "Если не знаешь ответа — честно скажи об этом."
)

EXIT_WORDS = {"хватит", "стоп", "пока", "выход", "закончить", "отмена"}


def is_exit(text: str) -> bool:
    return text.strip().lower() in EXIT_WORDS


@app.post("/")
async def main(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "bad json"}, status_code=400)

    session = body.get("session", {})
    version = body.get("version", "1.0")
    user_text = (body.get("request", {}).get("original_utterance") or "").strip()

    # Пустой запрос = старт навыка (Алиса так делает)
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

    # Запрос к DeepSeek
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            r = await client.post(
                DEEPSEEK_API_URL,
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                json={
                    "model": "deepseek-chat",  # см. примечание про модель
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
            )
            r.raise_for_status()
            answer = r.json()["choices"][0]["message"]["content"].strip()
    except httpx.TimeoutException:
        answer = "Извини, поиск затянулся. Попробуй ещё раз."
    except Exception as e:
        # логируем, но не падаем
        print(f"DeepSeek error: {e}")
        answer = "Что-то пошло не так с поиском. Попробуй позже."

    # Обрезаем под лимит Алисы (1024 символа на text)
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