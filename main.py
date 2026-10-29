import os
import httpx
import time
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ========== Время и местоположение ==========
MSK = timezone(timedelta(hours=3))

def now_msk() -> datetime:
    return datetime.now(MSK)

def now_str() -> str:
    """Человекочитаемая дата и время для промпта."""
    n = now_msk()
    months = [
        "января", "февраля", "марта", "апреля", "мая", "июня",
        "июля", "августа", "сентября", "октября", "ноября", "декабря",
    ]
    days = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
    return f"{days[n.weekday()]}, {n.day} {months[n.month - 1]} {n.year} года, {n.strftime('%H:%M')} МСК"

# ========== Прямой запрос погоды (wttr.in) ==========
async def get_weather_yaroslavl(client: httpx.AsyncClient) -> str | None:
    """Берём实时 погоду в Ярославле у wttr.in, возвращаем строку для ответа."""
    try:
        r = await client.get(
            "https://wttr.in/Yaroslavl",
            params={"format": "j1"},
            timeout=2.5,  # оставляем 2.5 секунды, чтобы не задерживать общий ответ
        )
        r.raise_for_status()
        data = r.json()
        current = data["current_condition"][0]
        temp = current["temp_C"]
        feels = current["FeelsLikeC"]
        desc = current.get("weatherDesc", [{}])[0].get("value", "")
        weather_map = {
            "Sunny": "солнечно", "Clear": "ясно", "Partly cloudy": "переменная облачность",
            "Cloudy": "облачно", "Overcast": "пасмурно", "Light rain": "небольшой дождь",
            "Moderate rain": "умеренный дождь", "Light snow": "небольшой снег",
            "Mist": "дымка", "Fog": "туман"
        }
        desc_ru = weather_map.get(desc, desc)
        return f"Сейчас в Ярославле {temp}°C, {desc_ru}, ощущается как {feels}°C."
    except Exception as e:
        print(f"Ошибка получения погоды: {e}")
        return None

# ========== Ключевые слова ==========
WEATHER_KEYWORDS = {"погода", "погоду", "температура", "градус", "дождь", "снег", "ветер", "солнечно"}

def is_weather_question(text: str) -> bool:
    t = text.strip().lower()
    return any(k in t for k in WEATHER_KEYWORDS)

DATE_KEYWORDS = {
    "какое сегодня число", "какое число", "какой сегодня день",
    "какое сегодня", "сегодняшняя дата", "текущая дата",
    "который час", "сколько времени", "какое время",
    "какой день недели", "какой сейчас день",
}

def is_date_question(text: str) -> bool:
    t = text.strip().lower()
    return any(k in t for k in DATE_KEYWORDS)

# ========== Управление сессией и контекстом ==========
EXIT_WORDS = {"хватит", "стоп", "пока", "выход", "закончить", "отмена"}
RESET_PHRASES = {
    "создай новый контекст", "новый контекст", "сбрось контекст",
    "сбросить контекст", "очисти контекст", "очистить контекст",
    "забудь всё", "забудь все", "начни заново", "начать заново",
    "очисти историю", "очистить историю",
}

CONVERSATIONS: dict[str, list[dict]] = {}
MAX_HISTORY_MESSAGES = 10

def is_exit(text: str) -> bool:
    return text.strip().lower() in EXIT_WORDS

def is_reset(text: str) -> bool:
    return text.strip().lower() in RESET_PHRASES

def get_history(session_id: str) -> list[dict]:
    return CONVERSATIONS.get(session_id, [])

def trim_history(history: list[dict]) -> list[dict]:
    if len(history) > MAX_HISTORY_MESSAGES:
        return history[-MAX_HISTORY_MESSAGES:]
    return history

def save_history(session_id: str, history: list[dict]):
    CONVERSATIONS[session_id] = trim_history(history)

# ========== Кэш ==========
ANSWER_CACHE: dict[str, tuple[str, float]] = {}
CACHE_TTL = 180  # 3 минуты — чтобы спасти повторные запросы после таймаута

def get_cached(query: str) -> str | None:
    entry = ANSWER_CACHE.get(query.strip().lower())
    if entry and time.time() - entry[1] < CACHE_TTL:
        return entry[0]
    return None

def set_cached(query: str, answer: str):
    ANSWER_CACHE[query.strip().lower()] = (answer, time.time())

# ========== Системный промпт (динамический) ==========
def build_system_prompt() -> str:
    return (
        f"Ты — Иннокентий, голосовой поисковик внутри умной колонки. "
        f"Сейчас {now_str()}. "
        f"Пользователь находится в городе Ярославль. "
        "Если спрашивают о погоде, мероприятиях, пробках и других локальных вопросах — "
        "по умолчанию считай, что речь о Ярославле. "
        "Отвечай кратко (1-3 предложения), простым разговорным языком, "
        "без markdown, без списков, без ссылок — только суть. "
        "Если тебя спрашивают, кто ты — отвечай, что ты Иннокентий, "
        "поисковый помощник. Не называй себя Алисой и не упоминай DeepSeek. "
        "Если спрашивают текущую дату или время — используй данные выше. "
        "Если в предоставленной информации нет ответа — честно скажи об этом."
    )

# ========== Основной обработчик ==========
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

    # 1. Пустой запрос = приветствие
    if not user_text:
        return {
            "version": version,
            "session": session,
            "response": {
                "end_session": False,
                "text": "Привет! Меня зовут Иннокентий. Спроси что-нибудь, и я поищу в интернете.",
            },
        }

    # 2. Выход
    if is_exit(user_text):
        CONVERSATIONS.pop(session_id, None)
        return {
            "version": version,
            "session": session,
            "response": {"end_session": True, "text": "Ок, до связи!"},
        }

    # 3. Сброс контекста
    if is_reset(user_text):
        CONVERSATIONS.pop(session_id, None)
        return {
            "version": version,
            "session": session,
            "response": {"end_session": False, "text": "Готово, начал с чистого листа. О чём поговорим?"},
        }

    # 4. Вопрос о дате/времени — отвечаем сразу, без API
    if is_date_question(user_text):
        return {
            "version": version,
            "session": session,
            "response": {"end_session": False, "text": f"Сегодня {now_str()}."},
        }

    # 5. Проверка кэша
    cached = get_cached(user_text)
    if cached:
        return {
            "version": version,
            "session": session,
            "response": {"end_session": False, "text": cached},
        }

    history = get_history(session_id)
    answer = ""

    try:
        # Таймаут 3.5 секунды, чтобы уложиться в лимит Алисы (4.5 секунды)
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.5, connect=1.5)) as client:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}"}

            # 5.1 Вопрос о погоде — сначала прямой запрос к wttr.in
            if is_weather_question(user_text):
                weather = await get_weather_yaroslavl(client)
                if weather:
                    answer = weather
                    set_cached(user_text, answer)
                    return {
                        "version": version,
                        "session": session,
                        "response": {"end_session": False, "text": answer},
                    }
                # Если погода не получилась — идём в обычный поиск

            # 5.2 Обычный поиск
            search_resp = await client.post(
                f"{DEEPSEEK_BASE}/v1/tools/search",
                headers=headers,
                json={"query": user_text, "count": 3, "need_content": False},
            )
            search_resp.raise_for_status()
            results = search_resp.json().get("results", [])

            context_parts = []
            for r in results[:3]:
                summary = r.get("summary", "").strip()
                if summary:
                    context_parts.append(f"- {r.get('title', '')}: {summary}")
            context = "\n".join(context_parts)

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

            # 5.3 Запрос к модели с учётом истории
            messages = (
                [{"role": "system", "content": build_system_prompt()}]
                + history
                + [{"role": "user", "content": user_content}]
            )

            chat_resp = await client.post(
                f"{DEEPSEEK_BASE}/v1/chat/completions",
                headers=headers,
                json={
                    "model": "deepseek-flash",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 300,
                },
            )
            chat_resp.raise_for_status()
            answer = chat_resp.json()["choices"][0]["message"]["content"].strip()

            # Сохраняем историю
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": answer})
            save_history(session_id, history)

            # Пишем в кэш
            set_cached(user_text, answer)

    except httpx.TimeoutException:
        answer = "Извини, поиск затянулся. Попробуй ещё раз."
    except Exception as e:
        print(f"Ошибка DeepSeek: {e}")
        answer = "Что-то пошло не так с поиском. Попробуй позже."

    if len(answer) > 900:
        answer = answer[:900].rsplit(" ", 1)[0] + "..."

    return {
        "version": version,
        "session": session,
        "response": {"end_session": False, "text": answer},
    }