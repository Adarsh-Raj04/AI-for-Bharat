import json
from typing import List, Dict
from redis import Redis
from sqlalchemy.orm import Session

from app.models import Message

MAX_HISTORY = 20
REDIS_TTL_SECONDS = 60 * 60 * 24  # 24 hrs


def load_chat_history(
    redis_client: Redis,
    db: Session,
    session_id: int,
) -> List[Dict]:

    key = f"chat:session:{session_id}"

    # 1️⃣ Try Redis
    cached = redis_client.lrange(key, -MAX_HISTORY, -1)

    if cached:
        return [json.loads(m) for m in cached]

    # 2️⃣ DB fallback
    messages = (
        db.query(Message)
        .filter(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
        .limit(MAX_HISTORY)
        .all()
    )

    history = [{"role": m.role, "content": m.content} for m in messages]

    # Warm Redis cache
    for msg in history:
        redis_client.rpush(key, json.dumps(msg))

    redis_client.expire(key, REDIS_TTL_SECONDS)

    return history


def append_message_to_redis(
    redis_client: Redis,
    session_id: int,
    role: str,
    content: str,
):
    key = f"chat:session:{session_id}"

    redis_client.rpush(
        key,
        json.dumps({"role": role, "content": content}),
    )

    # keep only last N messages
    redis_client.ltrim(key, -MAX_HISTORY, -1)
    redis_client.expire(key, REDIS_TTL_SECONDS)
