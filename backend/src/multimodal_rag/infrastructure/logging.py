"""请求日志采用字段白名单，不记录请求体、查询参数、认证头或异常原文。"""
import json
import logging
from datetime import datetime, timezone

LOGGER = logging.getLogger("multimodal_rag")
ALLOWED_FIELDS = {"request_id", "method", "route", "status", "duration_ms", "error_code"}


def configure_logging(level: str):
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False


def log_event(event: str, **fields):
    safe = {k: v for k, v in fields.items() if k in ALLOWED_FIELDS}
    LOGGER.info(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                            "event": event, **safe}, ensure_ascii=False))

