"""有预算的同步 HTTP 边界；只给幂等 GET 有限重试，POST/PUT 不盲目重发。

离线测试必须显式注入 MockTransport。真实调用要求 mode=api、开关和调用预算。
任何异常均丢弃供应商正文/URL，审计只保存安全字段，防止泄露签名链接和密钥。
"""
from dataclasses import dataclass, field
import json
import re
import threading
import time
from urllib.parse import urlsplit

import httpx2 as httpx

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.settings import Settings


@dataclass
class CallBudget:
    max_requests: int
    used: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        if not 1 <= self.max_requests <= 128:
            raise ValueError("调用预算必须为1～128次HTTP请求")

    def consume(self):
        with self._lock:
            if self.used >= self.max_requests:
                raise AppError("call_budget_exhausted", "本批调用次数预算已耗尽", 429)
            self.used += 1


def safe_id(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) else None


class HttpGateway:
    def __init__(self, settings: Settings, service: str, *, budget: CallBudget,
                 transport: httpx.MockTransport | None = None, sleep=time.sleep):
        if service not in {"chat", "vision", "embedding", "rerank", "parser"}:
            raise AppError("invalid_service", "不支持的服务类型")
        if transport is not None:
            if not isinstance(transport, httpx.MockTransport) or settings.mode != "mock":
                raise AppError("mock_mode_required", "Mock传输只允许在显式mock模式使用")
        elif settings.mode != "api" or not settings.api_enabled:
            raise AppError("live_api_disabled", "真实调用需要api模式和显式开启API开关", 403)
        key = settings.service_key(service).get_secret_value()
        base = settings.service_base_url(service)
        if not key.strip() or not base:
            raise AppError("service_not_configured", "服务密钥或地址未配置", 503)
        self.settings, self.service, self.budget = settings, service, budget
        self.base, self.key, self.sleep = base, key, sleep
        self.records: list[dict] = []
        proxy = settings.https_proxy or None
        self.client = httpx.Client(transport=transport, timeout=settings.api_timeout_seconds,
                                   follow_redirects=False, trust_env=False, proxy=proxy)

    def close(self):
        self.client.close()

    def _transfer_url(self, url: str):
        allowed = {h.strip().lower() for h in self.settings.parser_transfer_hosts.split(",") if h.strip()}
        try:
            if not isinstance(url, str):
                raise ValueError("URL must be a string")
            parsed = urlsplit(url)
            valid = (parsed.scheme == "https" and parsed.hostname in allowed and parsed.port in {None, 443}
                     and not parsed.username and not parsed.password and not parsed.fragment)
        except ValueError:
            valid = False
        if not valid:
            raise AppError("unsafe_transfer_url", "上传或下载地址不在已确认的HTTPS域名白名单内", 403)

    def request(self, method: str, path: str, *, payload=None, content: bytes | None = None,
                transfer_url: str | None = None, binary=False, max_bytes: int | None = None):
        if transfer_url is not None:
            if self.service != "parser" or method not in {"GET", "PUT"}:
                raise AppError("invalid_transfer", "资源传输仅供解析服务使用")
            self._transfer_url(transfer_url)
            url, headers = transfer_url, {}  # 绝不能把MinerU令牌带到签名OSS/CDN地址。
        else:
            url = self.base + path
            headers = {"Authorization": f"Bearer {self.key}"}
        limit = max_bytes or self.settings.api_max_response_bytes
        retries = self.settings.api_get_retries if method == "GET" else 0
        for attempt in range(retries + 1):
            self.budget.consume()
            start = time.monotonic()
            record = {"service": self.service, "method": method, "attempt": attempt + 1,
                      "transfer": transfer_url is not None, "status_code": None, "outcome": "unknown"}
            retryable = False
            try:
                with self.client.stream(method, url, headers=headers, json=payload, content=content) as response:
                    record["status_code"] = response.status_code
                    record["provider_request_id"] = safe_id(response.headers.get("x-request-id"))
                    if not 200 <= response.status_code < 300:
                        retryable = response.status_code in {429, 500, 502, 503, 504}
                        code = "authentication_failed" if response.status_code in {401, 403} else "provider_http_error"
                        raise AppError(code, "供应商返回失败状态；详见脱敏调用记录", 502)
                    chunks, length = [], 0
                    for chunk in response.iter_bytes():
                        length += len(chunk)
                        if length > limit:
                            raise AppError("response_too_large", "供应商响应超过安全大小限制", 502)
                        chunks.append(chunk)
                    body = b"".join(chunks)
                record["response_bytes"] = length
                if binary:
                    data = body
                else:
                    try:
                        data = json.loads(body)
                    except (ValueError, UnicodeError):
                        raise AppError("invalid_provider_json", "供应商未返回合法JSON", 502) from None
                    if not isinstance(data, dict):
                        raise AppError("invalid_provider_json", "供应商响应不是JSON对象", 502)
                    if data.get("error") or data.get("code") not in (None, 0, "0"):
                        raise AppError("provider_business_error", "供应商业务状态失败，HTTP成功不代表业务成功", 502)
                    record["provider_request_id"] = safe_id(data.get("request_id") or data.get("trace_id") or data.get("id")) or record["provider_request_id"]
                record["outcome"] = "http_success"
                return data
            except httpx.RequestError:
                retryable = True
                error = AppError("provider_transport_error", "连接或读取超时；非幂等请求不会自动重发", 502)
                record["outcome"] = error.code
            except AppError as exc:
                error = exc
                record["outcome"] = exc.code
            finally:
                record["duration_ms"] = round((time.monotonic() - start) * 1000, 2)
                self.records.append(record)
            if retryable and attempt < retries:
                self.sleep(min(0.25 * 2 ** attempt, 1.0))
            else:
                raise error from None
        raise AssertionError("unreachable")
