"""百炼独立职责适配器：使用HTTP契约，不把所有接口误当成Chat Completions。

0B只验证响应结构与用量，不代表模型回答正确或已经实现RAG。
"""
import base64
import math

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.http_gateway import HttpGateway


def nonempty(value):
    if not isinstance(value, str) or not value.strip():
        raise AppError("empty_input", "输入文本不能为空", 422)


def usage_of(data: dict) -> dict | None:
    source = data.get("usage")
    if not isinstance(source, dict):
        return None  # 缺失用量不是零用量，更不能推算费用。
    return {k: source[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")
            if type(source.get(k)) is int and source[k] >= 0}


def invalid(gateway: HttpGateway, data: dict | None = None):
    if gateway.records:
        gateway.records[-1]["outcome"] = "invalid_response_contract"
        if isinstance(data, dict):
            choices = data.get("choices")
            choice = choices[0] if isinstance(choices, list) and choices else None
            message = choice.get("message") if isinstance(choice, dict) else None
            # 只记录响应形状，不记录模型回答正文，便于定位供应商兼容差异。
            gateway.records[-1]["response_shape"] = {
                "top_keys": sorted(data.keys()),
                "choice_keys": sorted(choice.keys()) if isinstance(choice, dict) else None,
                "message_keys": sorted(message.keys()) if isinstance(message, dict) else None,
                "content_type": type(message.get("content")).__name__ if isinstance(message, dict) else None,
                "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            }
    raise AppError("invalid_response_contract", "响应不满足服务契约，不能作为成功结果", 502)


def successful(gateway: HttpGateway, data: dict):
    gateway.records[-1]["outcome"] = "validated"
    gateway.records[-1]["usage"] = usage_of(data)


class ChatAdapter:
    def __init__(self, gateway: HttpGateway):
        if gateway.service not in {"chat", "vision"}:
            raise ValueError("chat/vision gateway required")
        self.gateway = gateway

    def _complete(self, content):
        settings = self.gateway.settings
        model = settings.vision_model if self.gateway.service == "vision" else settings.chat_model
        data = self.gateway.request("POST", "/chat/completions", payload={
            "model": model, "messages": [{"role": "user", "content": content}],
            "stream": False, "enable_thinking": False, "temperature": 0,
            "max_tokens": settings.api_max_output_tokens})
        try:
            choice = data["choices"][0]
            text, finish = choice["message"]["content"], choice["finish_reason"]
            if not isinstance(text, str) or not text.strip() or finish != "stop":
                invalid(self.gateway, data)
        except (KeyError, IndexError, TypeError):
            invalid(self.gateway, data)
        successful(self.gateway, data)
        return {"model": model, "text": text, "finish_reason": finish, "usage": usage_of(data)}

    def complete(self, prompt: str):
        nonempty(prompt)
        return self._complete(prompt)


class VisionAdapter(ChatAdapter):
    def __init__(self, gateway: HttpGateway):
        if gateway.service != "vision":
            raise ValueError("vision gateway required")
        super().__init__(gateway)

    def describe(self, prompt: str, image: bytes, mime: str):
        return self.describe_many(prompt, [(image, mime)])

    def describe_many(self, prompt: str, images: list[tuple[bytes, str]]):
        nonempty(prompt)
        if not 1 <= len(images) <= 3:
            raise AppError("invalid_image_count", "单次视觉请求只允许1到3张图片", 422)
        content = [{"type": "text", "text": prompt}]
        for image, mime in images:
            if len(image) > self.gateway.settings.api_max_image_bytes:
                raise AppError("image_too_large", "图像超过本批验证大小限制", 413)
            # 这里只做文件签名检查；像素预算和图像解码属于独立解析管线。
            if not ((mime == "image/png" and image.startswith(b"\x89PNG\r\n\x1a\n")) or
                    (mime == "image/jpeg" and image.startswith(b"\xff\xd8\xff"))):
                raise AppError("invalid_image", "视觉接口只接受具有正确签名的PNG/JPEG", 422)
            uri = f"data:{mime};base64," + base64.b64encode(image).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": uri}})
        return self._complete(content)


class EmbeddingAdapter:
    def __init__(self, gateway: HttpGateway):
        if gateway.service != "embedding":
            raise ValueError("embedding gateway required")
        self.gateway = gateway

    def embed(self, texts: list[str]):
        if not texts or len(texts) > 10:
            raise AppError("invalid_batch", "0B向量验证每批仅允许1～10条文本", 422)
        for text in texts:
            nonempty(text)
        settings = self.gateway.settings
        data = self.gateway.request("POST", "/embeddings", payload={"model": settings.embedding_model,
            "input": texts, "encoding_format": "float", "dimensions": settings.embedding_dimensions})
        try:
            rows = data["data"]
            if not isinstance(rows, list) or len(rows) != len(texts):
                invalid(self.gateway)
            by_index = {}
            for row in rows:
                index, vector = row["index"], row["embedding"]
                if type(index) is not int or index in by_index or not 0 <= index < len(texts):
                    invalid(self.gateway)
                if not isinstance(vector, list) or len(vector) != settings.embedding_dimensions:
                    invalid(self.gateway)
                if not all(type(x) in {float, int} and math.isfinite(x) for x in vector) or not any(vector):
                    invalid(self.gateway)
                by_index[index] = vector
        except (KeyError, TypeError):
            invalid(self.gateway)
        successful(self.gateway, data)
        return {"model": settings.embedding_model, "dimensions": settings.embedding_dimensions,
                "vectors": [by_index[i] for i in range(len(texts))], "usage": usage_of(data)}


class RerankAdapter:
    def __init__(self, gateway: HttpGateway):
        if gateway.service != "rerank":
            raise ValueError("rerank gateway required")
        self.gateway = gateway

    def rerank(self, query: str, documents: list[str], top_n: int):
        nonempty(query)
        if not documents or not 1 <= top_n <= len(documents) <= 20:
            raise AppError("invalid_candidates", "0B重排验证只允许1～20条候选且top_n合法", 422)
        for document in documents:
            nonempty(document)
        model = self.gateway.settings.rerank_model
        data = self.gateway.request("POST", "/reranks", payload={"model": model, "query": query,
                                                                   "documents": documents, "top_n": top_n})
        try:
            rows = data["results"]  # qwen3-rerank为顶层results，不是旧接口output.results。
            if not isinstance(rows, list) or len(rows) != top_n:
                invalid(self.gateway)
            indices, results = set(), []
            for row in rows:
                index, score = row["index"], row["relevance_score"]
                if (type(index) is not int or not 0 <= index < len(documents) or index in indices or
                    type(score) not in {float, int} or not math.isfinite(score) or not 0 <= score <= 1):
                    invalid(self.gateway)
                indices.add(index)
                results.append({"index": index, "score": score})
            if any(a["score"] < b["score"] for a, b in zip(results, results[1:])):
                invalid(self.gateway)
        except (KeyError, TypeError):
            invalid(self.gateway)
        successful(self.gateway, data)
        return {"model": model, "results": results, "usage": usage_of(data)}
