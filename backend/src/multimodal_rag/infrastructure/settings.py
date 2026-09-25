"""统一配置；路径不随启动目录变化，测试可以显式注入临时项目根目录。"""
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MRAG_", env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8", extra="forbid", hide_input_in_errors=True,
    )
    project_root: Path = PROJECT_ROOT
    mode: Literal["offline", "mock", "api"] = "offline"
    host: str = "127.0.0.1"
    port: int = Field(default=8010, ge=1024, le=65535)
    log_level: Literal["INFO", "WARNING", "ERROR"] = "INFO"
    max_document_bytes: int = Field(default=2_097_152, ge=1, le=10_485_760)
    max_pdf_bytes: int = Field(default=20_971_520, ge=1_048_576, le=52_428_800)
    max_pdf_pages: int = Field(default=200, ge=1, le=500)
    # 入库白名单相对项目根目录配置；默认值保持现有 0A/阶段1行为。
    ingestion_manifest: str = "data/manifests/ingestion_chinese_md.jsonl"
    # 网页上传只登记到独立运行时清单，避免修改随代码发布的基准数据清单。
    runtime_upload_manifest: str = "data/manifests/runtime_uploads.jsonl"
    postgres_dsn: SecretStr = SecretStr("")
    parser_api_key: SecretStr = SecretStr("")
    chat_api_key: SecretStr = SecretStr("")
    vision_api_key: SecretStr = SecretStr("")
    embedding_api_key: SecretStr = SecretStr("")
    rerank_api_key: SecretStr = SecretStr("")
    # 共用百炼密钥可以只填这一项；单服务密钥优先，绝不读取旧工程配置。
    bailian_api_key: SecretStr = SecretStr("")
    bailian_workspace_id: str = Field(default="", pattern=r"^(|[A-Za-z0-9_-]{1,100})$")
    api_enabled: bool = False
    chat_model: str = "qwen-plus"
    vision_model: str = "qwen3-vl-plus"
    embedding_model: Literal["text-embedding-v4"] = "text-embedding-v4"
    rerank_model: Literal["qwen3-rerank"] = "qwen3-rerank"
    embedding_dimensions: int = Field(default=1024)
    chat_base_url: str = ""
    vision_base_url: str = ""
    embedding_base_url: str = ""
    rerank_base_url: str = ""
    parser_base_url: str = "https://mineru.net/api/v4"
    parser_model: Literal["pipeline", "vlm"] = "vlm"
    api_timeout_seconds: float = Field(default=45, gt=0, le=120)
    api_get_retries: int = Field(default=2, ge=0, le=2)
    api_max_output_tokens: int = Field(default=256, ge=1, le=1024)
    api_max_image_bytes: int = Field(default=2_097_152, ge=1, le=5_242_880)
    api_max_response_bytes: int = Field(default=8_388_608, ge=1024, le=67_108_864)
    parser_max_zip_bytes: int = Field(default=33_554_432, ge=1024, le=67_108_864)
    parser_max_unpacked_bytes: int = Field(default=67_108_864, ge=1024, le=134_217_728)
    parser_transfer_hosts: str = "mineru.oss-cn-shanghai.aliyuncs.com,cdn-mineru.openxlab.org.cn"
    https_proxy: str = ""

    @field_validator("https_proxy")
    @classmethod
    def valid_proxy(cls, value):
        if value:
            try:
                parsed = urlsplit(value)
                valid = (parsed.scheme in {"http", "https"} and bool(parsed.hostname)
                         and not parsed.username and not parsed.password
                         and not parsed.query and not parsed.fragment
                         and parsed.path in {"", "/"} and parsed.port is not None)
            except ValueError:
                valid = False
            if not valid:
                raise ValueError("代理地址必须是带协议、主机和端口的HTTP(S)地址，不能包含凭证或查询参数")
        return value

    @field_validator("embedding_dimensions")
    @classmethod
    def valid_dimensions(cls, value):
        allowed = {64, 128, 256, 512, 768, 1024, 1536, 2048}
        if value not in allowed:
            raise ValueError(f"embedding_dimensions 必须是 {sorted(allowed)} 之一")
        return value

    @field_validator("chat_base_url", "vision_base_url", "embedding_base_url", "rerank_base_url", "parser_base_url")
    @classmethod
    def https_endpoint(cls, value):
        if value:
            parsed = urlsplit(value)
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username or
                    parsed.password or parsed.query or parsed.fragment):
                raise ValueError("API 地址必须是无凭证、无查询串的 HTTPS 地址")
        return value.rstrip("/")

    def service_key(self, service: str) -> SecretStr:
        value = getattr(self, f"{service}_api_key")
        if value.get_secret_value().strip() or service == "parser":
            return value
        return self.bailian_api_key

    def service_base_url(self, service: str) -> str:
        explicit = getattr(self, f"{service}_base_url")
        if explicit:
            return explicit
        if not self.bailian_workspace_id:
            return ""
        suffix = "compatible-api/v1" if service == "rerank" else "compatible-mode/v1"
        return f"https://{self.bailian_workspace_id}.cn-beijing.maas.aliyuncs.com/{suffix}"

    @field_validator("host")
    @classmethod
    def local_host_only(cls, value):
        if value not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("当前仅允许本机监听；鉴权与公开部署尚未实现")
        return value


def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        # 不打印 ValidationError 的原始输入，未知配置值也可能含密钥。
        fields = sorted({str(e["loc"][0]) for e in exc.errors()})
        raise RuntimeError("配置校验失败，请检查 .env 字段及类型：" + ", ".join(fields)) from None
