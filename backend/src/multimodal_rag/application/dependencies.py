"""只检查是否填写配置；不发网络请求，不能将“已填写”当成“已连接”。"""
from multimodal_rag.infrastructure.settings import Settings


def dependency_report(settings: Settings) -> dict:
    names = ["postgres", "parser", "chat", "vision", "embedding", "rerank"]
    services = {}
    for name in names:
        key = settings.postgres_dsn if name == "postgres" else settings.service_key(name)
        configured = bool(key.get_secret_value().strip())
        address_set = True if name == "postgres" else bool(settings.service_base_url(name))
        services[name] = {
            "configured": configured,
            "address_configured": address_set,
            "adapter_implemented": True,
            "status": ("configured_not_checked" if address_set else "configuration_incomplete") if configured else "not_configured",
            "connectivity_checked": False,
        }
    return {"mode": settings.mode, "check_type": "configuration_only", "ready": False,
            "external_calls": 0, "services": services,
            "note": "本接口只检查配置，不发网络请求；完整问答仍需接入Chat和Context Builder。"}
