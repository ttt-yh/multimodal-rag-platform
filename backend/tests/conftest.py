"""小型夹具用于程序正确性，不属于模型评测数据或正式中文金标准。"""
import hashlib
import json
import socket

import pytest

from multimodal_rag.infrastructure.settings import Settings

SAMPLE = """---
title: 离线夹具
---
# 部署说明

先检查连接；错误码 `E_CONN`。

## 参数

| 名称 | 值 |
| --- | --- |
| timeout | 30s |

![架构图](/media/architecture.png)

```python
# 这不是标题
print("<unknown>")
```

说明含图片 ![示意](https://example.invalid/pic.png)。
"""


@pytest.fixture(autouse=True)
def isolated_offline_environment(monkeypatch):
    """清除个人配置并禁止外部连接；Windows asyncio 的 socketpair 需要回环。"""
    import os
    for name in list(os.environ):
        if name.startswith("MRAG_"):
            monkeypatch.delenv(name)
    original_connect = socket.socket.connect

    def no_connect(sock, address):
        if sock.family in {socket.AF_INET, socket.AF_INET6} and address[0] not in {"127.0.0.1", "::1"}:
            raise AssertionError("0A 离线测试禁止网络连接")
        return original_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", no_connect)


@pytest.fixture
def source_project(tmp_path):
    source = tmp_path / "data/raw/tidb_zh/source/sample.md"
    source.parent.mkdir(parents=True)
    source.write_text(SAMPLE, encoding="utf-8", newline="")
    entry = {"document_id": "doc_demo", "path": source.relative_to(tmp_path).as_posix(),
             "format": "md", "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
             "split": "dev", "title": "示例", "knowledge_base": "fixture",
             "source_family": "demo", "source_revision": "test", "license": "test-fixture"}
    manifest = tmp_path / "data/manifests/ingestion_chinese_md.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(entry), encoding="utf-8")
    return tmp_path, source, manifest, entry


@pytest.fixture
def settings(source_project):
    return Settings(_env_file=None, project_root=source_project[0])
