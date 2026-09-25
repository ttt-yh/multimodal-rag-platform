"""公开仓库前静态审计：只报告文件名，不输出疑似密钥内容。"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = ("backend", "frontend/src", "scripts", "docs")
SCAN_FILES = ("README.md", ".env.example", "pyproject.toml")
TEXT_SUFFIXES = {".py", ".md", ".toml", ".json", ".jsonl", ".sql", ".ts", ".vue", ".css", ".txt"}
SECRET_PATTERNS = {
    "provider_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "workspace_id": re.compile(r"\bws-[a-z0-9]{12,}\b", re.IGNORECASE),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
ABSOLUTE_PATH = re.compile(r"(?:(?<![A-Za-z0-9_])[A-Za-z]:\\|/Users/|/home/)[^\s`'\"]+")


def candidates() -> list[Path]:
    result: list[Path] = []
    for relative in SCAN_ROOTS:
        base = ROOT / relative
        if not base.exists():
            continue
        result.extend(path for path in base.rglob("*")
                      if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
                      and "node_modules" not in path.parts and "__pycache__" not in path.parts)
    result.extend(ROOT / name for name in SCAN_FILES if (ROOT / name).is_file())
    return sorted(set(result))


def main() -> None:
    secret_files: dict[str, list[str]] = {name: [] for name in SECRET_PATTERNS}
    absolute_path_files: list[str] = []
    large_files: list[str] = []
    for path in candidates():
        relative = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > 5 * 1024 * 1024:
            large_files.append(relative)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                secret_files[name].append(relative)
        if (not relative.startswith("backend/tests/")
                and relative != "scripts/audit_public_release.py"
                and ABSOLUTE_PATH.search(content)):
            absolute_path_files.append(relative)

    # 测试中的固定假密钥不匹配上面的真实密钥长度，因此无需豁免真实命中。
    secret_files = {name: paths for name, paths in secret_files.items() if paths}
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    required_ignores = (".env", "data/raw/", "data/derived/", "evals/results/",
                        "storage/", "indexes/", "frontend/node_modules/")
    missing_ignores = [entry for entry in required_ignores if entry not in gitignore]
    warnings: list[str] = []
    if absolute_path_files:
        warnings.append("历史开发文档仍包含本机绝对路径，公开前可按需改为相对路径。")
    if not (ROOT / ".git").exists():
        warnings.append("当前目录尚未初始化为独立 Git 仓库。")
    failures: list[str] = []
    if secret_files:
        failures.append("发现疑似真实凭证或 Workspace ID。")
    if missing_ignores:
        failures.append(".gitignore 缺少必要规则。")
    if large_files:
        failures.append("源码公开范围存在超过 5 MiB 的文件。")

    report = {
        "status": "failed" if failures else "passed_with_warnings" if warnings else "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scanned_file_count": len(candidates()),
        "secret_candidate_files": secret_files,
        "large_files": large_files,
        "missing_gitignore_rules": missing_ignores,
        "absolute_path_files": absolute_path_files,
        "warnings": warnings,
        "failures": failures,
        "note": "报告只包含文件名，不包含任何疑似凭证值。",
    }
    output = ROOT / "evals" / "results" / "stage6b" / "public_release_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
