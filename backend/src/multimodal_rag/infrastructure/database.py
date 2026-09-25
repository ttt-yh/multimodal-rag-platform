"""显式数据库操作；从不在启动/健康接口中建表，也不允许连接旧工程库。

0B 使用有序 SQL + 校验和做小型迁移；后续复杂升级可迁移到 Alembic。
只有单库事务，不把它描述为向量/BM25 跨系统事务。
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path

import psycopg
from psycopg.conninfo import conninfo_to_dict

from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.settings import Settings

DATABASE_NAME = "multimodal_rag"
DATABASE_ROLE = "multimodal_rag_app"
MIGRATIONS = Path(__file__).with_name("migrations")


def checked_dsn(settings: Settings) -> str:
    dsn = settings.postgres_dsn.get_secret_value()
    if not dsn.strip():
        raise AppError("database_not_configured", "请先在本工程 .env 配置独立数据库连接", 503)
    try:
        params = conninfo_to_dict(dsn)
        valid = (params.get("dbname") == DATABASE_NAME and params.get("user") == DATABASE_ROLE
                 and params.get("host") in {"127.0.0.1", "localhost", "::1"}
                 and not params.get("service") and not params.get("options")
                 and params.get("hostaddr", "127.0.0.1") in {"127.0.0.1", "::1"})
    except (psycopg.Error, ValueError):
        valid = False
    if not valid:
        raise AppError("unsafe_database_target", "只允许本机 multimodal_rag 库与 multimodal_rag_app 账号", 403)
    return dsn


@contextmanager
def connection(settings: Settings, *, read_only=False):
    dsn = checked_dsn(settings)
    try:
        with psycopg.connect(dsn, connect_timeout=5, application_name="multimodal-rag",
                             options="-c statement_timeout=10000 -c lock_timeout=5000") as conn:
            if read_only:
                conn.execute("SET TRANSACTION READ ONLY")
            db, role = conn.execute("SELECT current_database(),current_user").fetchone()
            if (db, role) != (DATABASE_NAME, DATABASE_ROLE):
                raise AppError("unsafe_database_target", "实际连接身份与项目约束不符", 403)
            yield conn
    except psycopg.Error:
        # libpq 异常可能带 DSN、认证信息；不转发其原始文本。
        raise AppError("database_operation_failed", "数据库连接或SQL执行失败，请检查环境与迁移报告", 503) from None


def check_database(settings: Settings) -> dict:
    with connection(settings, read_only=True) as conn:
        row = conn.execute("SELECT current_database(),current_user,current_setting('server_version'),to_regclass('mrag.schema_migrations')").fetchone()
        migrations = conn.execute("SELECT name FROM mrag.schema_migrations ORDER BY name").fetchall() if row[3] else []
    return {"status": "connected", "database": row[0], "role": row[1], "server_version": row[2],
            "applied_migrations": [r[0] for r in migrations], "read_only_check": True}


def migrate(settings: Settings, directory: Path = MIGRATIONS) -> dict:
    applied, existing = [], []
    with connection(settings) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(826409240)")
        conn.execute("CREATE SCHEMA IF NOT EXISTS mrag")
        conn.execute("CREATE TABLE IF NOT EXISTS mrag.schema_migrations (name text PRIMARY KEY, sha256 text NOT NULL, applied_at timestamptz NOT NULL DEFAULT now())")
        for path in sorted(directory.glob("[0-9][0-9][0-9]_*.sql")):
            sql = path.read_text(encoding="utf-8")
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            row = conn.execute("SELECT sha256 FROM mrag.schema_migrations WHERE name=%s", (path.name,)).fetchone()
            if row:
                if row[0] != checksum:
                    raise AppError("migration_changed", "已执行的迁移文件被修改，拒绝继续；请新增迁移", 409)
                existing.append(path.name)
                continue
            conn.execute(sql, prepare=False)
            conn.execute("INSERT INTO mrag.schema_migrations(name,sha256) VALUES (%s,%s)", (path.name, checksum))
            applied.append(path.name)
    return {"status": "migrated", "applied": applied, "already_applied": existing}
