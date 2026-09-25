"""用已安装的PostgreSQL二进制启动隔离测试实例；不连接用户现有5432实例。

临时目录、随机回环端口、随机SCRAM密码。执行bootstrap/迁移/事务/约束后停止。
保留测试数据目录用于诊断；随机密码文件在initdb后删除，密码不写入日志。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import tempfile

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from multimodal_rag.application.preview import preview_document
from multimodal_rag.core.errors import AppError
from multimodal_rag.core.quality import assess_parser_result
from multimodal_rag.infrastructure.database import connection, check_database, migrate, MIGRATIONS
from multimodal_rag.infrastructure.ingestion_repository import persist_preview
from multimodal_rag.infrastructure.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def run_fixture(pg_bin: Path, output: Path) -> dict:
    (ROOT / "tmp").mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pg-stage0b-", dir=ROOT / "tmp")).resolve()
    cluster = work / "cluster"
    password_file = work / "init-password"
    password = secrets.token_urlsafe(32)
    password_file.write_text(password, encoding="utf-8")
    checks = []
    started = False
    report = {"kind": "isolated_postgresql_integration", "status": "failed", "checks": checks,
              "target_is_user_database": False, "external_api_calls": 0, "workdir": work.relative_to(ROOT).as_posix()}
    env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    suffix = ".exe" if os.name == "nt" else ""
    def binary(name):
        return str(pg_bin / (name + suffix))
    def run(command, command_env=env):
        # Windows后台服务可能继承管道句柄；直接写日志避免capture_output等待EOF。
        with (output / "postgres-tools.log").open("ab") as log:
            completed = subprocess.run(command, cwd=ROOT, env=command_env, stdout=log, stderr=log,
                                       timeout=45, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if completed.returncode:
            raise RuntimeError("postgres_tool_failed")
    def passed(name):
        checks.append({"name": name, "status": "passed"})
    try:
        run([binary("initdb"), "-D", str(cluster), "-U", "postgres", "--auth=scram-sha-256",
             "--pwfile", str(password_file), "--encoding=UTF8", "--locale=C"])
        password_file.unlink()
        run([binary("pg_ctl"), "-D", str(cluster), "-l", str(output / "postgres-server.log"),
             "-o", f"-h 127.0.0.1 -p {port} -c shared_buffers=16MB -c max_connections=10", "-w", "-t", "20", "start"])
        started = True
        tool_env = dict(env, PGPASSWORD=password)
        bootstrap = [binary("psql"), "-w", "-h", "127.0.0.1", "-p", str(port), "-U", "postgres",
                     "-d", "postgres", "-f", str(ROOT / "scripts/bootstrap_database.sql")]
        run(bootstrap, tool_env)
        run(bootstrap, tool_env)
        passed("bootstrap_reentrant_dedicated_database")
        with psycopg.connect(host="127.0.0.1", port=port, dbname="postgres", user="postgres", password=password) as admin:
            admin.execute(sql.SQL("ALTER ROLE multimodal_rag_app PASSWORD {}").format(sql.Literal(password)))
            privileges = admin.execute("SELECT rolsuper,rolcreatedb,rolcreaterole,rolreplication FROM pg_roles WHERE rolname='multimodal_rag_app'").fetchone()
            assert privileges == (False, False, False, False)
        passed("application_role_has_no_cluster_admin_privileges")
        settings = Settings(_env_file=None, project_root=ROOT,
            postgres_dsn=f"host=127.0.0.1 port={port} dbname=multimodal_rag user=multimodal_rag_app password={password}")
        assert migrate(settings)["applied"] == ["001_ingestion.sql", "002_processing_versions.sql", "003_element_processing_identity.sql", "004_worker_leases.sql", "005_quality_reviews.sql", "006_chunks.sql", "007_index_versions.sql"]
        assert migrate(settings)["already_applied"] == ["001_ingestion.sql", "002_processing_versions.sql", "003_element_processing_identity.sql", "004_worker_leases.sql", "005_quality_reviews.sql", "006_chunks.sql", "007_index_versions.sql"]
        passed("migration_apply_and_idempotency")
        health = check_database(settings)
        assert health["applied_migrations"] == ["001_ingestion.sql", "002_processing_versions.sql", "003_element_processing_identity.sql", "004_worker_leases.sql", "005_quality_reviews.sql", "006_chunks.sql", "007_index_versions.sql"]
        report["server_version"] = health["server_version"]
        passed("readonly_connectivity_check")
        preview = preview_document("tidb85_b5ecd98ba8b7dd54", settings)
        doc, version, element = preview.document, preview.version, preview.elements[0]
        rows = [{'element': item.model_dump(), 'index_eligible': bool(item.raw_text.strip() or item.image_ref)}
                for item in preview.elements]
        assessment = assess_parser_result({'elements': rows,
            'warnings': [{'code': warning} for warning in preview.warnings],
            'artifact': {'missing_image_references': 0}})
        persisted = persist_preview(settings, preview, assessment, profile={'fixture': 'database', 'chunking': 'not_started'})
        repeated = persist_preview(settings, preview, assessment, profile={'fixture': 'database', 'chunking': 'not_started'})
        assert repeated.idempotent is True and repeated.processing_version_id == persisted.processing_version_id
        assert persisted.quality_status in {'candidate', 'needs_review'} and persisted.release_status == 'draft'
        with connection(settings) as conn:
            text, source = conn.execute("SELECT raw_text,source FROM mrag.elements WHERE element_id=%s", (element.element_id,)).fetchone()
            assert text == element.raw_text and source == element.source.model_dump()
            counts = conn.execute("SELECT count(*),count(DISTINCT processing_version_id) FROM mrag.elements WHERE version_id=%s", (version.version_id,)).fetchone()
            assert counts == (len(preview.elements), 1)
            job_count = conn.execute("SELECT count(*) FROM mrag.ingestion_jobs WHERE idempotency_key=%s", ('ingest:'+persisted.processing_version_id,)).fetchone()[0]
            assert job_count == 1
        passed("processing_version_and_element_json_roundtrip")
        changed = persist_preview(settings, preview, assessment, profile={'fixture': 'database', 'chunking': 'v2'})
        assert changed.processing_version_id != persisted.processing_version_id
        with connection(settings) as conn:
            assert conn.execute("SELECT count(*) FROM mrag.processing_versions WHERE version_id=%s", (version.version_id,)).fetchone()[0] == 2
        passed("processing_profile_change_isolated")
        with connection(settings) as conn:
            try:
                with conn.transaction():
                    conn.execute("INSERT INTO mrag.ingestion_jobs(job_id,document_id,version_id,operation,status,stage,idempotency_key) VALUES ('fixture-job-2',%s,%s,'ingest','pending','queued','fixture-key')", (doc.document_id,version.version_id))
            except psycopg.errors.UniqueViolation:
                passed("duplicate_job_idempotency_key_rejected")
            else:
                raise AssertionError("missing unique constraint")
            try:
                with conn.transaction():
                    conn.execute("INSERT INTO mrag.document_versions(version_id,document_id,content_sha256,source_revision) VALUES ('bad-version','missing-doc',%s,'x')", ("a"*64,))
            except psycopg.errors.ForeignKeyViolation:
                passed("orphan_version_rejected")
            else:
                raise AssertionError("missing foreign key")
        try:
            with connection(settings) as conn:
                conn.execute("UPDATE mrag.documents SET title='must_rollback' WHERE document_id=%s", (doc.document_id,))
                raise RuntimeError("intentional_rollback")
        except RuntimeError:
            pass
        with connection(settings) as conn:
            assert conn.execute("SELECT title FROM mrag.documents WHERE document_id=%s", (doc.document_id,)).fetchone()[0] == doc.title
        passed("transaction_failure_rolls_back")
        migration_copy = work / "migrations"
        migration_copy.mkdir()
        original = MIGRATIONS / "001_ingestion.sql"
        (migration_copy / original.name).write_bytes(original.read_bytes() + b"\n-- fixture alteration\n")
        try:
            migrate(settings, migration_copy)
        except AppError as exc:
            assert exc.code == "migration_changed"
        else:
            raise AssertionError("changed migration accepted")
        passed("migration_checksum_change_rejected")
        shutil.copyfile(original, migration_copy / original.name)
        (migration_copy / "002_failure.sql").write_text("CREATE TABLE mrag.must_rollback(id int); SELECT absent_column;", encoding="utf-8")
        try:
            migrate(settings, migration_copy)
        except AppError:
            pass
        else:
            raise AssertionError("invalid migration succeeded")
        with connection(settings) as conn:
            assert conn.execute("SELECT to_regclass('mrag.must_rollback')").fetchone()[0] is None
            assert conn.execute("SELECT count(*) FROM mrag.schema_migrations").fetchone()[0] == 7
        passed("failed_migration_is_atomic")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        if isinstance(exc, AppError):
            report["error_code"] = exc.code
        # 不将数据库异常/连接字符串或密码放进报告。
    finally:
        if password_file.exists():
            password_file.unlink()
        if started or (cluster / "postmaster.pid").exists():
            try:
                run([binary("pg_ctl"), "-D", str(cluster), "-m", "fast", "-w", "-t", "20", "stop"])
                report["server_stopped"] = True
            except Exception:
                report.update(status="failed", server_stopped=False)
        report["checks_passed"] = len(checks)
        (output / "database.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pg-bin", type=Path, default=Path(shutil.which("psql") or ".").parent)
    args = parser.parse_args()
    output = ROOT / "evals/results/database-fixture" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    result = run_fixture(args.pg_bin, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
