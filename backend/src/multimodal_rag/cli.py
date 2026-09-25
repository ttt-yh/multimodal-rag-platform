"""在线检查和数据库写入均为独立、显式的命令，不在启动时自动执行。"""
import argparse
import json
from datetime import datetime, timezone

from multimodal_rag.application.dependencies import dependency_report
from multimodal_rag.application.preview import preview_document
from multimodal_rag.core.errors import AppError
from multimodal_rag.infrastructure.settings import load_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="企业多模态 RAG：文档入库、混合检索与可信图文问答")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("serve", help="在本机启动 API /docs 文档页")
    check = commands.add_parser("check-env", help="仅检查配置，不连接外部服务")
    check.add_argument("--real", action="store_true", help="请改用独立db-check/check-service命令")
    commands.add_parser("db-check", help="显式只读检查项目数据库")
    db = commands.add_parser("db-migrate", help="对独立项目库执行版本化迁移")
    db.add_argument("--confirm", action="store_true", help="确认对本项目独立数据库建表")
    service = commands.add_parser("check-service", help="固定样例的真实API契约检查；会消耗额度")
    service.add_argument("--service", required=True, choices=["chat", "vision", "embedding", "rerank", "parser"])
    service.add_argument("--confirm-live", action="store_true", help="明确确认本批真实调用")
    service.add_argument("--max-requests", type=int, required=True, choices=range(1,129), metavar="1..128")
    service.add_argument("--task-id", help="恢复MinerU任务；提供后不会重新提交")
    demo = commands.add_parser("preview", help="读取开发白名单文档并输出元素 JSON")
    demo.add_argument("--document-id", default="tidb85_b5ecd98ba8b7dd54")
    search = commands.add_parser("retrieve", help="对已激活索引执行一次受控混合检索")
    search.add_argument("--query", required=True)
    search.add_argument("--confirm-live", action="store_true", help="确认消耗1次Embedding和最多1次Rerank调用")
    search.add_argument("--max-requests", type=int, default=2, choices=(1, 2))
    qa = commands.add_parser("qa", help="基于已激活索引生成一次带证据引用的RAG回答")
    qa.add_argument("--query", required=True)
    qa.add_argument("--confirm-live", action="store_true", help="确认消耗1次Embedding、1次Rerank和1次Chat调用")
    args = parser.parse_args()
    try:
        settings = load_settings()
        if args.command == "serve":
            import uvicorn
            from multimodal_rag.api.app import create_app
            # 禁用通用 access log，避免记录 URL 查询串中的敏感内容。
            uvicorn.run(create_app(settings), host=settings.host, port=settings.port,
                        access_log=False, log_level=settings.log_level.lower())
        elif args.command == "check-env":
            if args.real:
                parser.exit(2, "请显式使用 db-check 或带确认与预算的 check-service；本命令未发起网络请求。\n")
            print(json.dumps(dependency_report(settings), ensure_ascii=False, indent=2))
        elif args.command == "db-check":
            from multimodal_rag.infrastructure.database import check_database
            print(json.dumps(check_database(settings), ensure_ascii=False, indent=2))
        elif args.command == "db-migrate":
            if not args.confirm:
                parser.exit(2, "迁移会建表，请明确提供 --confirm；未执行SQL。\n")
            from multimodal_rag.infrastructure.database import migrate
            print(json.dumps(migrate(settings), ensure_ascii=False, indent=2))
        elif args.command == "check-service":
            if not args.confirm_live:
                parser.exit(2, "需要 --confirm-live 确认真实调用与预算；未发起网络请求。\n")
            if args.task_id and args.service != "parser":
                parser.exit(2, "--task-id 仅供解析任务恢复使用。\n")
            from multimodal_rag.application.service_checks import check_service
            report = check_service(settings, args.service, args.max_requests, task_id=args.task_id)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["status"] == "validated" else (3 if report["status"] == "pending" else 1)
        elif args.command == "retrieve":
            if not args.confirm_live:
                parser.exit(2, "检索会调用查询Embedding并可能调用重排服务，请提供 --confirm-live；未发起网络请求。\n")
            from multimodal_rag.application.retrieval_service import retrieve
            report = retrieve(settings, args.query, max_requests=args.max_requests)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = settings.project_root / "evals/results/retrieval" / f"{stamp}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"report_path": str(output), **report}, ensure_ascii=False, indent=2))
            return 0
        elif args.command == "qa":
            if not args.confirm_live:
                parser.exit(2, "问答会调用Embedding、Rerank和Chat，请提供 --confirm-live；未发起网络请求。\n")
            from multimodal_rag.application.qa_service import answer
            report = answer(settings, args.query, max_requests=3)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = settings.project_root / "evals/results/qa" / f"{stamp}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"report_path": str(output), **report}, ensure_ascii=False, indent=2))
            return 0
        else:
            result = preview_document(args.document_id, settings)
            output = settings.project_root / "evals/results/offline-preview/preview.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            print(json.dumps({"output": str(output), "elements": len(result.elements),
                              "source_verified": result.source_verified, "external_calls": 0,
                              "persisted": False}, ensure_ascii=False, indent=2))
    except (AppError, RuntimeError) as exc:
        parser.exit(2, str(exc) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
