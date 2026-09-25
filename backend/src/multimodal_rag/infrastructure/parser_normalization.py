"""0C：将MinerU内容列表映射回原文来源，不把样例局部页号冒充原书页号。

云端版本的bbox量纲未固定前保留原值，输出页级定位，不猜测归一化坐标。
本模块不联网、不写业务库；返回的raw_text是解析器转写，不是人工真值。
"""
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from multimodal_rag.core.errors import AppError
from multimodal_rag.core.models import Element, SourceLocation
from multimodal_rag.infrastructure.parser_adapter import inspect_result_archive

NORMALIZER_VERSION = "mineru-content-list-source-v2"


def text_field(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(x,str) for x in value):
        return "\n".join(value)
    raise AppError("invalid_parser_text", "解析文本字段格式不符合契约", 502)


def normalize_archive(content: bytes, sample: dict, max_unpacked=67_108_864) -> dict:
    artifact = inspect_result_archive(content, max_unpacked)
    pages = sample["pages"]
    if len(pages) != sample["page_count"] or any(p["split"] != "dev" for p in pages):
        raise AppError("invalid_sample_mapping", "验证样例页映射不完整或包含非开发数据", 422)
    elements, warnings, seen, heading_stack = [], [], set(), []
    with ZipFile(BytesIO(content)) as archive:
        structured = next(n for n in archive.namelist() if n.endswith("content_list.json"))
        base = PurePosixPath(structured).parent
        rows = json.loads(archive.read(structured))
        for index,row in enumerate(rows):
            page_idx = row["page_idx"]
            if not 0 <= page_idx < len(pages):
                raise AppError("parser_page_out_of_range", "解析页号超出已提交片段，拒绝错误回映射", 502)
            origin = pages[page_idx]
            seen.add(page_idx)
            type_ = row.get("type", "unknown")
            if not isinstance(type_, str):
                raise AppError("invalid_parser_type", "解析元素类型不合法", 502)
            reference = row.get("img_path")
            asset = None
            if reference:
                if not isinstance(reference,str):
                    raise AppError("invalid_image_reference", "图像引用不是字符串", 502)
                path = PurePosixPath(reference)
                if path.is_absolute() or '..' in path.parts or '\\' in reference or ':' in reference:
                    raise AppError("invalid_image_reference", "拒绝越界或外部图像引用", 502)
                asset = (base/path).as_posix()
                if asset not in archive.namelist():
                    warnings.append({"row":index,"code":"missing_image_resource"})
            decoration_candidate = type_ in {'header', 'footer', 'page_number'}
            # Live samples mislabel screenshot controls (e.g. Settings) as headers.
            # Only exclude recognizable project footers / numeric page labels;
            # keep other nonempty text until a provenance-aware cleaning pass.
            raw_label = text_field(row.get('text')).strip()
            boilerplate = ((type_ == 'footer' and raw_label.startswith('PingCAP docs-cn |'))
                           or (type_ == 'page_number' and raw_label.isdigit()))
            known = {'text','title','table','image','chart','code','equation','list',
                     'header','footer','page_number','page_footnote'}
            kind = {"text":"text","title":"heading","table":"table","image":"image","chart":"image","code":"code",
                    "equation":"text","list":"text"}.get(type_, "text")
            if type_ == "text" and row.get("text_level"):
                kind = "heading"
            if type_ not in known:
                warnings.append({"row":index,"code":"unknown_type_preserved_as_text","provider_type":type_})
            fields = {"table":["table_caption","table_body","table_footnote"],
                      "image":["image_caption","content","image_footnote"],
                      "chart":["chart_caption","content","chart_footnote"],"list":["list_items"],
                      "code":["code_body","text"],"equation":["text"]}.get(type_, ["text"])
            text = "\n".join(text_field(row.get(field)) for field in fields if row.get(field) is not None)
            if not text.strip() and not asset and not decoration_candidate:
                warnings.append({"row":index,"code":"empty_parsed_element"})
            if type_ == 'chart' and not text_field(row.get('content')).strip():
                warnings.append({"row":index,"code":"chart_values_require_visual_reading"})
            if origin["source_kind"] == "pdf":
                location = SourceLocation(source_path=origin["source_path"],kind="pdf",precision="page",page_number=origin["source_page"])
            else:
                location = SourceLocation(source_path=origin["source_path"],kind="image",precision="image",
                    image_ref=origin["source_path"],image_width=origin["width"],image_height=origin["height"])
            identity = f'{NORMALIZER_VERSION}:{artifact["archive_sha256"]}:{sample["sample_id"]}:{index}'
            version_id = origin.get('version_id') or 'ver_'+origin['source_sha256'][:24]
            if kind == "heading" and text.strip():
                provider_level = row.get("text_level")
                level = provider_level if type(provider_level) is int and provider_level > 0 else 1
                level = min(level, len(heading_stack) + 1)
                heading_stack = heading_stack[:level - 1] + [text.strip()]
            element = Element(element_id='el_'+hashlib.sha256(identity.encode()).hexdigest()[:24],
                document_id=origin['document_id'],version_id=version_id,kind=kind,
                order=index,raw_text=text,heading_path=list(heading_stack),source=location,
                image_ref=f'{sample["sample_id"]}/{asset}' if asset else None)
            elements.append({"element":element.model_dump(),"provider_page_idx":page_idx,"provider_type":type_,
                "provider_bbox":row.get("bbox"),"bbox_interpreted":False,"archive_image_member":asset,
                "index_eligible":not boilerplate and bool(text.strip() or asset),
                "page_decoration_candidate":decoration_candidate,
                "excluded_reason":"page_decoration" if boilerplate else ("empty_element" if not text.strip() and not asset else None),
                "text_provenance":"parser_transcription_not_ground_truth"})
    missing = sorted(set(range(len(pages))) - seen)
    if missing:
        warnings.append({"code":"pages_without_elements","local_page_indices":missing})
    return {"normalizer_version":NORMALIZER_VERSION,"sample_id":sample['sample_id'],"artifact":artifact,
            "elements":elements,"warnings":warnings,"mapped_elements":len(elements),
            "source_precision":"page_or_original_image","quality_review":"pending",
            "status":"mapped_with_warnings" if warnings else "mapped"}


def materialize_archive_assets(content: bytes, normalized: dict, project_root: Path,
                               processing_version_id: str) -> dict:
    """Persist only image members referenced by normalized elements.

    The parser archive is untrusted.  ``normalize_archive`` and
    ``inspect_result_archive`` have already rejected traversal, encrypted
    entries and ambiguous structures; this function still resolves every
    target below one dedicated derived-data root and never calls extractall.
    """
    allowed_suffixes = {".png", ".jpg", ".jpeg"}
    asset_root = (project_root / "data/derived/mineru_assets" / processing_version_id).resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(content)) as archive:
        available = set(archive.namelist())
        for row in normalized.get("elements", []):
            member = row.get("archive_image_member")
            if not member:
                continue
            pure = PurePosixPath(member)
            if (member not in available or pure.is_absolute() or ".." in pure.parts
                    or pure.suffix.lower() not in allowed_suffixes):
                raise AppError("invalid_image_reference", "解析图片资源不安全或格式不受支持", 502)
            target = (asset_root / Path(*pure.parts)).resolve()
            if not target.is_relative_to(asset_root):
                raise AppError("invalid_image_reference", "解析图片资源越界", 502)
            payload = archive.read(member)
            if not payload:
                raise AppError("invalid_image_resource", "解析图片资源为空", 502)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(payload).digest():
                    raise AppError("parser_asset_conflict", "同一处理版本的图片产物发生冲突", 409)
            else:
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_bytes(payload)
                temporary.replace(target)
            relative = target.relative_to(project_root.resolve()).as_posix()
            row["element"]["image_ref"] = relative
            row["materialized_asset"] = {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "byte_size": len(payload),
            }
    return normalized
