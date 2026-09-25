"""Read visual assets that are actually referenced by retrieved Chunks.

MinerU may classify a screenshot containing tabular text as ``table`` rather
than ``image``.  The durable visual contract is therefore a validated
``image_ref``, not one provider-specific element kind.
"""
from __future__ import annotations

from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import Settings


def load_image_elements_for_chunks(settings: Settings, chunk_ids: list[str]) -> list[dict]:
    if not chunk_ids:
        return []
    with connection(settings, read_only=True) as conn:
        rows = conn.execute(
            """SELECT c.chunk_id,e.element_id,e.document_id,e.ordinal,e.image_ref,
                      e.raw_text,e.heading_path,e.source,d.title,d.source_path,
                      (SELECT n.raw_text FROM mrag.elements n
                        WHERE n.processing_version_id=e.processing_version_id
                          AND n.ordinal<e.ordinal AND n.raw_text<>''
                        ORDER BY n.ordinal DESC LIMIT 1) AS visual_context
                 FROM mrag.chunks c
                 JOIN LATERAL jsonb_array_elements_text(c.element_ids) ids(element_id) ON TRUE
                 JOIN mrag.elements e ON e.element_id=ids.element_id
                 JOIN mrag.documents d ON d.document_id=e.document_id
                WHERE c.chunk_id=ANY(%s) AND e.image_ref IS NOT NULL
                ORDER BY e.document_id,e.ordinal""",
            (chunk_ids,),
        ).fetchall()
    rank = {chunk_id: index for index, chunk_id in enumerate(chunk_ids)}
    result = [{
        "chunk_id": row[0], "element_id": row[1], "document_id": row[2],
        "ordinal": row[3], "image_ref": row[4], "raw_text": row[5],
        "heading_path": row[6], "source": row[7], "document_title": row[8],
        "document_source_path": row[9], "visual_context": row[10] or "",
    } for row in rows]
    result.sort(key=lambda item: (rank.get(item["chunk_id"], len(rank)), item["ordinal"]))
    return result


def load_active_image_element(settings: Settings, element_id: str) -> dict | None:
    """Resolve one visual element belonging to a currently active processing version."""
    with connection(settings, read_only=True) as conn:
        row = conn.execute(
            """SELECT e.element_id,e.document_id,e.image_ref,e.source,d.title,d.source_path
                 FROM mrag.elements e
                 JOIN mrag.processing_versions pv ON pv.processing_version_id=e.processing_version_id
                 JOIN mrag.documents d ON d.document_id=e.document_id
                WHERE e.element_id=%s AND e.image_ref IS NOT NULL
                  AND pv.release_status='active'
                LIMIT 1""",
            (element_id,),
        ).fetchone()
    if not row:
        return None
    return {"element_id": row[0], "document_id": row[1], "image_ref": row[2],
            "source": row[3], "document_title": row[4], "document_source_path": row[5]}


def load_active_image_elements_for_title(settings: Settings, title: str) -> list[dict]:
    """Return visual assets linked to active Chunks for an exact title."""
    with connection(settings, read_only=True) as conn:
        rows = conn.execute(
            """SELECT e.element_id,e.document_id,e.ordinal,e.image_ref,e.raw_text,
                      e.heading_path,e.source,d.title,d.source_path,c.chunk_id,c.text,
                      (SELECT n.raw_text FROM mrag.elements n
                        WHERE n.processing_version_id=e.processing_version_id
                          AND n.ordinal<e.ordinal AND n.raw_text<>''
                        ORDER BY n.ordinal DESC LIMIT 1) AS visual_context
                 FROM mrag.documents d
                 JOIN mrag.processing_versions pv ON pv.document_id=d.document_id
                  AND pv.release_status='active'
                 JOIN mrag.elements e ON e.processing_version_id=pv.processing_version_id
                  AND e.image_ref IS NOT NULL
                 JOIN mrag.chunks c ON c.processing_version_id=pv.processing_version_id
                  AND c.element_ids ? e.element_id
                WHERE d.title=%s
                ORDER BY e.ordinal,c.ordinal""",
            (title,),).fetchall()
    result, seen = [], set()
    for row in rows:
        if row[0] in seen:
            continue
        seen.add(row[0])
        result.append({
            "element_id": row[0], "document_id": row[1], "ordinal": row[2],
            "image_ref": row[3], "raw_text": row[4], "heading_path": row[5],
            "source": row[6], "document_title": row[7],
            "document_source_path": row[8], "chunk_id": row[9],
            "chunk_text": row[10], "visual_context": row[11] or "",
        })
    return result
