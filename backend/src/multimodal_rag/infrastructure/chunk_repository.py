"""Idempotent persistence for structural chunks; no vector index is built here."""
from __future__ import annotations

from psycopg.types.json import Jsonb

from multimodal_rag.core.models import Chunk
from multimodal_rag.infrastructure.database import connection
from multimodal_rag.infrastructure.settings import Settings


def persist_chunks(settings: Settings, chunks: list[Chunk]) -> int:
    if not chunks:
        return 0
    with connection(settings) as conn:
        for chunk in chunks:
            conn.execute(
                """INSERT INTO mrag.chunks(chunk_id,processing_version_id,document_id,version_id,ordinal,text,
                     heading_path,element_ids,source_locations,length_unit,estimated_length)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (chunk_id) DO UPDATE SET text=EXCLUDED.text,
                     heading_path=EXCLUDED.heading_path,element_ids=EXCLUDED.element_ids,
                     source_locations=EXCLUDED.source_locations,estimated_length=EXCLUDED.estimated_length""",
                (chunk.chunk_id, chunk.processing_version_id, chunk.document_id, chunk.version_id, chunk.ordinal,
                 chunk.text, Jsonb(chunk.heading_path), Jsonb(chunk.element_ids),
                 Jsonb([location.model_dump() for location in chunk.source_locations]), chunk.length_unit,
                 chunk.estimated_length),
            )
    return len(chunks)


def load_adjacent_chunks(settings: Settings, anchor_chunk_ids: list[str], *,
                         radius: int = 1) -> dict[str, list[Chunk]]:
    """Load same-section neighbours for each retrieved anchor Chunk."""
    if not anchor_chunk_ids:
        return {}
    if not isinstance(radius, int) or not 1 <= radius <= 2:
        raise ValueError("邻接半径必须为1或2")
    with connection(settings, read_only=True) as conn:
        rows = conn.execute(
            """SELECT a.chunk_id,n.chunk_id,n.document_id,n.version_id,n.processing_version_id,
                      n.ordinal,n.text,n.heading_path,n.element_ids,n.source_locations,
                      n.length_unit,n.estimated_length
                 FROM mrag.chunks a
                 JOIN mrag.chunks n
                   ON n.processing_version_id=a.processing_version_id
                  AND n.heading_path=a.heading_path
                  AND n.ordinal BETWEEN a.ordinal-%s AND a.ordinal+%s
                  AND n.chunk_id<>a.chunk_id
                WHERE a.chunk_id=ANY(%s)
                ORDER BY a.chunk_id,ABS(n.ordinal-a.ordinal),n.ordinal""",
            (radius, radius, anchor_chunk_ids),
        ).fetchall()
    result: dict[str, list[Chunk]] = {chunk_id: [] for chunk_id in anchor_chunk_ids}
    from multimodal_rag.core.models import SourceLocation
    for row in rows:
        result.setdefault(row[0], []).append(Chunk(
            chunk_id=row[1], document_id=row[2], version_id=row[3],
            processing_version_id=row[4], ordinal=row[5], text=row[6],
            heading_path=row[7], element_ids=row[8],
            source_locations=[SourceLocation(**item) for item in row[9]],
            length_unit=row[10], estimated_length=row[11],
        ))
    return result
