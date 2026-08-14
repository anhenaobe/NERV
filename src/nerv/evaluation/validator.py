import re
from collections.abc import Sequence
from typing import Any

_QUERY_ID_PATTERN = re.compile(r"^q(\d{3})$")


def validate_results(records: Sequence[dict[str, Any]]) -> None:
    if len(records) != 50:
        raise ValueError(
            "Se esperaban exactamente 50 líneas (q001-q050), "
            f"se recibieron {len(records)}."
        )

    seen_ids: set[str] = set()

    for position, record in enumerate(records, start=1):
        query_id = record.get("query_id")
        if not _QUERY_ID_PATTERN.match(query_id or ""):
            raise ValueError(
                f"Registro #{position}: query_id inválido {query_id!r}; "
                "se esperaba el formato qNNN (q001-q050)."
            )
        expected_id = f"q{position:03d}"
        if query_id != expected_id:
            raise ValueError(
                f"Registro #{position}: se esperaba query_id={expected_id!r} "
                f"en orden secuencial, se encontró {query_id!r}."
            )
        if query_id in seen_ids:
            raise ValueError(f"query_id duplicado: {query_id!r}.")
        seen_ids.add(query_id)

        documents = record.get("documents")
        if not isinstance(documents, list) or len(documents) != 3:
            found = len(documents) if isinstance(documents, list) else "ninguno"
            raise ValueError(
                f"{query_id}: se esperaban exactamente 3 documentos, "
                f"se encontraron {found}."
            )
        for doc_rank, doc in enumerate(documents, start=1):
            if doc.get("rank") != doc_rank or not doc.get("doc_id"):
                raise ValueError(
                    f"{query_id}: documento en posición {doc_rank} "
                    f"mal formado: {doc!r}."
                )

        fragments = record.get("fragments")
        if not isinstance(fragments, list) or len(fragments) != 10:
            found = len(fragments) if isinstance(fragments, list) else "ninguno"
            raise ValueError(
                f"{query_id}: se esperaban exactamente 10 fragmentos, "
                f"se encontraron {found}."
            )
        for frag_rank, fragment in enumerate(fragments, start=1):
            if fragment.get("rank") != frag_rank:
                raise ValueError(
                    f"{query_id}: fragmento en posición {frag_rank} tiene rank "
                    f"incorrecto: {fragment.get('rank')!r}."
                )
            for required_field in ("chunk_id", "doc_id", "text"):
                if not fragment.get(required_field):
                    raise ValueError(
                        f"{query_id}: fragmento #{frag_rank} sin campo "
                        f"'{required_field}'."
                    )
            word_count = len(fragment["text"].split())
            if word_count > 250:
                raise ValueError(
                    f"{query_id}: fragmento #{frag_rank} tiene {word_count} palabras "
                    "(máximo permitido: 250)."
                )
