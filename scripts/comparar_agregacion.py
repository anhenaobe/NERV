"""Compara CombSUM vs max pooling en la agregación a nivel de documento.

Reutiliza el index.faiss y metadata.jsonl ya construidos (no regenera
embeddings, no necesita GPU). Corre las 50 consultas reales con las dos
estrategias en paralelo y muestra dónde cambian los resultados. Al final,
escribe resultados.jsonl usando max pooling (la versión que vamos a usar).

Uso:
    PYTHONPATH=src python3 scripts/comparar_agregacion.py \
        --base-vectorial entrega/base_vectorial/encoder_multilingual-e5-small \
        --queries corpus/queries/queries.jsonl
"""

import argparse
import json
from pathlib import Path

from nerv.embeddings.encoder import load_encoder
from nerv.evaluation.validator import validate_results
from nerv.retrieval.query_encoder import encode_query
from nerv.retrieval.ranking import deduplicate_by_chunk, rank_results
from nerv.retrieval.retriever import retrieve
from nerv.vector_database.load_index import load_index

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_POOL_SIZE = 50
MAX_WORDS_PER_FRAGMENT = 250


def cargar_metadata(path: Path) -> list[dict]:
    registros = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                registros.append(json.loads(line))
    return registros


def cargar_queries(path: Path) -> list[dict]:
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def agregar_por_suma(fragment_candidates: list[dict]) -> list[dict]:
    """CombSUM: la estrategia anterior, para comparar."""
    totales: dict[str, float] = {}
    for f in fragment_candidates:
        totales[f["doc_id"]] = totales.get(f["doc_id"], 0.0) + f["score"]
    return [{"doc_id": doc_id, "score": score} for doc_id, score in totales.items()]


def agregar_por_maximo(fragment_candidates: list[dict]) -> list[dict]:
    """Max pooling: la estrategia nueva."""
    mejores: dict[str, float] = {}
    for f in fragment_candidates:
        if f["doc_id"] not in mejores or f["score"] > mejores[f["doc_id"]]:
            mejores[f["doc_id"]] = f["score"]
    return [{"doc_id": doc_id, "score": score} for doc_id, score in mejores.items()]


def truncar_respetando_oraciones(texto: str, max_palabras: int) -> str:
    palabras = texto.split()
    if len(palabras) <= max_palabras:
        return texto
    truncado = " ".join(palabras[:max_palabras])
    corte = max(truncado.rfind(". "), truncado.rfind("? "), truncado.rfind("! "))
    if corte == -1:
        return texto
    return truncado[: corte + 1].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara CombSUM vs max pooling")
    parser.add_argument("--base-vectorial", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument(
        "--salida", type=Path, default=PROJECT_ROOT / "resultados.jsonl"
    )
    args = parser.parse_args()

    print("[1/3] Cargando índice, metadata, encoder y consultas...")
    metadata_path = args.base_vectorial / "metadata.jsonl"
    index = load_index(args.base_vectorial / "index.faiss", metadata_path=metadata_path)
    metadata = cargar_metadata(metadata_path)
    encoder = load_encoder()
    queries = cargar_queries(args.queries)

    print("[2/3] Corriendo las 50 consultas con ambas estrategias...\n")
    cambios = 0
    resultados_max = []

    for query_item in queries:
        query_id = query_item["query_id"]
        texto_query = query_item.get("text") or query_item.get("query")

        query_embedding = encode_query(texto_query, encoder)
        hits = retrieve(query_embedding, index, top_k=CANDIDATE_POOL_SIZE)

        candidatos = [
            {
                "doc_id": metadata[pos]["doc_id"],
                "chunk_id": metadata[pos]["chunk_id"],
                "text": metadata[pos]["texto"],
                "score": score,
            }
            for pos, score in hits
        ]
        candidatos = deduplicate_by_chunk(candidatos)

        top_docs_suma = rank_results(agregar_por_suma(candidatos), top_k=3)
        top_docs_max = rank_results(agregar_por_maximo(candidatos), top_k=3)

        ids_suma = [d["doc_id"] for d in top_docs_suma]
        ids_max = [d["doc_id"] for d in top_docs_max]

        if ids_suma != ids_max:
            cambios += 1
            print(f"{query_id}: CAMBIÓ")
            print(f"   suma:  {ids_suma}")
            print(f"   max:   {ids_max}")
        else:
            print(f"{query_id}: igual   {ids_max}")

        top_fragmentos = rank_results(candidatos, top_k=10)
        resultados_max.append(
            {
                "query_id": query_id,
                "documents": [
                    {"rank": r, "doc_id": d["doc_id"]}
                    for r, d in enumerate(top_docs_max, start=1)
                ],
                "fragments": [
                    {
                        "rank": r,
                        "chunk_id": f["chunk_id"],
                        "doc_id": f["doc_id"],
                        "text": truncar_respetando_oraciones(
                            f["text"], MAX_WORDS_PER_FRAGMENT
                        ),
                    }
                    for r, f in enumerate(top_fragmentos, start=1)
                ],
            }
        )

    print(
        f"\n[3/3] Total de consultas donde cambió el top-3 de documentos: {cambios}/{len(queries)}"
    )

    validate_results(resultados_max)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("w", encoding="utf-8") as f:
        for r in resultados_max:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nresultados.jsonl (max pooling) escrito en: {args.salida}")


if __name__ == "__main__":
    main()
