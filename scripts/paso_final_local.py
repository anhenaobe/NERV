"""Paso final: construir el índice FAISS real y generar resultados.jsonl.

Usa los embeddings ya generados en Colab (embeddings.npy) junto con el
chunks.jsonl real, corre las 50 consultas reales, valida el resultado y
deja todo en la estructura oficial de entrega. Esta parte es rápida
(segundos), no necesita GPU: la única etapa lenta era generar los
embeddings, y eso ya está resuelto.

Uso:
    PYTHONPATH=src python3 scripts/paso_final_local.py \
        --chunks corpus/processed/chunks.jsonl \
        --embeddings corpus/processed/embeddings.npy \
        --queries corpus/queries/queries.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from nerv.embeddings.encoder import load_encoder
from nerv.evaluation.validator import validate_results
from nerv.retrieval.queries import load_queries
from nerv.retrieval.query_encoder import encode_query
from nerv.retrieval.ranking import (
    aggregate_scores_by_document,
    deduplicate_by_chunk,
    rank_results,
)
from nerv.retrieval.retriever import retrieve
from nerv.vector_database.build_index import build_index_from_artifact
from nerv.vector_database.save_index import save_index

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_WORDS_PER_FRAGMENT = 250
CANDIDATE_POOL_SIZE = 50


def cargar_chunks(path: Path) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def truncar_respetando_oraciones(texto: str, max_palabras: int) -> str:
    palabras = texto.split()
    if len(palabras) <= max_palabras:
        return texto
    truncado = " ".join(palabras[:max_palabras])
    corte = max(truncado.rfind(". "), truncado.rfind("? "), truncado.rfind("! "))
    if corte == -1:
        return texto
    return truncado[: corte + 1].strip()


def resolver_consulta(query_id: str, hits, chunks: list[dict]) -> dict:
    candidatos = [
        {
            "doc_id": chunks[pos]["doc_id"],
            "chunk_id": chunks[pos]["chunk_id"],
            "text": chunks[pos]["texto"],
            "score": score,
        }
        for pos, score in hits
    ]
    candidatos = deduplicate_by_chunk(candidatos)
    top_fragmentos = rank_results(candidatos, top_k=10)
    top_documentos = rank_results(aggregate_scores_by_document(candidatos), top_k=3)

    return {
        "query_id": query_id,
        "documents": [
            {"rank": r, "doc_id": d["doc_id"]}
            for r, d in enumerate(top_documentos, start=1)
        ],
        "fragments": [
            {
                "rank": r,
                "chunk_id": f["chunk_id"],
                "doc_id": f["doc_id"],
                "text": truncar_respetando_oraciones(f["text"], MAX_WORDS_PER_FRAGMENT),
            }
            for r, f in enumerate(top_fragmentos, start=1)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construye el índice real y genera resultados.jsonl"
    )
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--embeddings", required=True, type=Path)
    parser.add_argument("--embedding-manifest", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument(
        "--base-vectorial",
        type=Path,
        default=PROJECT_ROOT / "entrega" / "base_vectorial",
        help="Carpeta donde se guarda index.faiss + metadata.jsonl (estructura oficial de entrega)",
    )
    parser.add_argument(
        "--salida", type=Path, default=PROJECT_ROOT / "resultados.jsonl"
    )
    parser.add_argument(
        "--ejemplos",
        type=int,
        default=3,
        help="Cuántas consultas mostrar en detalle al final, para inspección rápida",
    )
    args = parser.parse_args()

    tiempos: dict[str, float] = {}
    t0 = time.perf_counter()

    print("[1/6] Cargando chunks y embeddings...")
    inicio = time.perf_counter()
    chunks = cargar_chunks(args.chunks)
    tiempos["carga_datos"] = time.perf_counter() - inicio
    print(f"      {len(chunks)} chunks cargados ({tiempos['carga_datos']:.1f} s)")

    print("[2/6] Construyendo índice FAISS...")
    inicio = time.perf_counter()
    index, _manifest = build_index_from_artifact(
        args.embeddings,
        args.embedding_manifest,
        args.chunks,
        args.chunks,
    )
    tiempos["construccion_indice"] = time.perf_counter() - inicio
    print(f"      {tiempos['construccion_indice']:.2f} s")

    print("[3/6] Guardando index.faiss y metadata.jsonl en la estructura oficial...")
    inicio = time.perf_counter()
    args.base_vectorial.mkdir(parents=True, exist_ok=True)
    save_index(
        index,
        args.base_vectorial / "index.faiss",
        embedding_manifest=_manifest,
        metadata_path=args.chunks,
    )
    with (args.base_vectorial / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    tiempos["guardado"] = time.perf_counter() - inicio
    print(f"      Guardado en: {args.base_vectorial} ({tiempos['guardado']:.2f} s)")

    print("[4/6] Cargando encoder del contrato semantico para consultas...")
    inicio = time.perf_counter()
    encoder = load_encoder()
    tiempos["carga_encoder"] = time.perf_counter() - inicio
    print(f"      {tiempos['carga_encoder']:.1f} s")

    print("[5/6] Ejecutando las 50 consultas reales...")
    queries = load_queries(args.queries)
    inicio = time.perf_counter()
    resultados = []
    for query_item in queries:
        query_id = query_item["query_id"]
        texto = query_item["text"]
        query_embedding = encode_query(texto, encoder)
        hits = retrieve(query_embedding, index, top_k=CANDIDATE_POOL_SIZE)
        resultados.append(resolver_consulta(query_id, hits, chunks))
    tiempos["recuperacion"] = time.perf_counter() - inicio
    print(
        f"      {tiempos['recuperacion']:.1f} s ({tiempos['recuperacion'] / len(queries):.3f} s/consulta)"
    )

    print("[6/6] Validando y escribiendo resultados.jsonl...")
    validate_results(resultados)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("w", encoding="utf-8") as f:
        for r in resultados:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("      Validación OK: 50 líneas, esquema correcto.")

    tiempo_total = time.perf_counter() - t0

    print(
        "\n===== RESUMEN DE TIEMPOS (CPU local, embeddings ya generados en GPU) ====="
    )
    print(f"Carga de chunks + embeddings:   {tiempos['carga_datos']:.1f} s")
    print(f"Construcción del índice FAISS:  {tiempos['construccion_indice']:.2f} s")
    print(f"Guardado en disco:              {tiempos['guardado']:.2f} s")
    print(f"Carga del encoder (consultas):  {tiempos['carga_encoder']:.1f} s")
    print(f"Recuperación (50 consultas):    {tiempos['recuperacion']:.1f} s")
    print(f"Tiempo total (esta etapa):      {tiempo_total:.1f} s")
    print(f"\nresultados.jsonl escrito en: {args.salida}")
    print(f"base_vectorial en: {args.base_vectorial}")

    if args.ejemplos > 0:
        print(
            f"\n===== {args.ejemplos} CONSULTAS DE EJEMPLO (para inspección rápida) ====="
        )
        for r in resultados[: args.ejemplos]:
            query_texto = next(
                (q.get("text") or q.get("query"))
                for q in queries
                if q["query_id"] == r["query_id"]
            )
            print(f"\n{r['query_id']}: {query_texto}")
            for doc in r["documents"]:
                print(f"   doc #{doc['rank']}: {doc['doc_id']}")
            for frag in r["fragments"][:2]:
                print(
                    f"   [frag #{frag['rank']}] {frag['chunk_id']}: {frag['text'][:100]}..."
                )


if __name__ == "__main__":
    main()
