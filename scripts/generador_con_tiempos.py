"""Genera resultados.jsonl con el corpus real y mide el tiempo de cada etapa.

Pensado para un corpus grande (cientos de miles de chunks): antes de lanzar
la corrida completa, calibra con una muestra pequeña y te dice cuánto va a
tardar. Guarda checkpoints durante la generación de embeddings, así que si
se corta a mitad de camino, la próxima corrida retoma donde iba.

Uso:
    # Solo calibrar (no genera nada, solo estima tiempos):
    python3 scripts/generador_con_tiempos.py --chunks ruta/chunks.jsonl \
        --queries corpus/queries/queries.jsonl --solo-estimar

    # Corrida completa:
    python3 scripts/generador_con_tiempos.py --chunks ruta/chunks.jsonl \
        --queries corpus/queries/queries.jsonl

    # Corrida de prueba con una muestra (por ejemplo, primeros 5000 chunks):
    python3 scripts/generador_con_tiempos.py --chunks ruta/chunks.jsonl \
        --queries corpus/queries/queries.jsonl --muestra 5000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nerv.chunking.configuration import load_encoder_config
from nerv.embeddings.embedding_generator import generate_embeddings
from nerv.embeddings.encoder import load_encoder
from nerv.evaluation.validator import validate_results
from nerv.retrieval.query_encoder import encode_query
from nerv.retrieval.ranking import (
    aggregate_scores_by_document,
    deduplicate_by_chunk,
    rank_results,
)
from nerv.retrieval.retriever import retrieve
from nerv.vector_database.build_index import build_index

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCH_SIZE = 64
CHECKPOINT_EVERY = 20000
MAX_WORDS_PER_FRAGMENT = 250
CANDIDATE_POOL_SIZE = 50
MUESTRA_CALIBRACION = 200


def cargar_chunks(path: Path, limite: int | None) -> list[dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limite is not None and i >= limite:
                break
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def cargar_queries(path: Path) -> list[dict]:
    queries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def calibrar(chunks: list[dict], encoder) -> float:
    """Codifica una muestra pequeña y devuelve el ritmo (chunks/segundo)."""
    muestra = chunks[:MUESTRA_CALIBRACION]
    textos = [c["texto"] for c in muestra]
    inicio = time.perf_counter()
    generate_embeddings(textos, encoder, batch_size=BATCH_SIZE)
    transcurrido = time.perf_counter() - inicio
    return len(muestra) / transcurrido


def generar_embeddings_con_checkpoint(
    chunks: list[dict], encoder, checkpoint_path: Path
) -> tuple[np.ndarray, float]:
    total = len(chunks)
    dimension = load_encoder_config()["embedding_dimension"]
    embeddings = np.zeros((total, dimension), dtype="float32")
    inicio_idx = 0

    progreso_path = checkpoint_path.with_suffix(".progreso.txt")
    if checkpoint_path.exists() and progreso_path.exists():
        completados = int(progreso_path.read_text().strip())
        if 0 < completados <= total:
            embeddings[:completados] = np.load(checkpoint_path)[:completados]
            inicio_idx = completados
            print(f"      Retomando checkpoint: {inicio_idx}/{total} ya generados")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tiempo_inicio = time.perf_counter()

    for lote_inicio in range(inicio_idx, total, BATCH_SIZE):
        lote = chunks[lote_inicio : lote_inicio + BATCH_SIZE]
        textos = [c["texto"] for c in lote]
        vectores = generate_embeddings(textos, encoder, batch_size=len(textos))
        embeddings[lote_inicio : lote_inicio + len(lote)] = vectores

        procesados = lote_inicio + len(lote)
        proximo_checkpoint = (procesados // CHECKPOINT_EVERY) * CHECKPOINT_EVERY
        if procesados == proximo_checkpoint or procesados == total:
            np.save(checkpoint_path, embeddings)
            progreso_path.write_text(str(procesados))
            transcurrido = time.perf_counter() - tiempo_inicio
            ritmo = (procesados - inicio_idx) / transcurrido if transcurrido > 0 else 0
            restantes = total - procesados
            eta_min = (restantes / ritmo / 60) if ritmo > 0 else float("inf")
            print(
                f"      {procesados}/{total} chunks "
                f"({ritmo:.1f} chunks/s, ETA {eta_min:.1f} min)"
            )

    tiempo_total = time.perf_counter() - tiempo_inicio
    return embeddings, tiempo_total


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
    raise RuntimeError(
        "Este generador legado con checkpoint no vincula filas por manifiesto. "
        "Permanece deshabilitado hasta que la Fase 2 lo sustituya por "
        "nerv.embeddings.artifacts.generate_embedding_artifact."
    )
    parser = argparse.ArgumentParser(
        description="Genera resultados.jsonl y mide tiempos."
    )
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument(
        "--muestra", type=int, default=None, help="Usar solo los primeros N chunks"
    )
    parser.add_argument(
        "--solo-estimar",
        action="store_true",
        help="Calibra y termina, sin generar nada",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "sandbox_prueba" / "embeddings_checkpoint.npy",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=PROJECT_ROOT / "sandbox_prueba" / "resultados.jsonl",
    )
    args = parser.parse_args()

    tiempos: dict[str, float] = {}
    t0 = time.perf_counter()

    print("[1/5] Cargando chunks reales...")
    chunks = cargar_chunks(args.chunks, args.muestra)
    etiqueta_muestra = (
        f" (muestra de {args.muestra})" if args.muestra else " (corpus completo)"
    )
    print(f"      {len(chunks)} chunks cargados{etiqueta_muestra}")

    print("[2/5] Cargando encoder del contrato semantico...")
    inicio = time.perf_counter()
    encoder = load_encoder()
    tiempos["carga_encoder"] = time.perf_counter() - inicio
    print(f"      {tiempos['carga_encoder']:.1f} s")

    print(f"[3/5] Calibrando con {MUESTRA_CALIBRACION} chunks...")
    ritmo = calibrar(chunks, encoder)
    eta_total_min = len(chunks) / ritmo / 60
    print(f"      Ritmo medido: {ritmo:.1f} chunks/s")
    print(
        f"      Estimado para {len(chunks)} chunks: {eta_total_min:.1f} minutos "
        f"({eta_total_min / 60:.1f} horas)"
    )

    if args.solo_estimar:
        print(
            "\n[Solo estimación] No se generaron embeddings. Corre sin --solo-estimar para procesar de verdad."
        )
        return

    print("[4/5] Generando embeddings (con checkpoints)...")
    embeddings, tiempos["embeddings"] = generar_embeddings_con_checkpoint(
        chunks, encoder, args.checkpoint
    )
    print(f"      Completado en {tiempos['embeddings'] / 60:.1f} minutos")

    print("[5/5] Construyendo índice, corriendo consultas y validando...")
    inicio = time.perf_counter()
    index = build_index(embeddings)
    tiempos["construccion_indice"] = time.perf_counter() - inicio

    queries = cargar_queries(args.queries)
    inicio = time.perf_counter()
    resultados = []
    for query_item in queries:
        query_id = query_item["query_id"]
        texto = query_item.get("text") or query_item.get("query")
        query_embedding = encode_query(texto, encoder)
        hits = retrieve(query_embedding, index, top_k=CANDIDATE_POOL_SIZE)
        resultados.append(resolver_consulta(query_id, hits, chunks))
    tiempos["recuperacion_50_consultas"] = time.perf_counter() - inicio

    validate_results(resultados)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("w", encoding="utf-8") as f:
        for r in resultados:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tiempo_total = time.perf_counter() - t0
    ms_por_chunk = tiempos["embeddings"] / len(chunks) * 1000
    s_por_consulta = tiempos["recuperacion_50_consultas"] / len(queries)

    print("\n===== RESUMEN DE TIEMPOS =====")
    print(f"Chunks procesados:              {len(chunks)}")
    print(f"Consultas procesadas:           {len(queries)}")
    print(f"Carga del encoder:              {tiempos['carga_encoder']:.1f} s")
    print(
        f"Generación de embeddings:       {tiempos['embeddings'] / 60:.1f} min ({ms_por_chunk:.1f} ms/chunk)"
    )
    print(f"Construcción del índice FAISS:  {tiempos['construccion_indice']:.2f} s")
    print(
        f"Recuperación (50 consultas):    {tiempos['recuperacion_50_consultas']:.1f} s "
        f"({s_por_consulta:.3f} s/consulta)"
    )
    print(f"Tiempo total:                   {tiempo_total / 60:.1f} min")
    print(f"\nresultados.jsonl escrito en: {args.salida}")


if __name__ == "__main__":
    main()
