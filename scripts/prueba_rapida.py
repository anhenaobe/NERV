"""Prueba rápida end-to-end con datos sintéticos.

NO forma parte de la entrega oficial. Sirve para comprobar HOY que tu
FAISS + recuperación + validación funcionan de punta a punta, usando los
31 chunks sintéticos que ya existen en:
    tests/chunking/results/frozen_pipeline_chunks.jsonl

Cuando lleguen los chunks y embeddings reales del corpus, este script se
descarta -- generador.py es el que se usa para la entrega real.

Cómo correrlo (desde la raíz de NERV/):
    python scripts/prueba_rapida.py
"""

import json
from pathlib import Path

from nerv.chunking.configuration import load_encoder_config
from nerv.chunking.token_counter import TokenCounter
from nerv.embeddings.artifacts import generate_embedding_artifact
from nerv.embeddings.encoder import load_encoder
from nerv.evaluation.validator import validate_results
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
CHUNKS_PATH = PROJECT_ROOT / "tests/chunking/results/frozen_pipeline_chunks.jsonl"
SANDBOX_DIR = PROJECT_ROOT / "sandbox_prueba"

# Consultas de ejemplo hechas a mano sobre el contenido conocido de los
# chunks sintéticos (orbital, IA, militar) -- NO son las 50 consultas
# oficiales del reto, son solo para verificar que la recuperación trae
# resultados razonables.
CONSULTAS_DE_PRUEBA = [
    "¿Qué pasó con los restos espaciales cerca de la órbita?",
    "How did the AI system classify the signals?",
    "sensores de un vehículo militar experimental",
]


def main() -> None:
    print("[1/6] Cargando chunks sintéticos...")
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as chunks_file:
        for line in chunks_file:
            if line.strip():
                chunks.append(json.loads(line))
    print(f"      {len(chunks)} chunks cargados desde {CHUNKS_PATH.name}")

    print("[2/6] Cargando encoder del contrato semantico...")
    encoder = load_encoder()

    print("[3/6] Generando artefacto y manifiesto de embeddings...")
    semantic = load_encoder_config()
    counter = TokenCounter(
        semantic["encoder_model_name"],
        revision=semantic["tokenizer_revision"],
        add_special_tokens=semantic["add_special_tokens"],
        document_prefix=semantic["document_prefix"],
        tokenizer=encoder.tokenizer,
    )
    SANDBOX_DIR.mkdir(exist_ok=True)
    embeddings_path = SANDBOX_DIR / "embeddings.npy"
    manifest_path = SANDBOX_DIR / "embeddings.manifest.json"
    manifest = generate_embedding_artifact(
        CHUNKS_PATH,
        embeddings_path,
        manifest_path,
        encoder=encoder,
        token_counter=counter,
        batch_size=8,
        device="cpu",
    )

    print("[4/6] Construyendo índice FAISS y guardando en sandbox_prueba/...")
    index, _validated_manifest = build_index_from_artifact(
        embeddings_path,
        manifest_path,
        CHUNKS_PATH,
        CHUNKS_PATH,
    )
    save_index(
        index,
        SANDBOX_DIR / "index.faiss",
        embedding_manifest=manifest,
        metadata_path=CHUNKS_PATH,
    )
    with (SANDBOX_DIR / "metadata.jsonl").open("w", encoding="utf-8") as metadata_file:
        for chunk in chunks:
            metadata_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"      Índice y metadata guardados en {SANDBOX_DIR}")

    print("[5/6] Ejecutando consultas de prueba...\n")
    for consulta in CONSULTAS_DE_PRUEBA:
        query_embedding = encode_query(consulta, encoder)
        hits = retrieve(query_embedding, index, top_k=10)

        candidates = [
            {
                "doc_id": chunks[pos]["doc_id"],
                "chunk_id": chunks[pos]["chunk_id"],
                "text": chunks[pos]["texto"],
                "score": score,
            }
            for pos, score in hits
        ]
        candidates = deduplicate_by_chunk(candidates)
        top_fragments = rank_results(candidates, top_k=3)
        top_docs = rank_results(aggregate_scores_by_document(candidates), top_k=3)

        print(f"Consulta: {consulta}")
        for doc in top_docs:
            print(f"   doc {doc['doc_id']} (score={doc['score']:.3f})")
        for fragment in top_fragments:
            preview = fragment["text"][:80]
            print(f"   [{fragment['score']:.3f}] {fragment['chunk_id']}: {preview}...")
        print()

    print("[6/6] Prueba de validate_results con un registro mínimo ficticio...")
    registro_ejemplo = {
        "query_id": "q001",
        "documents": [{"rank": i, "doc_id": f"DOC-{i}"} for i in range(1, 4)],
        "fragments": [
            {
                "rank": i,
                "chunk_id": f"C-{i}",
                "doc_id": "DOC-1",
                "text": "texto de prueba",
            }
            for i in range(1, 11)
        ],
    }
    try:
        validate_results([registro_ejemplo] * 50)
        print("      OK: 50 registros idénticos con ids repetidos deberían fallar...")
    except ValueError as error:
        print(f"      OK, el validador detecta problemas correctamente: {error}")

    print(
        "\n[Listo] Si viste resultados coherentes arriba, tu pipeline funciona end-to-end."
    )


if __name__ == "__main__":
    main()
