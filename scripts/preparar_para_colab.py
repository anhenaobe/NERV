"""Orquesta el pipeline desde el chunks.jsonl crudo hasta la puerta de Colab.

Un solo comando reemplaza la secuencia manual dispersa de antes. Es
IDEMPOTENTE: puedes correrlo las veces que quieras con el mismo
chunks.jsonl y no repite trabajo ya hecho; si detecta que chunks.jsonl
cambió (por ejemplo, tu compañero mandó una versión nueva con los chunks
gigantes ya corregidos), automáticamente vuelve a validar y filtrar.

Uso (correr las veces que haga falta, sin miedo a repetir trabajo):
    PYTHONPATH=src python3 scripts/preparar_para_colab.py \
        --chunks corpus/processed/chunks.jsonl \
        --queries corpus/queries/queries.jsonl

Qué hace, en orden:
    1. Valida chunks.jsonl contra el límite del contrato semántico.
    2. Filtra automáticamente cualquier chunk contaminado (fuga de las
       consultas oficiales hacia el corpus).
    3. Revisa si ya existen embeddings alineados con la versión actual de
       chunks_limpios.jsonl:
         - Si SÍ existen y coinciden -> te dice que ya puedes seguir con
           paso_final_local.py, sin pasar por Colab de nuevo.
         - Si NO existen o no coinciden -> te da las instrucciones EXACTAS
           de qué subir a Colab y dónde debe quedar el resultado, y se
           detiene ahí (esa parte no se puede automatizar desde acá,
           corre en el navegador).
"""

import argparse
import hashlib
import json
from pathlib import Path

from nerv.chunking.configuration import load_encoder_config
from nerv.embeddings.artifacts import validate_embedding_artifact
from nerv.evaluation.contamination import filtrar_chunks_contaminados

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def sha256_de_archivo(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            hasher.update(bloque)
    return hasher.hexdigest()


def cargar_jsonl(path: Path) -> list[dict]:
    registros = []
    with path.open("r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                registros.append(json.loads(linea))
    return registros


def escribir_jsonl(path: Path, registros: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def validar_chunks(chunks: list[dict]) -> None:
    print(f"[1/3] Validando {len(chunks)} chunks entrantes...")
    hard_limit = load_encoder_config()["encoder_max_input_tokens"]
    campos_obligatorios = (
        "doc_id",
        "chunk_id",
        "fuente",
        "formato",
        "fenomeno",
        "posicion",
        "num_tokens",
        "texto",
    )
    faltantes = 0
    sobre_limite = 0
    for chunk in chunks:
        if not all(campo in chunk for campo in campos_obligatorios):
            faltantes += 1
        if chunk.get("num_tokens", 0) > hard_limit:
            sobre_limite += 1

    if faltantes:
        raise ValueError(
            f"{faltantes} chunks sin todos los campos obligatorios. Revisa con quien te lo entregó."
        )

    if sobre_limite:
        raise ValueError(
            f"{sobre_limite} chunks superan encoder_max_input_tokens={hard_limit}; "
            "no se permite truncarlos silenciosamente."
        )
    print(f"      OK: todos los chunks respetan el límite de {hard_limit} tokens.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara el corpus hasta la puerta de Colab"
    )
    parser.add_argument(
        "--chunks",
        required=True,
        type=Path,
        help="chunks.jsonl crudo, tal como lo entrega el equipo",
    )
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument(
        "--chunks-limpios",
        type=Path,
        default=PROJECT_ROOT / "corpus/processed/chunks_limpios.jsonl",
    )
    parser.add_argument(
        "--reporte",
        type=Path,
        default=PROJECT_ROOT / "corpus/processed/reporte_contaminacion.json",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=PROJECT_ROOT / "corpus/processed/embeddings_limpios.npy",
    )
    parser.add_argument(
        "--embedding-manifest",
        type=Path,
        default=PROJECT_ROOT / "corpus/processed/embeddings_limpios.manifest.json",
    )
    parser.add_argument(
        "--estado",
        type=Path,
        default=PROJECT_ROOT / "corpus/processed/estado_pipeline.json",
    )
    args = parser.parse_args()

    hash_actual = sha256_de_archivo(args.chunks)
    estado = json.loads(args.estado.read_text()) if args.estado.exists() else {}

    if estado.get("chunks_hash") == hash_actual and args.chunks_limpios.exists():
        print(
            "[1-2/3] chunks.jsonl no cambió desde la última corrida -- reutilizando chunks_limpios.jsonl ya filtrado."
        )
        chunks_limpios = cargar_jsonl(args.chunks_limpios)
    else:
        chunks = cargar_jsonl(args.chunks)
        validar_chunks(chunks)

        print(
            "[2/3] Filtrando contaminación (marcadores sospechosos + fuga de consultas oficiales)..."
        )
        queries = cargar_jsonl(args.queries)
        chunks_limpios, reporte = filtrar_chunks_contaminados(chunks, queries)

        escribir_jsonl(args.chunks_limpios, chunks_limpios)
        args.reporte.parent.mkdir(parents=True, exist_ok=True)
        args.reporte.write_text(json.dumps(reporte, ensure_ascii=False, indent=2))

        if reporte:
            print(
                f"      ⚠ Se excluyeron {len(reporte)} chunks contaminados. Detalle en: {args.reporte}"
            )
            for item in reporte:
                print(
                    f"         - {item['chunk_id']} ({item['fuente']}) -> {item['razones']}"
                )
        else:
            print("      OK: no se encontró contaminación en este corpus.")

        print(
            f"      chunks_limpios.jsonl escrito: {len(chunks_limpios)} chunks -> {args.chunks_limpios}"
        )

        estado = {
            "chunks_hash": hash_actual,
            "chunks_limpios_total": len(chunks_limpios),
        }
        args.estado.parent.mkdir(parents=True, exist_ok=True)
        args.estado.write_text(json.dumps(estado, indent=2))

    print("\n[3/3] Verificando si ya existen embeddings alineados...")
    embeddings_listos = False
    if args.embeddings.exists() and args.embedding_manifest.exists():
        try:
            validate_embedding_artifact(
                args.embeddings,
                args.embedding_manifest,
                args.chunks_limpios,
            )
            embeddings_listos = True
        except (OSError, ValueError) as error:
            print(f"      Artefacto de embeddings no reutilizable: {error}")

    if embeddings_listos:
        print(
            f"      OK: {args.embeddings} ya está alineado con chunks_limpios.jsonl ({len(chunks_limpios)} filas)."
        )
        print(
            "\n✅ Todo listo. Puedes seguir directo con paso_final_local.py, sin pasar por Colab de nuevo:"
        )
        print(
            f"   PYTHONPATH=src python3 scripts/paso_final_local.py "
            f"--chunks {args.chunks_limpios} --embeddings {args.embeddings} "
            f"--embedding-manifest {args.embedding_manifest} "
            f"--queries {args.queries}"
        )
        return

    print("\n" + "=" * 70)
    print("ACCIÓN MANUAL REQUERIDA -- este paso corre en el navegador, no aquí")
    print("=" * 70)
    print(f"""
1. Sube este archivo a tu Google Drive (carpeta codefest/):
       {args.chunks_limpios.resolve()}

2. Abre el notebook NERV_embeddings_colab.ipynb en Colab.
   En la celda "4. Configurar rutas", cambia RUTA_CHUNKS para que apunte
   al archivo que acabas de subir (chunks_limpios.jsonl, no chunks.jsonl).

3. Actívalo con GPU T4 (Entorno de ejecución > Cambiar tipo de entorno de
   ejecución) y corre todas las celdas.

4. Descarga el resultado y guárdalo EXACTAMENTE en esta ruta local:
       {args.embeddings.resolve()}

5. Vuelve a correr este mismo comando -- va a detectar que los embeddings
   ya están listos y te va a dar directo el siguiente comando a correr.
""")


if __name__ == "__main__":
    main()
