"""Deteccion automatica de contaminacion de datos en el corpus.

Filtra chunks que parecen ser una fuga del propio banco de evaluacion
(por ejemplo, un archivo interno que contenga las preguntas oficiales
junto a una respuesta ya armada) antes de que lleguen a generar
embeddings. Corre automaticamente en el pipeline para que esto no
dependa de que alguien lo note a mano cada vez que llega un corpus nuevo.

Tres señales independientes, cualquiera marca un chunk:

1. FUENTE CONOCIDA: el campo `fuente` coincide exactamente con un archivo
   ya identificado como contaminado (FUENTES_EXCLUIDAS). Esta es la
   defensa mas confiable contra que el MISMO archivo reaparezca en una
   version futura del corpus -- no depende de ningun patron de texto.
2. MARCADORES ESTRUCTURALES: aparecen al menos 2 de los 4 marcadores
   tipicos de ese formato interno (PREGUNTA:, FRAGMENTO:, DOCUMENTO:,
   [Hoja: F). No se exige que aparezcan TODOS juntos -- distintas filas
   del mismo archivo usan distintas combinaciones.
3. FUGA LITERAL: el texto completo de alguna de las 50 consultas
   oficiales aparece de forma (casi) literal dentro del chunk.
"""

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

# Fuentes ya confirmadas como contaminacion (no corpus real). Agregar aqui
# cualquier archivo nuevo que se identifique en el futuro.
FUENTES_EXCLUIDAS = {
    "F3_Dinamicas_Territoriales/FASE ORDENADA CODEFEST.xlsx",
}

MARCADORES_INDIVIDUALES = ("PREGUNTA:", "FRAGMENTO:", "DOCUMENTO:", "[Hoja: F")
MINIMO_MARCADORES = 2
LONGITUD_MINIMA_COINCIDENCIA = (
    20  # caracteres; evita falsos positivos con queries muy cortas
)


def _normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = "".join(
        c
        for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", texto).strip()


def detectar_chunks_contaminados(
    chunks: Sequence[dict[str, Any]],
    queries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reporta los chunks que parecen contaminados y la razon de cada uno."""
    queries_normalizadas = [
        (q["query_id"], _normalizar(q.get("text") or q.get("query") or ""))
        for q in queries
    ]

    reporte = []
    for chunk in chunks:
        texto = chunk.get("texto", "")
        texto_normalizado = _normalizar(texto)
        fuente = chunk.get("fuente", "")
        razones = []

        if fuente in FUENTES_EXCLUIDAS:
            razones.append("fuente_conocida_excluida")

        marcadores_presentes = sum(1 for m in MARCADORES_INDIVIDUALES if m in texto)
        if marcadores_presentes >= MINIMO_MARCADORES:
            razones.append(f"marcadores_sospechosos({marcadores_presentes})")

        for query_id, query_normalizada in queries_normalizadas:
            if (
                len(query_normalizada) > LONGITUD_MINIMA_COINCIDENCIA
                and query_normalizada in texto_normalizado
            ):
                razones.append(f"contiene_texto_de_{query_id}")

        if razones:
            reporte.append(
                {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "fuente": fuente,
                    "razones": razones,
                }
            )

    return reporte


def filtrar_chunks_contaminados(
    chunks: Sequence[dict[str, Any]],
    queries: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Devuelve (chunks_limpios, reporte_de_excluidos)."""
    reporte = detectar_chunks_contaminados(chunks, queries)
    chunk_ids_contaminados = {r["chunk_id"] for r in reporte}
    chunks_limpios = [c for c in chunks if c["chunk_id"] not in chunk_ids_contaminados]
    return chunks_limpios, reporte
