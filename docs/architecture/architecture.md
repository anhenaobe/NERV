# Arquitectura de NERV

NERV es un sistema modular de recuperación semántica para el reto CODEFEST AD
ASTRA 2026. El repositorio usa un único paquete Python en `src/nerv` y separa
los límites de datos, transformación semántica y persistencia.

## Flujo

```text
corpus
  -> nerv.ingestion
  -> documentos.jsonl
  -> nerv.chunking
  -> chunks.jsonl
  -> nerv.embeddings
  -> nerv.vector_database
  -> nerv.retrieval
  -> resultados.jsonl
```

## Responsabilidades

- `nerv.ingestion`: lectura de formatos, OCR opcional, limpieza, IDs,
  trazabilidad y publicación de `documentos.jsonl`/`errores.jsonl`.
- `nerv.chunking`: detección de idioma, segmentación, conteo exacto y división
  jerárquica con límite efectivo derivado.
- `nerv.embeddings`: codificación de pasajes y consultas con el contrato
  semántico compartido y validación previa al modelo.
- `nerv.vector_database`: construcción, carga y persistencia del índice FAISS.
- `nerv.retrieval`: codificación de consultas, búsqueda, agregación y ranking.
- `nerv.evaluation`: métricas, validación y controles de contaminación.
- `nerv.pipeline`: orquestación E2E, aislamiento por run, checkpoints,
  manifiestos, tiempos, reutilización validada y reporte fail-closed.
- `scripts/run_e2e.ps1`: entrada operativa para los modos `quick`, `full`,
  `fresh` y `resume-fresh` en el entorno CUDA validado.
- `generador.py` y los demás scripts: flujos manuales, diagnósticos y utilidades
  de entrega que no reemplazan al orquestador canónico.

## Contrato semántico validado

- Modelo: `intfloat/multilingual-e5-small`.
- Dimensión: 384.
- Prefijos: `passage: ` y `query: `.
- Normalización: L2.
- Índice: `IndexFlatIP`.
- Objetivo suave de chunk: 256 IDs.
- Capacidad total del encoder: 512 IDs.
- Sobrecarga especial medida: 2 IDs.
- Presupuesto almacenado efectivo: 510 IDs.

La configuración canónica está en `config/encoder_config.json`. Los paths de
ejecución son relativos a la raíz o son proporcionados por el llamador.

## Estado E2E

Los subsistemas y sus contratos cruzados están implementados y cuentan con
pruebas. La ejecución de 1.761 documentos y 335.393 chunks validó históricamente
la infraestructura E2E, embeddings CUDA, `IndexFlatIP` y recuperación, pero fue
anterior a la exclusión de contaminación y a la auditoría estricta de completitud
lingüística. Esos conteos son evidencia histórica, no el estado corregido actual.

La entrada aceptada para la corrección contiene 1.760 documentos. El preflight
acotado pasó, pero la línea completa de chunks corregidos todavía requiere
`CORRECTED_CHUNKS_PASS`; por eso embeddings, FAISS, metadata y resultados
corregidos permanecen pendientes. El adaptador privado de consultas q001-q050 y
el contrato de entrega fueron validados sobre la línea histórica, sin métricas
oficiales de relevancia. El estado vigente y los límites exactos están en
[`codefest_stage1_release_report.txt`](../integration/codefest_stage1_release_report.txt).
