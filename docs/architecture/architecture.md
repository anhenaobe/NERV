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
pruebas. El run fresh aceptado produjo 1761 documentos y 335393 chunks sin
errores de validación. La continuación `resume-fresh` ejecutó embeddings en CUDA,
publicó 335393 vectores de 384 dimensiones, construyó un `IndexFlatIP` con
`ntotal=335393` y completó recuperación/validación con el fixture sintético.

El documento real de 50 consultas existe como PDF privado, pero su capa de
texto no preserva correctamente los caracteres españoles. Requiere un adaptador
pequeño y verificado a JSONL antes de ejecutar consultas oficiales. Este trabajo
pendiente no invalida la aceptación de infraestructura E2E y no autoriza llamar
oficiales a los resultados sintéticos.
