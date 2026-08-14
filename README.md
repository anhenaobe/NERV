# NERV

Sistema de recuperación semántica para el reto CODEFEST AD ASTRA 2026. NERV
transforma un corpus heterogéneo en documentos trazables, chunks compatibles con
un encoder multilingüe, vectores normalizados y resultados de recuperación
auditables.

## Problema

El corpus combina PDF, JSON, CSV, hojas de cálculo, HTML, texto, imágenes y PBF.
El sistema debe extraer contenido útil sin perder trazabilidad, dividirlo sin
truncación silenciosa y recuperar evidencia relevante para cada consulta.

## Arquitectura

```text
Corpus
  -> Ingesta
  -> documentos.jsonl
  -> Chunking
  -> chunks.jsonl
  -> Embeddings
  -> FAISS
  -> Codificación de consulta
  -> Recuperación y ranking
  -> resultados.jsonl
```

Los límites de cada etapa están documentados en
[Arquitectura](docs/architecture/architecture.md) y
[Contrato del pipeline](docs/architecture/pipeline.md).

## Estructura del repositorio

```text
config/          Configuración semántica y ejemplo de ejecución
corpus/          Entradas, caché, documentos procesados y consultas no versionadas
docs/            Arquitectura, subsistemas, decisiones, evidencia y archivo
entrega/         Estructura esperada de la entrega CODEFEST
outputs/         Artefactos generados; solo .gitkeep se versiona
scripts/         Herramientas manuales, diagnósticas y de evaluación
src/nerv/        Único paquete Python autoritativo
tests/           Pruebas y fixtures pequeños por subsistema
generador.py     Flujo de generación actualmente utilizable
pyproject.toml   Empaquetado y configuración de pytest, Ruff y mypy
```

## Subsistemas

- `nerv.ingestion`: lectores, OCR opcional, limpieza, `doc_id`, orden estable,
  incidencias y JSONL. El contrato mínimo contiene `doc_id`, `fuente`,
  `formato`, `fenomeno` y `texto`.
- `nerv.chunking`: detección ES/EN/PT, segmentación acotada, conteo con el
  tokenizer real, solapamiento y división jerárquica determinista.
- `nerv.embeddings`: adaptación a SentenceTransformer, validación previa a
  encode, manifiestos y alineación fila a fila.
- `nerv.vector_database`: construcción y persistencia FAISS.
- `nerv.retrieval`: consulta, búsqueda, agregación documental y ranking.
- `nerv.evaluation`: validadores, métricas y controles de contaminación.

## Instalación

Requiere Python 3.12 o superior. En PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

No se deben instalar dependencias ni modelos dentro del repositorio.

## Configuración

- `config/encoder_config.json`: contrato semántico canónico.
- `config/config.example.yaml`: paths y parámetros de ejecución; cópielo como
  `config/config.yaml` para configuración local.

Contrato técnico validado:

| Propiedad | Valor |
|---|---|
| Encoder | `intfloat/multilingual-e5-small` |
| Dimensión | 384 |
| Prefijo de pasaje | `passage: ` |
| Prefijo de consulta | `query: ` |
| Objetivo suave de chunk | 256 IDs |
| Capacidad total del encoder | 512 IDs |
| Tokens especiales medidos | 2 IDs |
| Presupuesto almacenado efectivo | 510 IDs |
| Normalización | L2 |
| Índice FAISS | `IndexFlatIP` |

## Comandos principales

Ingesta, conservando el wrapper histórico:

```powershell
python src/lector_corpus.py --corpus corpus/raw --salida corpus/processed
# Tras instalación editable también está disponible:
nerv-ingest --corpus corpus/raw --salida corpus/processed
```

Chunking independiente:

```powershell
python -m nerv.chunking.pipeline corpus/processed/documentos.jsonl outputs/chunks.jsonl
python -m nerv.chunking.real_corpus validate --input corpus/processed/documentos.jsonl --chunks outputs/chunks.jsonl --local-files-only
```

Consulte `python -m nerv.chunking.real_corpus --help` antes de una ejecución
costosa. Los scripts de recuperación y entrega se describen en
[Inventario de scripts](docs/architecture/scripts.md).

## Estado actual

- `nerv.pipeline` y `scripts/run_e2e.ps1` orquestan ingesta, chunking,
  embeddings, FAISS, recuperación y validación con manifiestos y métricas por
  ejecución.
- La aceptación fresh más reciente validó 1761 documentos y 335393 chunks, con
  máximo almacenado 510, cero IDs duplicados y cero fallos de contrato.
- La continuación de producción generó 335393 vectores `float32` normalizados
  de 384 dimensiones en CUDA y un `IndexFlatIP` con `ntotal=335393`.
- La recuperación se validó con 50 consultas sintéticas. El documento real de
  consultas requiere una conversión fiel a JSONL antes de producir resultados
  oficiales; los resultados sintéticos no son resultados de competencia.
- Los artefactos completos de producción permanecen locales y excluidos de Git.

## Pruebas y calidad

```powershell
python -m pytest -q
mypy src generador.py
ruff check src tests generador.py
ruff check .
```

El último comando puede mostrar deuda de estilo histórica que se reporta por
separado; no se deben introducir hallazgos nuevos en archivos modificados.

## Artefactos excluidos de Git

El corpus real, `documentos.jsonl`, `chunks*.jsonl`, matrices de embeddings,
manifiestos grandes, índices FAISS, logs, cachés de modelos y directorios de
trabajo temporales están excluidos. `corpus/` y `outputs/` conservan únicamente
archivos `.gitkeep` para declarar la estructura.

## Documentación

- [Arquitectura](docs/architecture/architecture.md)
- [Pipeline](docs/architecture/pipeline.md)
- [Ingesta](docs/ingestion/ingesta.md)
- [Chunking](docs/chunking/current_chunking_pipeline_flow.txt)
- [Decisiones técnicas](docs/decisions/)
- [Evidencia de integración](docs/integration/)
- [Estructura de entrega](entrega/README.md)
- [Contribución](CONTRIBUTING.md)

## Equipo y contribuciones

NERV es un proyecto colaborativo. Los cambios deben respetar los contratos
compartidos, incluir evidencia proporcional al riesgo y evitar atribuir la
arquitectura global a un único subsistema o contribuyente. Consulte
[CONTRIBUTING.md](CONTRIBUTING.md) antes de proponer cambios.
