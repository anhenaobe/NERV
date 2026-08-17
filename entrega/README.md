# Entrega CODEFEST Stage 1 — NERV

Esta carpeta define el paquete oficial de recuperación determinista. Stage 1
no usa agentes, modelos generativos, OpenAI, NVIDIA ni APIs externas de
razonamiento. Sólo usa `intfloat/multilingual-e5-small`, los artefactos FAISS
entregados y agregación numérica reproducible.

## Contrato obligatorio

```text
entrega/
├── resultados.jsonl
├── generador.py
├── informe_tecnico.pdf
└── base_vectorial/
    └── encoder_intfloat_multilingual-e5-small/
        ├── index.faiss
        └── metadata.jsonl
```

`index.faiss`, `metadata.jsonl` y `resultados.jsonl` son artefactos de runtime,
no sustitutos de texto ni enlaces simbólicos. Deben copiarse únicamente desde
la misma línea corregida después de que sus manifiestos, hashes, conteos y
orden interno hayan pasado las auditorías. Los artefactos locales de la línea
anterior no son una entrega corregida.

El índice obligatorio es `IndexFlatIP`, dimensión 384, con una fila por registro
de metadata. Cada registro de metadata conserva `doc_id`, `chunk_id`, `fuente`,
`formato`, `fenomeno`, `posicion`, `num_tokens` y el `texto` exacto que produjo
su vector.

## Reproducción

Desde la raíz del repositorio, una vez instaladas las dependencias y disponible
el encoder público:

```powershell
python .\entrega\generador.py --local-files-only
```

El generador:

1. carga directamente el índice y metadata entregados;
2. valida que sus cardinalidades coincidan;
3. exige las consultas ordenadas `q001` a `q050`;
4. aplica el prefijo exacto `query: ` y normalización L2;
5. ejecuta búsqueda FAISS y max pooling numérico por documento;
6. produce 3 documentos y 10 fragmentos por consulta;
7. valida el esquema antes de escribir `resultados.jsonl`.

No reconstruye embeddings ni FAISS. La presentación limita cada fragmento a
250 palabras usando únicamente texto de la metadata; no genera ni reescribe
contenido.

## Estado de publicación

El código fuente puede publicarse antes de terminar una ejecución de producción,
pero el paquete no está listo hasta que la línea corregida obtenga
`CORRECTED_CHUNKS_PASS`, termine downstream, y el PDF técnico validado de ocho
páginas o menos esté presente. Los archivos binarios grandes pueden permanecer
fuera de Git, pero deben incluirse físicamente en el paquete entregado a
CODEFEST.
