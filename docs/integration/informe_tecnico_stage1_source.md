# NERV — Informe técnico de recuperación Stage 1

## 1. Contexto y objetivo

NERV es el sistema de recuperación semántica determinista desarrollado para
CODEFEST AD ASTRA 2026 Stage 1. Su objetivo es localizar evidencia trazable en
un corpus heterogéneo y producir, para cada consulta oficial, tres documentos y
diez fragmentos ordenados. El ranking depende exclusivamente de codificación
vectorial, similitud numérica y reglas deterministas; no interviene ningún
modelo generativo.

## 2. Corpus aceptado y extracción

La entrada aceptada para la corrección final contiene 1.760 documentos JSONL,
con SHA-256
`96542c7af4245ba7eaf21b69dba7c1c84034b23931d3a938dd4a83cd8d8c2686`.
Cada registro conserva identificador estable, fuente, formato, fenómeno y texto
extraído. La ingestión admite PDF, JSON, CSV, XLSX, HTML, TXT, imágenes con OCR
opcional y PBF mediante adaptador configurado.

Los lectores son específicos por formato. PDF conserva saltos de línea y
marcadores sintéticos de página para poder distinguir envolturas de línea,
cambios de página y contenido tabular. JSON y CSV se serializan como registros
estructurales trazables; hojas de cálculo y HTML conservan orden determinista;
OCR se utiliza sólo donde el flujo de ingestión lo habilita explícitamente.

El conjunto aceptado excluye la fuente conocida de fuga de evaluación. Los
controles verifican inventario, identidad de documentos, fuentes, cardinalidad
y ausencia de contaminación antes de autorizar chunking. No se reingiere el
corpus para una corrección de segmentación.

## 3. Idiomas y segmentación

La detección de idioma guía la segmentación. Español e inglés usan reglas PySBD;
portugués usa el fallback documentado de reglas españolas porque la versión
empleada no incluye reglas portuguesas. La segmentación se ejecuta por bloques
acotados para evitar una copia completa adicional de documentos extensos.

La corrección de límites PDF trabaja en orden de fuente. Reúne únicamente pares
adyacentes cuando existen señales deterministas de continuación: minúscula o
conector léxico después de espacio continuo, envoltura de una línea entre dos
segmentos de prosa, marcador de página, rótulo corto seguido de prosa, nota de
fuente, enumeración o pie. La puntuación terminal sigue siendo autoritativa y
los campos JSON, celdas tabulares y rótulos independientes no se fusionan. Sólo
se normaliza espacio; no se inventan palabras ni puntuación.

## 4. Completitud lingüística y límites de encoder

El empaquetado usa objetivo suave de 256 tokens y solapamiento de 32 tokens en
unidades completas. El encoder admite 512 IDs; descontando los dos tokens
especiales medidos, el máximo almacenado es 510. Cada chunk se recuenta con el
tokenizer canónico antes de escribirse.

Los registros CSV/JSON y las listas, tablas, consultas o bloques estructurales
reconocidos pueden exceder una unidad natural. Se subdividen mediante el camino
jerárquico trazable y mantienen `hard_split`, identidad de unidad padre y rango
de partes. Una unidad de prosa indivisible que exceda 510 tokens falla cerrada
con `GENUINE_OVERSIZED_SENTENCE_BLOCKER`; nunca se trunca silenciosamente.

El preflight global examinó 1.709.477 unidades reconstruidas en 1.760
documentos. Recontó exactamente 42.661 candidatas: 497 superaron 510 tokens,
todas clasificadas como estructura (435 CSV, 17 JSON y 45 PDF/TXT), con cero
candidatas de prosa y cero ambiguas. La validación focalizada cubrió 184 límites
PDF conocidos, 81 continuaciones bajo espacio continuo, cinco controles
correctos, estructuras CSV/JSON y el caso de tabla que motivó el último ajuste.

La aceptación de corpus completo requiere además un artefacto nuevo, auditoría
`CORRECTED_CHUNKS_PASS`, máximo almacenado 510, cero prosa incompleta, cero
violaciones de continuidad, cero contaminación y trazas de subdivisión válidas.

## 5. Encoder y representación vectorial

El encoder público es `intfloat/multilingual-e5-small`, elegido por su cobertura
multilingüe ES/EN/PT y su uso simétrico documentado de prefijos. Los textos se
codifican con `passage: ` y las consultas con `query: `. Cada vector `float32`
tiene 384 dimensiones y se normaliza L2.

La base vectorial usa FAISS `IndexFlatIP`. En vectores normalizados, el producto
interno es equivalente al coseno. `IndexFlatIP` ejecuta búsqueda exacta, sin
entrenamiento ni aproximación del índice. La fila *i* de FAISS corresponde
exactamente a la fila *i* de `metadata.jsonl`, cuyo `texto` es el mismo texto que
produjo el vector. Manifiestos y hashes protegen esa identidad ordenada.

## 6. Recuperación, agregación y formato

Para cada consulta oficial, NERV codifica el texto con el prefijo canónico,
normaliza el vector y solicita un conjunto numérico de candidatos a FAISS. Los
fragmentos se deduplican por `chunk_id` y se ordenan por puntuación. La
agregación documental usa max pooling: la puntuación de un documento es el
máximo de sus chunks candidatos. Esta operación es numérica, estable y no usa
resúmenes ni juicios generativos.

`entrega/generador.py` carga directamente el índice y metadata entregados,
exige `q001` a `q050` en orden y produce exactamente 50 registros JSONL, tres
documentos y diez fragmentos por consulta. Cada fragmento conserva `chunk_id`,
`doc_id` y texto fuente; la presentación se limita a 250 palabras. Chunks
corregidos dentro del límite se conservan exactos. Si un fragmento de prosa
supera 250 palabras, sólo puede terminar en una frontera ya existente dentro
del texto; los datos estructurales se limitan determinísticamente sin reescribir
contenido.

## 7. Aislamiento generativo

La ruta Stage 1 —ingestión, chunking, embeddings, FAISS, consulta, agregación y
serialización— no importa ni llama agentes, OpenAI, NVIDIA ni modelos
generativos. El código experimental de agentes pertenece exclusivamente a
Stage 2 y no forma parte del ejecutable de entrega.

## 8. Reproducibilidad, artefactos y tiempos

Cada ejecución se escribe en un directorio nuevo. Los manifiestos registran
commit, configuración semántica, padres, estado por etapa, conteos, hashes y
rutas de artefactos. El downstream corregido sólo puede iniciarse si la
auditoría de chunks devuelve `CORRECTED_CHUNKS_PASS`; después debe verificar
filas/dimensión/finitud/normalización de embeddings, tipo y cardinalidad FAISS,
identidad FAISS/metadata y contrato completo de las 50 consultas.

El paquete final contiene `resultados.jsonl`, `generador.py`, este informe en
PDF, `index.faiss` y `metadata.jsonl` bajo el directorio seguro del encoder. En
el corte de publicación de este fuente técnico, la entrada validada tiene 1.760
documentos, pero la ejecución corregida de corpus completo aún no cuenta con un
manifiesto final ni auditoría de aceptación. Por esa razón no se publican
conteos de chunks, vectores, índice o resultados de una línea anterior como si
fueran finales.

El último chunking completo de referencia tardó 885,257 segundos, pero pertenece
a la línea previa no conforme y se incluye sólo como contexto operativo. No hay
tiempo medido aceptado para la línea corregida mientras su manifiesto permanezca
incompleto. Tampoco se declaran NDCG@10 ni F1@3 oficiales: las etiquetas ocultas
necesarias para esas métricas no están disponibles.
