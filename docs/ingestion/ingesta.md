# Modulo de ingesta

El modulo de ingesta corresponde al Miembro 1 del proyecto NERV para CODEFEST AD ASTRA 2026. Su responsabilidad termina en producir `documentos.jsonl` y `errores.jsonl` con documentos limpios, identificados y listos para que el Miembro 2 los consuma linea por linea.

No implementa chunking, `chunk_id`, embeddings, indices vectoriales, recuperacion ni generacion final.

## Contrato de salida

Cada linea de `documentos.jsonl` es un objeto JSON UTF-8 con:

- `doc_id`: identificador estable con prefijo `DOC-`.
- `fuente`: ruta relativa al corpus, en formato POSIX.
- `formato`: `pdf`, `html`, `txt`, `md`, `json`, `csv`, `xlsx`, `imagen` o `pbf`.
- `fenomeno`: entero `1`, `2` o `3` segun carpetas `F1_`, `F2_`, `F3_`.
- `idioma`: `es`, `en`, `pt` o `desconocido`.
- `texto`: contenido textual limpio y no vacio.

## Identificacion documental

`doc_id` se calcula con SHA-256 a partir de:

1. la ruta relativa al corpus normalizada en minusculas con `as_posix()`;
2. el contenido binario del archivo leido por bloques.

La ruta forma parte del hash para que dos archivos con contenido identico conserven identidad documental distinta si provienen de fuentes diferentes.

## OCR

PDF siempre intenta primero extraccion normal con PyMuPDF. OCR solo se ejecuta si todas las paginas carecen de texto o si un PDF mixto tiene paginas puntuales sin texto. Las paginas se procesan una por una a 220 DPI.

Imagenes pasan por OCR local con Tesseract. El modulo no inventa descripciones visuales; si una imagen no contiene texto util se registra como vacia.

Idiomas OCR configurados: `spa`, `eng`, `por`, `chi_sim`.

## PBF

Los `.pbf` del corpus se tratan como teselas Mapbox Vector Tile por su ubicacion en rutas `tiles/<zoom>/<x>/<archivo>.pbf` y por la estructura de capas/features inspeccionada. Se extraen propiedades semanticas y se omite geometria.

La deduplicacion usa un set global de fingerprints durante el procesamiento:

- `capa + id` si el feature trae identificador estable;
- `capa + propiedades normalizadas` si no trae id.

La geometria no participa en el fingerprint para evitar variaciones entre zooms.
