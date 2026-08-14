# NERV - Modulo de ingesta documental

Modulo de ingesta del proyecto NERV para CODEFEST AD ASTRA 2026. Corresponde al rol del Miembro 1: recorrer el corpus, extraer texto util, limpiar y normalizar contenido, identificar documentos, detectar fenomeno e idioma, registrar incidencias y generar JSON Lines consumible por el siguiente paso del pipeline.

Este modulo no implementa chunking, `chunk_id`, conteo final de tokens, embeddings, FAISS, recuperacion, `resultados.jsonl` ni generacion de respuestas.

## Estructura

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   └── lector_corpus.py
├── scripts/
│   └── validar_documentos.py
├── tests/
│   └── test_ingesta.py
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── docs/
│   └── ingesta.md
└── datos_procesados/
    ├── lector.py
    └── validar_documentos.py
```

`datos_procesados/lector.py` y `datos_procesados/validar_documentos.py` son wrappers de compatibilidad para comandos antiguos. El codigo fuente principal vive en `src/` y `scripts/`.

## Formatos soportados

- `.pdf`
- `.json`
- `.csv`
- `.xlsx`
- `.txt`
- `.md`
- `.html`, `.htm`
- `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.webp`, `.avif`
- `.pbf`

Normalizacion de `formato`:

- `.html` y `.htm` -> `html`
- imagenes -> `imagen`
- `.pbf` -> `pbf`

## Corpus esperado

El fenomeno se detecta desde carpetas cuyo nombre comienza por:

- `F1_` -> fenomeno `1`
- `F2_` -> fenomeno `2`
- `F3_` -> fenomeno `3`

Ejemplo:

```text
corpus_prueba/
├── F1_IA_Y_Capacidades_Estrategicas/
├── F2_Seguridad_Entorno_Espacial/
└── F3_Dinamicas_Territoriales/
```

## Salida

El lector genera:

- `data/processed/documentos.jsonl`
- `data/processed/errores.jsonl`

Cada linea de `documentos.jsonl` es un JSON UTF-8 con estos campos:

```json
{
  "doc_id": "DOC-<id_estable>",
  "fuente": "ruta/relativa/del/archivo",
  "formato": "pdf|html|txt|md|json|csv|xlsx|imagen|pbf",
  "fenomeno": 1,
  "idioma": "es|en|pt|desconocido",
  "texto": "Texto limpio y normalizado.",
  "metadata": {
    "procedencia": {
      "urls": ["https://fuente.example/documento"]
    }
  }
}
```

`metadata` es opcional. Las URLs encontradas en archivos JSON se conservan en
`metadata.procedencia.urls` para trazabilidad, sin depender de mezclarlas con el
texto que se indexara. `fuente` siempre es relativa al corpus y usa separadores
POSIX (`/`). Por defecto, `doc_id` usa SHA-256 sobre ruta relativa, tamano y
fecha de modificacion para evitar una segunda lectura binaria completa; con
`--doc-id-contenido` usa ruta relativa y contenido binario completo.

Cada incidencia en `errores.jsonl` tiene:

```json
{
  "archivo": "ruta/relativa",
  "estado": "omitido|vacío|sin_texto_util|pendiente_ocr|fallido",
  "error": "mensaje entendible",
  "metadata": {
    "imagen": {
      "ancho": 1200,
      "alto": 800,
      "modo": "RGB"
    }
  }
}
```

Para imagenes sin texto OCR util, el lector no inventa descripciones visuales:
registra la incidencia como `sin_texto_util` y conserva metadata tecnica de la
imagen. Los indices o tablas de contenido de PDFs se conservan como parte del
mismo PDF leido, no como documentos independientes.

## Dependencias Python

```text
beautifulsoup4
openpyxl
PyMuPDF
Pillow
pytesseract
pytest
```

`mapbox-vector-tile` no es requerido: el proyecto usa un decodificador interno ligero para teselas Mapbox Vector Tile y extrae solo propiedades semanticas.

## Dependencias externas

OCR requiere Tesseract instalado en el sistema. Idiomas usados:

- `spa` para espanol
- `eng` para ingles
- `por` para portugues
- `chi_sim` para chino simplificado

En Windows, este workspace detecta automaticamente:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
tessdata\eng.traineddata
tessdata\spa.traineddata
tessdata\por.traineddata
tessdata\chi_sim.traineddata
```

Los modelos `tessdata/*.traineddata` se ignoran en Git porque son archivos descargados.

## Instalacion

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Instalar Tesseract en Windows:

```powershell
winget install -e --id UB-Mannheim.TesseractOCR
```

Verificar idiomas:

```powershell
& 'C:\Program Files\Tesseract-OCR\tesseract.exe' --list-langs --tessdata-dir tessdata
```

## Ejecucion

Con rutas por defecto:

```powershell
.venv\Scripts\python.exe src\lector_corpus.py
```

Con rutas explicitas:

```powershell
.venv\Scripts\python.exe src\lector_corpus.py --corpus corpus_prueba --salida data\processed
```

Por defecto la ingesta lee varios archivos en paralelo para reducir el tiempo
total sin cambiar el texto extraido. En este equipo el valor por defecto es 6.
Para evitar caidas de librerias nativas, PDF e imagenes se leen en procesos
separados, con 2 procesos pesados por defecto. TXT, MD, CSV, JSON, HTML y XLSX
pueden leerse en paralelo con hilos. Si el equipo se queda corto de CPU o RAM,
se puede ajustar:

```powershell
.venv\Scripts\python.exe src\lector_corpus.py --corpus corpus_prueba --salida data\processed --trabajadores 6 --procesos-pesados 2
```

Use `--trabajadores 1 --procesos-pesados 1` para volver al modo mas
conservador.

La extraccion usa cache persistente en `data/cache/lecturas.jsonl`. Si un
archivo conserva ruta, tamano y fecha de modificacion, se reutiliza el texto ya
extraido. Para forzar lectura completa:

```powershell
.venv\Scripts\python.exe src\lector_corpus.py --sin-cache
```

Por velocidad, la corrida normal empieza sin OCR para extraer texto nativo de
PDFs. Si quedan PDFs escaneados como `pendiente_ocr`, el programa ejecuta una
segunda fase automaticamente para resolver solo esos archivos con OCR. Para
hacer OCR desde la primera pasada:

```powershell
.venv\Scripts\python.exe src\lector_corpus.py --ocr
```

Para ejecutar solo la pasada rapida y dejar pendientes OCR:

```powershell
.venv\Scripts\python.exe src\lector_corpus.py --sin-ocr
```

Si ya existe una corrida rapida previa, se puede procesar solo lo pendiente de
OCR y fusionarlo con `documentos.jsonl`:

```powershell
.venv\Scripts\python.exe src\lector_corpus.py --solo-pendientes-ocr
```

Los archivos `.pbf` se interpretan como tiles Mapbox Vector Tile. El lector
deduplica features entre tiles para no repetir entidades territoriales. Cuando
un tile es valido pero solo contiene features ya vistas, se guarda un resumen
compacto en vez de marcarlo como vacio. Para revisar un PBF puntual:

```powershell
.venv\Scripts\python.exe src\lector_corpus.py --diagnosticar-pbf corpus_prueba\ruta\archivo.pbf
```

Para acelerar la primera corrida, el `doc_id` usa por defecto ruta, tamano y
fecha de modificacion, evitando una segunda lectura binaria completa de cada
archivo grande. Si se necesita el modo historico basado en contenido completo:

```powershell
.venv\Scripts\python.exe src\lector_corpus.py --doc-id-contenido
```

## Validacion

```powershell
.venv\Scripts\python.exe scripts\validar_documentos.py
```

Tambien acepta una ruta explicita:

```powershell
.venv\Scripts\python.exe scripts\validar_documentos.py data\processed\documentos.jsonl
```

## Pruebas

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Las pruebas unitarias de OCR usan mocks; no dependen de ejecutar Tesseract real.

## Flujo

1. Recorre el corpus recursivamente con orden determinista.
2. Ignora basura del sistema y omite auxiliares conocidos.
3. Detecta fenomeno desde carpetas `F1_`, `F2_`, `F3_`.
4. Extrae texto segun formato, en paralelo cuando el formato lo permite.
5. Usa OCR solo como fallback para PDF o para imagenes.
6. Limpia texto y detecta idioma.
7. Genera `doc_id` reproducible.
8. Escribe JSON Lines y registra incidencias.
9. Valida contrato para el Miembro 2.

## Limitaciones conocidas

- OCR no describe imagenes: solo recupera texto visible.
- AVIF depende del soporte disponible en Pillow.
- PDF protegido con contrasena se registra como fallido.
- El detector de idioma es ligero y solo clasifica `es`, `en`, `pt` o `desconocido`.
- Si faltan Tesseract o idiomas OCR, el archivo afectado se registra y el corpus continua.

## Integracion con el Miembro 2

El Miembro 2 debe leer `data/processed/documentos.jsonl` linea por linea. No necesita renombrar campos ni transformar la estructura basica del documento.
