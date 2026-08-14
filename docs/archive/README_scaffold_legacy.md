# NERV

> Semantic Retrieval System for the CODEFEST AD ASTRA 2026 Challenge. Built by Team NERV.

## Objetivo

NERV tiene como objetivo construir un sistema reproducible de recuperación semántica que transforme un corpus heterogéneo en una base vectorial consultable y genere resultados en el formato requerido por CODEFEST AD ASTRA 2026.

El flujo previsto es:

```text
corpus → extracción → limpieza → chunking → embeddings → FAISS → recuperación → resultados.jsonl
```

Esta primera versión define únicamente la estructura, los contratos y las herramientas de desarrollo. La lógica de procesamiento, indexación y recuperación todavía no está implementada.

## Estructura

```text
.devcontainer/   Entorno reproducible para GitHub Codespaces
.github/         Plantillas de colaboración e integración continua
config/          Configuración de ejemplo sin secretos
corpus/          Entradas, datos procesados y consultas (sin datos reales)
docs/            Arquitectura, pipeline, equipo y decisiones técnicas
entrega/         Especificación y artefactos de la entrega oficial
notebooks/       Exploración local no productiva
outputs/         Logs, reportes y resultados generados
scripts/         Automatizaciones futuras
src/nerv/        Paquete Python y módulos del pipeline
tests/           Pruebas automatizadas
```

## Instalación local

Se requiere Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

En Windows PowerShell, active el entorno con:

```powershell
.venv\Scripts\Activate.ps1
```

Las dependencias incluyen bibliotecas de aprendizaje automático de tamaño considerable. Instálelas únicamente en un entorno apropiado.

## GitHub Codespaces

1. Mantenga el repositorio alojado de forma privada en GitHub.
2. Abra el menú **Code** del repositorio.
3. Seleccione **Codespaces** y luego **Create codespace on main**.
4. Espere a que `.devcontainer/postCreateCommand.sh` termine de preparar el entorno.

El contenedor usa Python 3.12, instala las dependencias declaradas y registra el paquete en modo editable.

## Estado actual

**Scaffolding inicial.** Se han definido la estructura del repositorio, los contratos mínimos de los módulos, la configuración de herramientas y las plantillas de colaboración. No se han incorporado corpus reales, modelos, embeddings ni índices FAISS.

## Equipo

- Miembro 1: Ingesta y preprocesamiento
- Miembro 2: Chunking y embeddings
- Miembro 3: FAISS, recuperación y entrega

## Confidencialidad

Este repositorio debe mantenerse **privado durante la competencia**. No publique el corpus, los modelos, los índices, los resultados de trabajo ni información sensible.
