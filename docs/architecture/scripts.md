# Inventario de scripts

La consolidación conserva los scripts en una carpeta plana para evitar cambios
gratuitos en comandos e imports. Su clasificación es:

| Script | Clasificación | Propósito |
|---|---|---|
| `run_e2e.ps1` | PRODUCTION_ENTRYPOINT | Ejecuta los modos E2E aislados con el entorno CUDA y contratos de identidad. |
| `setup_cuda_env.ps1` | ENVIRONMENT_VALIDATION | Verifica un entorno CUDA ya preparado; no instala dependencias desde red. |
| `verify_cuda_env.py` | ENVIRONMENT_VALIDATION | Comprueba Torch/CUDA/GPU y un smoke local acotado del encoder. |
| `validar_documentos.py` | PRODUCTION_HELPER | Valida el contrato de `documentos.jsonl`. |
| `preparar_para_colab.py` | MANUAL_WORKFLOW | Prepara artefactos limpios para el flujo manual. |
| `paso_final_local.py` | MANUAL_WORKFLOW | Ejecuta recuperación y salida local con artefactos existentes. |
| `generador_con_tiempos.py` | DIAGNOSTIC | Instrumenta tiempos del flujo de generación. |
| `comparar_agregacion.py` | EVALUATION | Compara estrategias de agregación documental. |
| `prueba_rapida.py` | DIAGNOSTIC | Smoke local pequeño con fixtures versionados. |

Los evaluadores y diagnósticos propios de chunking permanecen bajo
`tests/chunking` porque importan fixtures y helpers de prueba de ese subsistema.
