# Phase 1: equivalencia con un único worker `spawn`

Fecha: 2026-08-11  
Repositorio: `C:\dev\NERV`  
Alcance: sólo Phase 1; no se habilitó concurrencia real.

## 1. Resumen ejecutivo

Se implementó una ruta opt-in `--workers 1` que ejecuta el algoritmo existente `_process_document` dentro de un worker persistente creado con el contexto explícito `spawn`. Cada documento se serializa en un temporal JSONL exclusivo, el worker devuelve un manifiesto compacto, el coordinador verifica tamaño/hash/confinamiento y fusiona bytes en orden antes de reutilizar la publicación atómica existente.

La salida full-corpus candidata resultó byte a byte idéntica al baseline: ambos archivos miden 309 628 718 bytes y tienen SHA-256 `461682E45B01FCF9FAEEDF69C274D09866087D9441F578C323911EF6E347ED9A`. El `validate` completo terminó con 336 239 chunks válidos y cero fallos, mismatches o excesos del hard limit.

## 2. Veredicto final

**PASS.** Se cumplieron todos los gates funcionales, estáticos y full-corpus de Phase 1. No se implementó Phase 2 ni se habilitaron workers >1.

## 3. Archivos creados

- `src/nerv/chunking/parallel_pipeline.py`
- `tests/chunking/regression/test_parallel_pipeline_phase_1.py`
- `docs/chunking/multiprocessing/phase_1_single_worker_equivalence_report.md`
- Artefactos de aceptación ignorados por Git:
  - `outputs/resultados/chunks.phase1_candidate.jsonl`
  - `outputs/resultados/chunking_phase1_candidate_metrics.json`
  - `outputs/resultados/chunking_phase1_candidate_progress.json`
  - `outputs/resultados/chunking_phase1_candidate_config.json`
  - `outputs/resultados/chunking_phase1_candidate_validation_metrics.json`
  - logs Phase 1 bajo `outputs/logs/`

El reporte previo `docs/chunking/multiprocessing_feasibility_report.md` ya existía como archivo no rastreado al comenzar esta fase y no se modificó.

## 4. Archivos modificados

- `src/nerv/chunking/real_corpus.py`: opt-in CLI, dispatch a la ruta Phase 1 y métricas agregadas.

No se modificaron `pipeline.py`, `token_counter.py`, `sentence_splitter.py`, `language_detector.py`, `chunker.py`, `hard_limit.py`, `configuration.py` ni `atomic_io.py`.

## 5. Funciones y clases agregadas o modificadas

En `parallel_pipeline.py` se agregaron:

- dataclasses `WorkerMetadata`, `WorkerDocumentResult`, `ParallelPipelineMetrics`, `ParallelPipelineRun` y estado privado `_WorkerState`;
- `configuration_fingerprint`;
- `_create_production_counter` e `initialize_worker`;
- `worker_metadata_task` y `process_document_task`;
- `_confined_document_path`, `verify_worker_result` y `merge_pending_results`;
- `_write_parallel_chunking_config`;
- `run_parallel_pipeline_phase_1`.

En `real_corpus.py` cambiaron `produce_command` y `parse_args`; el parser de `produce` ahora acepta sólo `--workers 1`. La ruta sin `--workers` continúa llamando al `run_pipeline` secuencial y conserva la inyección de `TokenCounter`/runner usada por tests.

## 6. Hashes de módulos protegidos antes y después

| Módulo | SHA-256 antes | SHA-256 después | Estado |
|---|---|---|---|
| `pipeline.py` | `0782768EDF11BF1FF9BA863124B47F1533D729D373FC0C6F33652D0E2875B1C4` | igual | Sin cambio |
| `real_corpus.py` | `2AB8C2A14FE1A788ED261A769FA972C85ADE489630C8F10F29C8DCE80A643CA5` | `D1502E162D0553E012F6D7C0B8BFC798502000FF290C7F434308B1F5217A8817` | Infraestructura CLI/dispatch/métricas |
| `token_counter.py` | `8BADE15702A2F361A2823D1D4605A30B8CCA124ED74695D278CA5395391B6B77` | igual | Sin cambio |
| `sentence_splitter.py` | `86BD8ED6FCB043EE010D865B61A55DE8B97C37DD0E3108616183840EA01DC156` | igual | Sin cambio |
| `language_detector.py` | `92C313FA44CC0E0619C11A67CB81C2E0FD483598A3FAFDBDBD9D94C6DCC6E922` | igual | Sin cambio |
| `chunker.py` | `81D655EBC397A2FC1A5D4E0686E5816FCEF7E5941475049457A3DEDCE7BBD43F` | igual | Sin cambio |
| `hard_limit.py` | `178FD3E4D3C94C5AC445403F897194D0AE6CA35978F6ACEB27F32347AB001D8D` | igual | Sin cambio |
| `configuration.py` | `8AD3B3B29DA7009B2B80245D7C3F41B4E7888B978E848FCCF99EA5E793D344E5` | igual | Sin cambio |
| `atomic_io.py` | `CAD0F94AB4797107A877AF5D9BE1D12EEC3A2F28A0B54B5FAD26EF4FFB390475` | igual | Sin cambio |
| Nuevo `parallel_pipeline.py` | No existía | `39EC3F4D0377BCF18A443F1D8F5AACD47CB25E38DE833EA1AE7AB79137DC3B07` | Nuevo |

La función algorítmica de referencia `_process_document` está en `pipeline.py`, cuyo hash no cambió.

## 7. Diagrama de arquitectura Phase 1

```text
MAIN
  read_documents -> validate_document -> document_index
       |
       | pool.apply (exactamente una tarea activa)
       v
SPAWN WORKER (persistente, único)
  initialize_worker una vez
  -> _process_document existente
  -> json.dumps exacto + UTF-8 + LF
  -> fsync temporal por documento
  -> WorkerDocumentResult compacto
       |
       v
MAIN
  pending[index] -> verificar path/tamaño/SHA
  -> merge binario en next_merge_index
  -> progress sólo después de merge
  -> eliminar temporal por documento
  -> fsync chunks.global.tmp
  -> _publish_completed_file
```

## 8. Diseño Windows `spawn`

La implementación usa `multiprocessing.get_context("spawn")` explícitamente y crea `Pool(processes=1)`. Initializer y tareas son funciones top-level importables. No se envían closures, tokenizer, PySBD, loggers ni archivos abiertos. El guard de CLI existente evita reentrada bajo import del child. La prueba de integración ejecutó un worker `spawn` real en Windows.

## 9. Comportamiento del initializer

`initialize_worker` copia la configuración, verifica su fingerprint, crea y verifica un único `TokenCounter` mediante `_create_production_counter`, carga marcadores de idioma, precalienta backends PySBD `es`/`en` y guarda estado privado por proceso. El tokenizer no se crea por documento ni se recibe del padre.

## 10. Prueba de una inicialización del tokenizer

`test_initializer_reuses_one_counter_for_multiple_tasks` usa un factory controlado con contador: dos tareas consecutivas producen manifiestos con el mismo PID y el factory registra exactamente una llamada. La corrida full-corpus informó `workers_initialized=1`, `worker_task_count=1761` e inicialización de 15,1718 s.

## 11. Contrato de la tarea worker

Entrada: índice, documento validado, ruta del directorio de documentos y fingerprint. La tarea comprueba estado/fingerprint, llama directamente a `_process_document`, serializa records, hace flush/fsync, calcula SHA y retorna el manifiesto. En errores borra su temporal incompleto y propaga la excepción. No retorna texto, sentences, chunks, records, tokenizer ni PySBD.

## 12. Esquema del manifiesto compacto

`WorkerDocumentResult`, schema 1, contiene identidad/índice/formato, ruta, bytes/SHA, conteos de records/sentences/tokens/oversized/hard split, mínimos/máximos, idioma, tiempos acotados, PID, caracteres/líneas y escalares bounded. Sus campos no incluyen payload del documento ni listas de contenido.

## 13. Estructura temporal

```text
C:\nerv_work\produce-<uuid>\
  documents\
    <index>-<hash-doc-id>-<uuid>.jsonl.tmp
  chunks.global.tmp
```

El directorio de corrida y nombres son únicos. Los temporales se crean con `xb`; su path resuelto debe tener exactamente como padre el directorio `documents`. El worker posee el archivo hasta retornar; después el coordinador lo verifica/fusiona/elimina. Al éxito no quedó directorio de corrida.

## 14. Merge ordenado

`merge_pending_results` usa `pending` y `next_merge_index`. Sólo consume índices canónicos consecutivos, verifica cada archivo antes de copiarlo y llama `on_merged` después de la copia. El test entrega primero el índice 2, comprueba que no avanza, entrega 1 y obtiene bytes `1,2`. Con workers=1 el mismo mecanismo ya está operativo sin concurrencia.

## 15. Serialización e identidad de bytes

El worker reutiliza exactamente `json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"`, codificado UTF-8. El merge usa `shutil.copyfileobj` binario; no parsea, reserializa ni normaliza saltos. Los records provienen de la misma `_process_document` sin modificación.

## 16. Publicación atómica

El coordinador construye y fsync el temporal global, calcula su hash y llama a `_publish_completed_file`, el mismo helper del pipeline. Cualquier error previo evita el replace y conserva la salida publicada anterior. El test de fallo controlado mantiene intacto un sentinel. La escritura del config ocurre después de publicar chunks, igual que la ruta secuencial existente.

## 17. Cambios CLI

- Sin `--workers`: ruta secuencial sin cambios.
- `--workers 1 --work-dir <ruta>`: ruta Phase 1.
- `--workers 2`, `4` o cualquier valor distinto de 1: error de parser explícito, pues la concurrencia pertenece a una fase futura.
- Phase 1 exige `--work-dir` para alojar temporales de forma deliberada.

## 18. Tests agregados

Quince tests cubren initializer/reuso y fallo, bytes y hard split, manifiesto compacto/SHA/identidad/conteo, merge fuera de orden, confinamiento, aliases, output anidado, conteo total inválido, subprocess `spawn` con equivalencia, preservación ante fallo y validación CLI de workers/work-dir. No se usó el corpus completo dentro de pytest.

## 19. Resultado pytest

Comando: `.\.venv\Scripts\python.exe -m pytest tests/chunking -q`  
Resultado final después de la remediación post-auditoría: **240 passed, 2 skipped in 61.51s**. Los skips existentes requieren `NERV_TEST_ENCODER_MODEL`; la prueba real Phase 1 tiene su propio subprocess spawn controlado.

## 20. Resultado Ruff

Comando: `.\.venv\Scripts\python.exe -m ruff check src tests scripts`  
Resultado: **All checks passed**.

## 21. Resultado mypy

Comando: `.\.venv\Scripts\python.exe -m mypy --no-incremental`  
Resultado: **Success: no issues found in 37 source files**.

## 22. Equivalencia representativa

El test `test_spawn_worker_pipeline_is_byte_identical_to_sequential` procesa inglés, español y un CSV de 900 caracteres que activa hard split. Compara la salida de `run_pipeline` con `run_parallel_pipeline_phase_1` y exige bytes/SHA iguales, un worker, `spawn`, una inicialización y tres tareas. Resultado: PASS.

Este test y la suite completa se repitieron después de la remediación de auditoría. La única corrida full-corpus se ejecutó antes de esas correcciones, conforme al límite de una corrida. Los cambios posteriores se limitaron a failure handling, preflight, verificación adicional de manifiesto, métricas y logging; no tocaron `_process_document`, serialización ni merge binario. Por ello no se repitió —ni se permitió repetir— el full produce.

## 23. SHA secuencial de referencia

`461682E45B01FCF9FAEEDF69C274D09866087D9441F578C323911EF6E347ED9A`

## 24. SHA candidato Phase 1

`461682E45B01FCF9FAEEDF69C274D09866087D9441F578C323911EF6E347ED9A`

## 25. Comparación SHA exacta

**IGUALES.** También coinciden los tamaños: 309 628 718 bytes en ambos archivos.

## 26. Conteo de chunks candidato

336 239, igual al baseline.

## 27. Conteos hard split

- Unidades fuente: 447.
- Chunks generados: 964.
- Iguales al baseline.

## 28. Máximo de tokens

512; se preservaron soft limit 256, hard limit 512 y overlap 32.

## 29. Resultado `validate`

Comando sobre el candidato separado: `.\.venv\Scripts\python.exe -m nerv.chunking.real_corpus validate --local-files-only --input outputs\resultados\documentos.jsonl --chunks outputs\resultados\chunks.phase1_candidate.jsonl --metrics-output outputs\resultados\chunking_phase1_candidate_validation_metrics.json --log-path outputs\logs\chunking_phase1_candidate_validate.log`.

Resultado: completed, 336 239 válidos, 0 inválidos, `failure_count=0`, `token_count_mismatch_count=0`, `encoder_hard_limit_exceeded_count=0`, `chunks_gt_hard_limit=0`, máximo 512. Duración diagnóstica: 148,5138 s.

## 30. Runtime diagnóstico workers=1

`processing_duration_seconds=1485,0865` (24,75 min), incluyendo inicialización, procesamiento, temporales, merge y publicación. No se interpreta como benchmark ni se compara para decidir scaling.

## 31. Bytes temporales escritos

`worker_temp_bytes_total=309628718`. El merge ordenado acumuló 44,3921 s; el servicio worker sumó 1392,2633 s. `peak_pending_documents=1`; el mayor temporal pendiente fue 76 409 002 bytes.

## 32. Errores encontrados y remediación

- Ruff detectó inicialmente dos imports/line-length y un import faltante `asdict`; se corrigieron antes de la suite completa.
- La auditoría `reviewer` encontró que una excepción lanzada directamente por el initializer de `Pool` podía provocar recreación repetida del worker. El initializer ahora captura un envelope de error acotado y `worker_metadata_task` lo propaga; una prueba subprocess confirma cierre y preservación del output.
- La misma auditoría detectó preflight incompleto, ausencia de creación del parent de output, vinculación débil del manifiesto y métricas de fallo inicializadas en cero. Se añadieron validaciones equivalentes a la ruta secuencial, parents anidados, comprobación índice/doc_id/contador de líneas, rechazo de duplicados y snapshots de arquitectura antes de persistir progreso.
- El mismo `reviewer` volvió a auditar la superficie remediada y confirmó que no quedan hallazgos bloqueantes ni materiales sin resolver.
- Las primeras lecturas del test completo devolvieron output parcial porque el proceso seguía activo; se continuó la misma sesión hasta obtener exit 0 y resumen final, sin relanzar trabajo funcional.
- Durante produce/validate aparecieron advertencias existentes de secuencias >512 previas al hard split, detección ambigua y fallback portugués a español. El output emitido mantuvo máximo 512 y SHA exacto; no se modificó ni silenció esa conducta.
- No hubo fallos de worker, manifiesto, merge, publicación o validación en full-corpus.

## 33. Riesgos restantes

- Phase 1 no implementa cancelación/terminación avanzada de Phase 3.
- Logging de progreso global es propiedad del coordinador; los warnings internos del child aún usan la configuración básica disponible bajo spawn. QueueHandler/QueueListener queda para una fase posterior.
- La ruta sólo admite una tarea activa; no prueba RAM, backpressure ni reordenamiento real con múltiples procesos.
- La salida candidata y sus métricas ocupan espacio adicional y se conservan como evidencia.
- La escritura de config posterior a chunks conserva la misma ventana de fallo tardío que el pipeline secuencial.

Ninguno de estos riesgos produjo una diferencia funcional en Phase 1.

## 34. Autorización de Phase 2

**Sí, Phase 2 queda técnicamente autorizada por los gates de equivalencia de Phase 1, pero no fue iniciada ni implementada.** Debe ejecutarse como una tarea separada con sus propios límites, pruebas de concurrencia, memoria y error handling. Esta tarea termina aquí con un único worker.
