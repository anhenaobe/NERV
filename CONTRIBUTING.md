# Contribuir a NERV

## Flujo de trabajo

No trabaje directamente sobre `main`. Cree una rama a partir de una versión actualizada y abra un pull request para integrar cada cambio.

Ramas principales de trabajo:

- `feature/ingestion`
- `feature/chunking-embeddings`
- `feature/faiss-retrieval`

Los cambios deben ser pequeños, revisables y estar relacionados con la responsabilidad de la rama. Toda integración en `main` debe pasar por un pull request.

## Convención de commits

Use un prefijo que describa la intención:

- `feat:` nueva funcionalidad
- `fix:` corrección de un defecto
- `docs:` documentación
- `test:` pruebas
- `refactor:` reestructuración sin cambio funcional

Ejemplo:

```text
feat: add PDF reader contract
```

## Validación

Antes de fusionar:

```bash
ruff check .
pytest
```

Corrija los errores y no introduzca advertencias nuevas en los archivos modificados.

## Datos y artefactos prohibidos

No suba al repositorio:

- el corpus real o consultas privadas;
- modelos o pesos pesados;
- índices FAISS;
- embeddings o matrices NumPy;
- logs y resultados generados;
- credenciales, secretos o configuración local;
- el informe o los resultados finales antes de la entrega autorizada.
