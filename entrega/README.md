# Estructura oficial de entrega

La entrega de CODEFEST AD ASTRA 2026 debe conservar exactamente esta organización:

```text
entrega/
├── resultados.jsonl
├── generador.py
├── informe_tecnico.pdf
├── README
├── base_vectorial/
│   └── encoder_<nombre>/
│       ├── index.faiss
│       └── metadata.jsonl
└── grafo/
    └── grafo.graphml
```

## Reglas

- `resultados.jsonl`, `generador.py`, `informe_tecnico.pdf` y `README` van en la raíz de `entrega/`.
- No se deben modificar los nombres ni las ubicaciones obligatorias.
- `grafo/` solo se incluye si el equipo implementa el bonus.
- El `generador.py` de la entrega debe contener comentarios que expliquen su flujo.
- El `README` final debe explicar los requisitos, comandos y pasos exactos para reproducir los resultados.
- `index.faiss` y `metadata.jsonl` deben pertenecer al mismo encoder y mantener correspondencia posicional.

Los archivos `.gitkeep` actuales reservan carpetas; no representan artefactos de entrega.
