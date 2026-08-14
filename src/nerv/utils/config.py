from pathlib import Path
from typing import Any

import yaml

_REQUIRED_SECTIONS = (
    "semantic_contract",
    "paths",
    "runtime",
    "faiss",
    "retrieval",
)


def load_config(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de configuración: {path}. "
            "Copia config/config.example.yaml a config/config.yaml y ajústalo."
        )

    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(
            f"El archivo de configuración está vacío o mal formado: {path}"
        )

    missing = [section for section in _REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(
            f"Faltan secciones obligatorias en {path}: {', '.join(missing)}."
        )

    return config
