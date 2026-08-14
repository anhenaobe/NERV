from pathlib import Path


def resolve_project_path(project_root: Path, relative_path: Path) -> Path:

    project_root = Path(project_root).resolve()
    relative_path = Path(relative_path)

    if relative_path.is_absolute():
        raise ValueError(
            "Se requiere una ruta relativa al proyecto, se recibió una ruta "
            f"absoluta: {relative_path}"
        )

    resolved = (project_root / relative_path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        raise ValueError(
            f"La ruta resuelta queda fuera del proyecto: {resolved}"
        ) from None
    return resolved
