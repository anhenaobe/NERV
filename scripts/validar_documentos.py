import argparse
import json
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]


CAMPOS_OBLIGATORIOS = {
    "doc_id",
    "fuente",
    "formato",
    "fenomeno",
    "texto"
}

FORMATOS_PERMITIDOS = {
    "pdf",
    "html",
    "txt",
    "md",
    "json",
    "csv",
    "xlsx",
    "imagen",
    "pbf"
}

IDIOMAS_PERMITIDOS = {
    "es",
    "en",
    "pt",
    "desconocido"
}


def validar_documento(documento, numero_linea, ids_encontrados):
    errores = []

    if not isinstance(documento, dict):
        errores.append(
            f"Línea {numero_linea}: el contenido no es un objeto JSON"
        )
        return errores

    campos_faltantes = (
        CAMPOS_OBLIGATORIOS - documento.keys()
    )

    if campos_faltantes:
        errores.append(
            f"Línea {numero_linea}: faltan campos obligatorios: "
            f"{sorted(campos_faltantes)}"
        )

    doc_id = documento.get("doc_id")

    if not isinstance(doc_id, str) or not doc_id.strip():
        errores.append(
            f"Línea {numero_linea}: doc_id inválido"
        )

    elif not doc_id.startswith("DOC-"):
        errores.append(
            f"Línea {numero_linea}: "
            f"doc_id no comienza con 'DOC-'"
        )

    elif doc_id in ids_encontrados:
        errores.append(
            f"Línea {numero_linea}: "
            f"doc_id duplicado: {doc_id}"
        )

    else:
        ids_encontrados.add(doc_id)

    fuente = documento.get("fuente")

    if not isinstance(fuente, str) or not fuente.strip():
        errores.append(
            f"Línea {numero_linea}: fuente inválida"
        )

    elif Path(fuente).is_absolute():
        errores.append(
            f"Línea {numero_linea}: "
            "la fuente no debe ser una ruta absoluta"
        )

    formato = documento.get("formato")

    if formato not in FORMATOS_PERMITIDOS:
        errores.append(
            f"Línea {numero_linea}: "
            f"formato no permitido: {formato}"
        )

    fenomeno = documento.get("fenomeno")

    if fenomeno not in {1, 2, 3}:
        errores.append(
            f"Línea {numero_linea}: "
            f"fenómeno inválido: {fenomeno}"
        )

    texto = documento.get("texto")

    if not isinstance(texto, str):
        errores.append(
            f"Línea {numero_linea}: texto no es una cadena"
        )

    elif not texto.strip():
        errores.append(
            f"Línea {numero_linea}: texto vacío"
        )

    idioma = documento.get("idioma")

    if idioma is not None and idioma not in IDIOMAS_PERMITIDOS:
        errores.append(
            f"Línea {numero_linea}: "
            f"idioma no permitido: {idioma}"
        )

    metadata = documento.get("metadata")

    if metadata is not None and not isinstance(metadata, dict):
        errores.append(
            f"Línea {numero_linea}: metadata debe ser un objeto"
        )

    return errores


def validar_jsonl(ruta_archivo):
    errores = []
    ids_encontrados = set()
    documentos_validos = 0
    total_lineas = 0

    if not ruta_archivo.exists():
        print(
            f"Error: no existe el archivo {ruta_archivo}"
        )
        return

    if not ruta_archivo.is_file():
        print(
            f"Error: la ruta {ruta_archivo} no es un archivo"
        )
        return

    with open(
        ruta_archivo,
        encoding="utf-8"
    ) as archivo:

        for numero_linea, linea in enumerate(
            archivo,
            start=1
        ):
            total_lineas += 1
            linea = linea.strip()

            if not linea:
                errores.append(
                    f"Línea {numero_linea}: línea vacía"
                )
                continue

            try:
                documento = json.loads(linea)

            except json.JSONDecodeError as error:
                errores.append(
                    f"Línea {numero_linea}: JSON inválido: {error}"
                )
                continue

            errores_documento = validar_documento(
                documento,
                numero_linea,
                ids_encontrados
            )

            if errores_documento:
                errores.extend(errores_documento)
            else:
                documentos_validos += 1

    print("\nResultado de la validación")
    print(f"Archivo revisado: {ruta_archivo}")
    print(f"Total de líneas: {total_lineas}")
    print(f"Documentos válidos: {documentos_validos}")
    print(f"Errores encontrados: {len(errores)}")

    if errores:
        print("\nDetalle de errores:")

        for error in errores:
            print(f"- {error}")

    else:
        print(
            "\nEl archivo documentos.jsonl "
            "cumple todas las validaciones."
        )


def construir_parser():
    parser = argparse.ArgumentParser(
        description="Valida el archivo documentos.jsonl del modulo de ingesta."
    )

    parser.add_argument(
        "ruta",
        nargs="?",
        default=str(
            RAIZ_PROYECTO / "corpus" / "processed" / "documentos.jsonl"
        ),
        help="Ruta al archivo documentos.jsonl."
    )

    return parser


def main(argv=None):
    argumentos = construir_parser().parse_args(argv)
    ruta_documentos = Path(argumentos.ruta).expanduser().resolve()

    validar_jsonl(ruta_documentos)


if __name__ == "__main__":
    main()
