from pathlib import Path

import yaml


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# CONFIG FILES
# ============================================================

LOCAL_CONFIG = (
    PROJECT_ROOT
    / "config.local.yaml"
)

EXAMPLE_CONFIG = (
    PROJECT_ROOT
    / "config.example.yaml"
)


if LOCAL_CONFIG.exists():

    CONFIG_FILE = LOCAL_CONFIG

elif EXAMPLE_CONFIG.exists():

    CONFIG_FILE = EXAMPLE_CONFIG

else:

    raise FileNotFoundError(
        "No se encontró config.local.yaml "
        "ni config.example.yaml en la raíz del proyecto."
    )


# ============================================================
# LOAD CONFIG
# ============================================================

with open(
    CONFIG_FILE,
    "r",
    encoding="utf-8"
) as f:

    CONFIG = yaml.safe_load(f)


if CONFIG is None:

    raise ValueError(
        f"El archivo de configuración está vacío: "
        f"{CONFIG_FILE}"
    )


if not isinstance(
    CONFIG,
    dict
):

    raise ValueError(
        "El archivo YAML debe contener "
        "un objeto/diccionario en el nivel raíz."
    )


# ============================================================
# VALIDATE CONFIG STRUCTURE
# ============================================================

required_sections = [
    "paths",
    "datasets",
    "outputs",
]

for section in required_sections:

    if section not in CONFIG:

        raise ValueError(
            f"Falta la sección requerida "
            f"'{section}' en {CONFIG_FILE.name}."
        )


# -----------------------------
# paths
# -----------------------------

required_path_keys = [
    "data_root",
]

for key in required_path_keys:

    if key not in CONFIG["paths"]:

        raise ValueError(
            f"Falta paths.{key} "
            f"en {CONFIG_FILE.name}."
        )


# -----------------------------
# datasets
# -----------------------------

if "behsof" not in CONFIG["datasets"]:

    raise ValueError(
        "Falta datasets.behsof "
        f"en {CONFIG_FILE.name}."
    )


if "byra" not in CONFIG["datasets"]:

    raise ValueError(
        "Falta datasets.byra "
        f"en {CONFIG_FILE.name}."
    )


required_behsof_keys = [
    "images_dir",
    "demo_csv",
    "images_csv",
]

for key in required_behsof_keys:

    if key not in CONFIG["datasets"]["behsof"]:

        raise ValueError(
            f"Falta datasets.behsof.{key} "
            f"en {CONFIG_FILE.name}."
        )


required_byra_keys = [
    "mat_file",
]

for key in required_byra_keys:

    if key not in CONFIG["datasets"]["byra"]:

        raise ValueError(
            f"Falta datasets.byra.{key} "
            f"en {CONFIG_FILE.name}."
        )


# -----------------------------
# outputs
# -----------------------------

required_output_keys = [
    "inventory_csv",
    "quality_audit_csv",
    "tal_contact_sheet_index",
    "beh_sample_review_csv",
    "beh_outliers_review_csv",
    "manifest_csv",
    "patient_folds_csv",
    "tal_contact_sheets_dir",
    "beh_sample_contact_sheets_dir",
    "beh_outlier_review_dir",
]

for key in required_output_keys:

    if key not in CONFIG["outputs"]:

        raise ValueError(
            f"Falta outputs.{key} "
            f"en {CONFIG_FILE.name}."
        )


# ============================================================
# PATH HELPERS
# ============================================================

def resolve_path(
    base,
    value
):
    """
    Resuelve una ruta.

    - Si value es absoluta, la conserva.
    - Si value es relativa, la resuelve respecto a base.
    - Expande '~' cuando corresponda.
    """

    if value is None:

        raise ValueError(
            "Se intentó resolver una ruta "
            "con valor None."
        )


    path = Path(
        str(value)
    ).expanduser()


    if path.is_absolute():

        return path.resolve()


    return (
        base
        / path
    ).resolve()


def project_path(
    value
):
    """
    Resuelve una ruta relativa a la raíz
    del repositorio tesis-xai-steatosis.
    """

    return resolve_path(
        PROJECT_ROOT,
        value
    )


# ============================================================
# DATA ROOT
# ============================================================

DATA_ROOT = project_path(
    CONFIG[
        "paths"
    ][
        "data_root"
    ]
)


if not DATA_ROOT.exists():

    raise FileNotFoundError(
        "No existe la carpeta raíz de datos:\n"
        f"{DATA_ROOT}\n\n"
        "Revisa paths.data_root en "
        f"{CONFIG_FILE.name}."
    )


if not DATA_ROOT.is_dir():

    raise NotADirectoryError(
        "DATA_ROOT existe, pero no es una carpeta:\n"
        f"{DATA_ROOT}"
    )


def data_path(
    value
):
    """
    Resuelve una ruta relativa a
    tesis-xai-steatosis-data.
    """

    return resolve_path(
        DATA_ROOT,
        value
    )


# ============================================================
# BEHSOF INPUT PATHS
# ============================================================

BEHSOF_IMAGES_DIR = data_path(
    CONFIG[
        "datasets"
    ][
        "behsof"
    ][
        "images_dir"
    ]
)


BEHSOF_DEMO_CSV = data_path(
    CONFIG[
        "datasets"
    ][
        "behsof"
    ][
        "demo_csv"
    ]
)


BEHSOF_IMAGES_CSV = data_path(
    CONFIG[
        "datasets"
    ][
        "behsof"
    ][
        "images_csv"
    ]
)


# ============================================================
# BYRA INPUT PATH
# ============================================================

BYRA_MAT_FILE = data_path(
    CONFIG[
        "datasets"
    ][
        "byra"
    ][
        "mat_file"
    ]
)


# ============================================================
# OUTPUT PATHS
# ============================================================

BEHSOF_INVENTORY_CSV = project_path(
    CONFIG[
        "outputs"
    ][
        "inventory_csv"
    ]
)


BEHSOF_QUALITY_AUDIT_CSV = project_path(
    CONFIG[
        "outputs"
    ][
        "quality_audit_csv"
    ]
)


TAL_CONTACT_SHEET_INDEX = project_path(
    CONFIG[
        "outputs"
    ][
        "tal_contact_sheet_index"
    ]
)


BEH_SAMPLE_REVIEW_CSV = project_path(
    CONFIG[
        "outputs"
    ][
        "beh_sample_review_csv"
    ]
)


BEH_OUTLIERS_REVIEW_CSV = project_path(
    CONFIG[
        "outputs"
    ][
        "beh_outliers_review_csv"
    ]
)


BEHSOF_MANIFEST_CSV = project_path(
    CONFIG[
        "outputs"
    ][
        "manifest_csv"
    ]
)


BEH_PATIENT_FOLDS_CSV = project_path(
    CONFIG[
        "outputs"
    ][
        "patient_folds_csv"
    ]
)


TAL_CONTACT_SHEETS_DIR = project_path(
    CONFIG[
        "outputs"
    ][
        "tal_contact_sheets_dir"
    ]
)


BEH_SAMPLE_CONTACT_SHEETS_DIR = project_path(
    CONFIG[
        "outputs"
    ][
        "beh_sample_contact_sheets_dir"
    ]
)


BEH_OUTLIER_REVIEW_DIR = project_path(
    CONFIG[
        "outputs"
    ][
        "beh_outlier_review_dir"
    ]
)