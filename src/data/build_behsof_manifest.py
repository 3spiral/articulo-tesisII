import math

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.settings import (
    BEHSOF_IMAGES_DIR,
    BEHSOF_DEMO_CSV,
    TAL_CONTACT_SHEET_INDEX,
    BEHSOF_MANIFEST_CSV,
    BEH_PATIENT_FOLDS_CSV,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = BEHSOF_IMAGES_DIR

DEMO_CSV = BEHSOF_DEMO_CSV

TAL_QC_CSV = TAL_CONTACT_SHEET_INDEX

OUTPUT = BEHSOF_MANIFEST_CSV

EXPORT_PATIENT_FOLDS = True

PATIENT_FOLDS_OUTPUT = (
    BEH_PATIENT_FOLDS_CSV
)

# Configuración reproducible de validación cruzada.
N_SPLITS = 5
CV_RANDOM_STATE = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# UTILIDADES
# ============================================================

def normalize_patient_id(value):
    """
    Normaliza valores como:

        BEH12345
        \\Data\\BEH12345
        /Data/BEH12345

    a:

        BEH12345
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    value = value.replace(
        "\\",
        "/"
    )

    value = value.rstrip("/")

    if value == "":
        return None

    return value.split("/")[-1].upper()


def normalize_path(value):
    """
    Normaliza separadores de ruta.

    Ejemplo:

        TAL06221\\Image4.jpg

    pasa a:

        TAL06221/Image4.jpg
    """

    if pd.isna(value):
        return None

    value = (
        str(value)
        .strip()
        .replace("\\", "/")
    )

    if value == "":
        return None

    return value


def normalize_grade(value):
    """
    Normaliza grados de esteatosis:

        0
        0.0
        S0
        s0

    a:

        0, 1, 2, 3

    Rechaza:
        - decimales no enteros, como 1.5
        - valores fuera de 0-3
        - valores no numéricos
        - NaN / infinito
    """

    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    if value.startswith("S"):
        value = value[1:]

    try:
        numeric_value = float(value)

    except (ValueError, TypeError):
        return None

    if not math.isfinite(numeric_value):
        return None

    if not numeric_value.is_integer():
        return None

    grade = int(
        numeric_value
    )

    if grade not in {
        0,
        1,
        2,
        3,
    }:
        return None

    return grade


def make_binary_target(grade):
    """
    Target principal:

        0 = S0/S1
        1 = S2/S3
    """

    if pd.isna(grade):
        return pd.NA

    grade = int(
        grade
    )

    if grade in {
        0,
        1,
    }:
        return 0

    if grade in {
        2,
        3,
    }:
        return 1

    return pd.NA


# ============================================================
# 1. VALIDAR ARCHIVOS DE ENTRADA
# ============================================================

if not ROOT.exists():
    raise FileNotFoundError(
        f"No existe ROOT: {ROOT}"
    )

if not ROOT.is_dir():
    raise NotADirectoryError(
        f"ROOT no es una carpeta: {ROOT}"
    )

if not DEMO_CSV.exists():
    raise FileNotFoundError(
        f"No existe Demo_data.csv: {DEMO_CSV}"
    )

if not TAL_QC_CSV.exists():
    raise FileNotFoundError(
        f"No existe TAL QC: {TAL_QC_CSV}"
    )


# ============================================================
# 2. CARGAR DEMO_DATA
# ============================================================

demo = pd.read_csv(
    DEMO_CSV
)

required_demo_columns = [
    "Patient ID",
    "Fibroscan S",
    "Steatosis stage",
    "Fibroscan F",
]

for column in required_demo_columns:

    if column not in demo.columns:

        raise ValueError(
            f"Falta columna requerida "
            f"en Demo_data.csv: {column}"
        )


# ============================================================
# 3. NORMALIZAR METADATOS
# ============================================================

demo["patient_id"] = (
    demo["Patient ID"]
    .apply(
        normalize_patient_id
    )
)

demo["fibroscan_s"] = (
    demo["Fibroscan S"]
    .apply(
        normalize_grade
    )
)

demo["steatosis_stage"] = (
    demo["Steatosis stage"]
    .apply(
        normalize_grade
    )
)

# Fibrosis se conserva numéricamente.
demo["fibroscan_f"] = pd.to_numeric(
    demo["Fibroscan F"],
    errors="coerce"
)

if "CAP score" in demo.columns:

    demo["cap_score"] = pd.to_numeric(
        demo["CAP score"],
        errors="coerce"
    )

else:

    demo["cap_score"] = pd.NA

if "E score" in demo.columns:

    demo["e_score"] = pd.to_numeric(
        demo["E score"],
        errors="coerce"
    )

else:

    demo["e_score"] = pd.NA


# ============================================================
# 4. VALIDAR PACIENTES EN DEMO_DATA
# ============================================================

if demo["patient_id"].isna().any():

    raise ValueError(
        "Hay Patient ID vacíos o inválidos "
        "en Demo_data.csv."
    )

if (
    demo["patient_id"]
    .astype(str)
    .str.strip()
    .eq("")
    .any()
):

    raise ValueError(
        "Hay Patient ID vacíos "
        "en Demo_data.csv."
    )

duplicate_demo_ids = (
    demo["patient_id"]
    .duplicated(
        keep=False
    )
)

if duplicate_demo_ids.any():

    print(
        "\nPacientes duplicados "
        "en Demo_data.csv:"
    )

    print(
        demo.loc[
            duplicate_demo_ids,
            [
                "patient_id",
            ]
        ]
    )

    raise ValueError(
        "Demo_data.csv contiene "
        "Patient ID duplicados."
    )

if demo[
    "patient_id"
].nunique() != 113:

    raise ValueError(
        "Se esperaban 113 pacientes "
        "en Demo_data.csv, pero hay "
        f"{demo['patient_id'].nunique()}."
    )


# ============================================================
# 5. VALIDAR VARIABLES CLÍNICAS
# ============================================================

invalid_fibroscan_s = (
    demo[
        "fibroscan_s"
    ].isna()
)

if invalid_fibroscan_s.any():

    print(
        "\nPacientes con Fibroscan S inválido:"
    )

    print(
        demo.loc[
            invalid_fibroscan_s,
            [
                "patient_id",
                "Fibroscan S",
            ]
        ]
    )

    raise ValueError(
        "Hay Fibroscan S faltante o inválido."
    )

invalid_steatosis_stage = (
    demo[
        "steatosis_stage"
    ].isna()
)

if invalid_steatosis_stage.any():

    print(
        "\nPacientes con Steatosis stage inválido:"
    )

    print(
        demo.loc[
            invalid_steatosis_stage,
            [
                "patient_id",
                "Steatosis stage",
            ]
        ]
    )

    raise ValueError(
        "Hay Steatosis stage faltante o inválido."
    )

if demo[
    "fibroscan_f"
].isna().any():

    print(
        "\nPacientes con Fibroscan F faltante "
        "o no numérico:"
    )

    print(
        demo.loc[
            demo[
                "fibroscan_f"
            ].isna(),
            [
                "patient_id",
                "Fibroscan F",
            ]
        ]
    )

    raise ValueError(
        "Hay Fibroscan F faltante o no numérico."
    )

# CAP score y E score NO se exigen completos,
# porque contienen valores faltantes legítimos.


# ============================================================
# 6. CREAR TARGET BINARIO
# ============================================================

demo[
    "target_binary"
] = (
    demo[
        "fibroscan_s"
    ]
    .apply(
        make_binary_target
    )
)

if demo[
    "target_binary"
].isna().any():

    raise ValueError(
        "No se pudo construir "
        "target_binary para todos los pacientes."
    )


# ============================================================
# 7. INVENTARIO REAL DE IMAGE_DATA
# ============================================================

image_rows = []

for patient_folder in sorted(
    ROOT.iterdir()
):

    if not patient_folder.is_dir():
        continue

    patient_id = (
        patient_folder.name.upper()
    )

    if not (
        patient_id.startswith("BEH")
        or
        patient_id.startswith("TAL")
    ):
        continue

    center = (
        patient_id[:3]
    )

    for path in sorted(
        patient_folder.rglob("*")
    ):

        if not path.is_file():
            continue

        if (
            path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        relative_path = normalize_path(
            path.relative_to(
                ROOT
            )
        )

        image_rows.append({
            "patient_id":
                patient_id,

            "center":
                center,

            "relative_path":
                relative_path,

            "filename":
                path.name,

            "extension":
                path.suffix.lower(),
        })

images = pd.DataFrame(
    image_rows
)

if len(
    images
) == 0:

    raise ValueError(
        "No se encontraron imágenes."
    )


# ============================================================
# 8. VALIDAR INVENTARIO
# ============================================================

print(
    "\n========================================"
)

print(
    "INVENTARIO DE IMÁGENES"
)

print(
    "========================================"
)

print(
    "\nImágenes:",
    len(images)
)

print(
    "Pacientes:",
    images[
        "patient_id"
    ].nunique()
)

print(
    "\nPor centro:"
)

print(
    images[
        "center"
    ].value_counts()
)

duplicate_paths = (
    images[
        "relative_path"
    ]
    .duplicated(
        keep=False
    )
)

if duplicate_paths.any():

    print(
        "\nRutas duplicadas:"
    )

    print(
        images.loc[
            duplicate_paths,
            [
                "patient_id",
                "relative_path",
            ]
        ]
    )

    raise ValueError(
        "Existen relative_path duplicados."
    )


# ============================================================
# 9. VALIDAR PACIENTES INVENTARIO ↔ DEMO
# ============================================================

inventory_patients = set(
    images[
        "patient_id"
    ].unique()
)

demo_patients = set(
    demo[
        "patient_id"
    ].unique()
)

missing_in_demo = (
    inventory_patients
    -
    demo_patients
)

missing_in_inventory = (
    demo_patients
    -
    inventory_patients
)

if missing_in_demo:

    print(
        "\nPacientes presentes en image_Data "
        "pero ausentes en Demo_data:"
    )

    for patient in sorted(
        missing_in_demo
    ):
        print(
            patient
        )

    raise ValueError(
        "Hay pacientes de image_Data "
        "sin metadata."
    )

if missing_in_inventory:

    print(
        "\nPacientes en Demo_data "
        "sin imágenes:"
    )

    for patient in sorted(
        missing_in_inventory
    ):
        print(
            patient
        )

    raise ValueError(
        "Hay pacientes en Demo_data "
        "sin imágenes."
    )


# ============================================================
# 10. UNIR IMÁGENES CON METADATOS
# ============================================================

metadata_columns = [
    "patient_id",
    "fibroscan_s",
    "target_binary",
    "steatosis_stage",
    "fibroscan_f",
    "cap_score",
    "e_score",
]

manifest = images.merge(
    demo[
        metadata_columns
    ],
    on="patient_id",
    how="left",
    validate="many_to_one",
)


# ============================================================
# 11. VALIDAR METADATA POST-MERGE
# ============================================================

if manifest[
    "fibroscan_s"
].isna().any():

    print(
        "\nImágenes sin Fibroscan S:"
    )

    print(
        manifest.loc[
            manifest[
                "fibroscan_s"
            ].isna(),
            [
                "patient_id",
                "relative_path",
            ]
        ]
    )

    raise ValueError(
        "Hay imágenes sin Fibroscan S válido."
    )

if manifest[
    "target_binary"
].isna().any():

    raise ValueError(
        "Hay imágenes sin target_binary."
    )


# ============================================================
# 12. CARGAR QC TAL
# ============================================================

tal_qc = pd.read_csv(
    TAL_QC_CSV
)

required_qc_columns = [
    "patient_id",
    "relative_path",
    "qc_label",
]

for column in required_qc_columns:

    if column not in tal_qc.columns:

        raise ValueError(
            f"Falta columna en TAL QC: {column}"
        )

tal_qc[
    "patient_id"
] = (
    tal_qc[
        "patient_id"
    ]
    .apply(
        normalize_patient_id
    )
)

tal_qc[
    "relative_path"
] = (
    tal_qc[
        "relative_path"
    ]
    .apply(
        normalize_path
    )
)

# StringDtype conserva NA reales.
tal_qc[
    "qc_label"
] = (
    tal_qc[
        "qc_label"
    ]
    .astype("string")
    .str.strip()
    .str.upper()
)


# ============================================================
# 13. VALIDAR CAMPOS OBLIGATORIOS DEL QC TAL
# ============================================================

if tal_qc[
    "patient_id"
].isna().any():

    raise ValueError(
        "Hay patient_id vacíos en TAL QC."
    )

if tal_qc[
    "relative_path"
].isna().any():

    raise ValueError(
        "Hay relative_path vacíos en TAL QC."
    )

if tal_qc[
    "qc_label"
].isna().any():

    raise ValueError(
        "Hay qc_label vacíos en TAL QC."
    )

if (
    tal_qc[
        "patient_id"
    ]
    .astype(str)
    .str.strip()
    .eq("")
    .any()
):

    raise ValueError(
        "Hay patient_id vacíos en TAL QC."
    )

if (
    tal_qc[
        "relative_path"
    ]
    .astype(str)
    .str.strip()
    .eq("")
    .any()
):

    raise ValueError(
        "Hay relative_path vacíos en TAL QC."
    )

if (
    tal_qc[
        "qc_label"
    ]
    .astype(str)
    .str.strip()
    .eq("")
    .any()
):

    raise ValueError(
        "Hay qc_label vacíos en TAL QC."
    )


# ============================================================
# 14. VALIDACIONES ESTRUCTURALES QC TAL
# ============================================================

if len(
    tal_qc
) != 213:

    raise ValueError(
        "Se esperaban 213 filas "
        f"en TAL QC, pero hay {len(tal_qc)}."
    )

tal_patient_count = (
    tal_qc[
        "patient_id"
    ].nunique()
)

if tal_patient_count != 18:

    raise ValueError(
        "Se esperaban 18 pacientes TAL "
        f"en QC, pero hay {tal_patient_count}."
    )

invalid_center = (
    ~tal_qc[
        "patient_id"
    ].str.startswith(
        "TAL"
    )
)

if invalid_center.any():

    print(
        "\nFilas QC que no pertenecen a TAL:"
    )

    print(
        tal_qc.loc[
            invalid_center,
            [
                "patient_id",
                "relative_path",
            ]
        ]
    )

    raise ValueError(
        "El QC TAL contiene "
        "pacientes que no son TAL."
    )


# ============================================================
# 15. COHERENCIA PATIENT_ID ↔ RUTA
# ============================================================

path_patient = (
    tal_qc[
        "relative_path"
    ]
    .str.split("/")
    .str[0]
    .str.upper()
)

patient_path_mismatch = (
    path_patient
    !=
    tal_qc[
        "patient_id"
    ]
)

if patient_path_mismatch.any():

    print(
        "\nPatient ID no coincide "
        "con la carpeta de relative_path:"
    )

    print(
        tal_qc.loc[
            patient_path_mismatch,
            [
                "patient_id",
                "relative_path",
            ]
        ]
    )

    raise ValueError(
        "Hay inconsistencias "
        "patient_id ↔ relative_path."
    )


# ============================================================
# 16. VALIDAR ETIQUETAS QC
# ============================================================

allowed_labels = {
    "V",
    "D",
    "L",
    "X",
}

invalid_labels = (
    ~tal_qc[
        "qc_label"
    ].isin(
        allowed_labels
    )
)

if invalid_labels.any():

    print(
        "\nEtiquetas QC inválidas:"
    )

    print(
        tal_qc.loc[
            invalid_labels,
            [
                "patient_id",
                "relative_path",
                "qc_label",
            ]
        ]
    )

    raise ValueError(
        "Existen etiquetas TAL QC inválidas."
    )


# ============================================================
# 17. VALIDAR RUTAS DUPLICADAS QC
# ============================================================

duplicate_tal_qc_paths = (
    tal_qc[
        "relative_path"
    ]
    .duplicated(
        keep=False
    )
)

if duplicate_tal_qc_paths.any():

    print(
        "\nRutas duplicadas en TAL QC:"
    )

    print(
        tal_qc.loc[
            duplicate_tal_qc_paths,
            [
                "patient_id",
                "relative_path",
            ]
        ]
    )

    raise ValueError(
        "Hay relative_path duplicados "
        "en TAL QC."
    )


# ============================================================
# 18. COMPARAR INVENTARIO TAL ↔ QC TAL
# ============================================================

inventory_tal_paths = set(
    manifest.loc[
        manifest[
            "center"
        ] == "TAL",
        "relative_path",
    ]
)

qc_tal_paths = set(
    tal_qc[
        "relative_path"
    ]
)

missing_in_qc = (
    inventory_tal_paths
    -
    qc_tal_paths
)

extra_in_qc = (
    qc_tal_paths
    -
    inventory_tal_paths
)

if missing_in_qc:

    print(
        "\nImágenes TAL reales "
        "sin fila QC:"
    )

    for path in sorted(
        missing_in_qc
    ):
        print(
            path
        )

    raise ValueError(
        "El QC TAL no cubre "
        "todas las imágenes reales."
    )

if extra_in_qc:

    print(
        "\nRutas QC que no existen "
        "en image_Data:"
    )

    for path in sorted(
        extra_in_qc
    ):
        print(
            path
        )

    raise ValueError(
        "El QC TAL contiene rutas "
        "que no existen en el inventario."
    )


# ============================================================
# 19. INCORPORAR QC TAL
# ============================================================

tal_qc_small = (
    tal_qc[
        [
            "relative_path",
            "qc_label",
        ]
    ]
    .copy()
)

manifest = manifest.merge(
    tal_qc_small,
    on="relative_path",
    how="left",
    validate="one_to_one",
)


# ============================================================
# 20. MÁSCARAS DE CENTRO
# ============================================================

beh_mask = (
    manifest[
        "center"
    ] == "BEH"
)

tal_mask = (
    manifest[
        "center"
    ] == "TAL"
)


# ============================================================
# 21. ASIGNAR QC A BEH
# ============================================================

# BEH fue evaluado mediante:
#
# - auditoría automática del 100 %
# - muestra visual estratificada
# - revisión dirigida de outliers
#
# V significa inclusión técnica final,
# NO validación clínica independiente.

manifest.loc[
    beh_mask,
    "qc_label"
] = "V"


# ============================================================
# 22. VALIDAR QUE TODA TAL TENGA QC
# ============================================================

missing_tal_qc = (
    tal_mask
    &
    manifest[
        "qc_label"
    ].isna()
)

if missing_tal_qc.any():

    print(
        "\nImágenes TAL sin QC:"
    )

    print(
        manifest.loc[
            missing_tal_qc,
            [
                "patient_id",
                "relative_path",
            ]
        ]
    )

    raise ValueError(
        "Hay imágenes TAL "
        "sin revisión manual."
    )


# ============================================================
# 23. QC STATUS
# ============================================================

qc_status_map = {

    "V":
        "INCLUDE",

    "D":
        "EXCLUDE_DUAL_PANEL",

    "L":
        "EXCLUDE_LOW_INFORMATION",

    "X":
        "EXCLUDE_NON_ULTRASOUND",
}

manifest[
    "qc_status"
] = (
    manifest[
        "qc_label"
    ]
    .map(
        qc_status_map
    )
)

if manifest[
    "qc_status"
].isna().any():

    raise ValueError(
        "Hay filas sin qc_status válido."
    )


# ============================================================
# 24. MÉTODO DE QC
# ============================================================

manifest[
    "qc_method"
] = ""

manifest.loc[
    beh_mask,
    "qc_method"
] = (
    "automatic_full_plus_visual_sampling"
)

manifest.loc[
    tal_mask,
    "qc_method"
] = (
    "manual_full_review"
)


# ============================================================
# 25. INCLUDE FINAL
# ============================================================

manifest[
    "include_final"
] = (
    manifest[
        "qc_label"
    ] == "V"
)


# ============================================================
# 26. MOTIVO DE EXCLUSIÓN
# ============================================================

reason_map = {

    "V":
        "",

    "D":
        "dual_panel",

    "L":
        "low_information",

    "X":
        "non_ultrasound",
}

manifest[
    "exclusion_reason"
] = (
    manifest[
        "qc_label"
    ]
    .map(
        reason_map
    )
)


# ============================================================
# 27. DATASET ROLE
# ============================================================

manifest[
    "dataset_role"
] = (
    manifest[
        "center"
    ]
    .map({
        "BEH":
            "development",

        "TAL":
            "external_test",
    })
)

if manifest[
    "dataset_role"
].isna().any():

    raise ValueError(
        "Hay filas sin dataset_role."
    )


# ============================================================
# 28. IMÁGENES INCLUIDAS POR PACIENTE
# ============================================================

patient_included_counts = (
    manifest.loc[
        manifest[
            "include_final"
        ]
    ]
    .groupby(
        "patient_id"
    )
    .size()
)

manifest[
    "included_images_patient"
] = (
    manifest[
        "patient_id"
    ]
    .map(
        patient_included_counts
    )
    .fillna(0)
    .astype(int)
)


# ============================================================
# 29. ELEGIBILIDAD A NIVEL PACIENTE
# ============================================================

manifest[
    "patient_eligible"
] = (
    manifest[
        "included_images_patient"
    ] > 0
)


# ============================================================
# 30. CREAR 5 FOLDS ESTRATIFICADOS A NIVEL DE PACIENTE
# ============================================================

# IMPORTANTE:
# - Solo BEH participa en cross-validation.
# - TAL NO recibe cv_fold.
# - La estratificación se realiza sobre una tabla con UNA fila
#   por paciente, no sobre imágenes.
# - Todas las imágenes de un paciente heredan el mismo fold.

beh_patients = (
    manifest.loc[
        (
            manifest[
                "center"
            ] == "BEH"
        )
        &
        (
            manifest[
                "patient_eligible"
            ]
        ),
        [
            "patient_id",
            "target_binary",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "patient_id"
    )
    .reset_index(
        drop=True
    )
)

if len(
    beh_patients
) != 95:

    raise ValueError(
        "Se esperaban 95 pacientes BEH "
        f"elegibles para CV, pero hay {len(beh_patients)}."
    )

if beh_patients[
    "patient_id"
].nunique() != 95:

    raise ValueError(
        "La tabla BEH de pacientes "
        "no contiene exactamente 95 IDs únicos."
    )

if (
    beh_patients[
        "target_binary"
    ].value_counts()
    .sort_index()
    .to_dict()
    !=
    {
        0: 29,
        1: 66,
    }
):

    raise ValueError(
        "La distribución BEH previa a CV "
        "no coincide con 29 clase 0 / 66 clase 1."
    )

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=CV_RANDOM_STATE,
)

beh_patients[
    "cv_fold"
] = pd.NA

for fold_number, (
    _train_idx,
    val_idx,
) in enumerate(
    skf.split(
        beh_patients[
            "patient_id"
        ],
        beh_patients[
            "target_binary"
        ],
    ),
    start=1,
):

    beh_patients.loc[
        val_idx,
        "cv_fold"
    ] = fold_number

beh_patients[
    "cv_fold"
] = (
    beh_patients[
        "cv_fold"
    ]
    .astype("Int64")
)

if beh_patients[
    "cv_fold"
].isna().any():

    raise ValueError(
        "Hay pacientes BEH sin cv_fold."
    )

if not set(
    beh_patients[
        "cv_fold"
    ].astype(int).unique()
) == {
    1,
    2,
    3,
    4,
    5,
}:

    raise ValueError(
        "Los folds generados no son exactamente 1-5."
    )

fold_by_patient = (
    beh_patients
    .set_index(
        "patient_id"
    )[
        "cv_fold"
    ]
    .to_dict()
)

# Al mapear solo BEH, TAL queda NA de forma intencional.
manifest[
    "cv_fold"
] = (
    manifest[
        "patient_id"
    ]
    .map(
        fold_by_patient
    )
    .astype("Int64")
)


# ============================================================
# 31. VALIDAR FOLDS Y AUSENCIA DE PATIENT LEAKAGE
# ============================================================

# TAL no debe recibir fold.
if manifest.loc[
    tal_mask,
    "cv_fold"
].notna().any():

    raise ValueError(
        "Se asignó cv_fold a TAL. "
        "TAL debe permanecer fuera de CV."
    )

# Todas las imágenes BEH deben recibir fold.
if manifest.loc[
    beh_mask,
    "cv_fold"
].isna().any():

    raise ValueError(
        "Hay imágenes BEH sin cv_fold."
    )

# Cada paciente BEH debe tener exactamente un fold.
folds_per_beh_patient = (
    manifest.loc[
        beh_mask
    ]
    .groupby(
        "patient_id"
    )[
        "cv_fold"
    ]
    .nunique()
)

if (
    folds_per_beh_patient != 1
).any():

    print(
        "\nPacientes BEH con múltiples folds:"
    )

    print(
        folds_per_beh_patient[
            folds_per_beh_patient != 1
        ]
    )

    raise ValueError(
        "Existe riesgo de patient leakage: "
        "un paciente BEH aparece en más de un fold."
    )

# Comprobar tamaño y distribución por fold a nivel paciente.
fold_patient_counts = (
    beh_patients
    .groupby(
        [
            "cv_fold",
            "target_binary",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

# Con 95 pacientes y 5 folds, cada fold debe contener 19 pacientes.
# Para una estratificación correcta:
#   - clase 0 (29 pacientes) se reparte como 5/6 entre folds
#   - clase 1 (66 pacientes) se reparte como 13/14 entre folds
#
# No se fuerza qué número de fold recibe exactamente el 5/14;
# eso evita una comprobación innecesariamente dependiente de detalles
# internos de la versión de scikit-learn.

class0_per_fold = []
class1_per_fold = []

for fold_number in range(
    1,
    N_SPLITS + 1
):

    fold_rows = (
        beh_patients[
            beh_patients[
                "cv_fold"
            ] == fold_number
        ]
    )

    class_counts = (
        fold_rows[
            "target_binary"
        ]
        .value_counts()
        .to_dict()
    )

    if len(
        fold_rows
    ) != 19:

        raise ValueError(
            f"Fold {fold_number} debería tener "
            f"19 pacientes, pero tiene {len(fold_rows)}."
        )

    c0 = class_counts.get(
        0,
        0
    )

    c1 = class_counts.get(
        1,
        0
    )

    class0_per_fold.append(
        c0
    )

    class1_per_fold.append(
        c1
    )

    if c0 not in {
        5,
        6,
    }:

        raise ValueError(
            f"Fold {fold_number}: número inesperado "
            f"de pacientes clase 0 ({c0})."
        )

    if c1 not in {
        13,
        14,
    }:

        raise ValueError(
            f"Fold {fold_number}: número inesperado "
            f"de pacientes clase 1 ({c1})."
        )

if sum(
    class0_per_fold
) != 29:

    raise ValueError(
        "Los folds no conservan los 29 pacientes clase 0."
    )

if sum(
    class1_per_fold
) != 66:

    raise ValueError(
        "Los folds no conservan los 66 pacientes clase 1."
    )

if max(
    class0_per_fold
) - min(
    class0_per_fold
) > 1:

    raise ValueError(
        "La estratificación de clase 0 difiere "
        "en más de un paciente entre folds."
    )

if max(
    class1_per_fold
) - min(
    class1_per_fold
) > 1:

    raise ValueError(
        "La estratificación de clase 1 difiere "
        "en más de un paciente entre folds."
    )


# ============================================================
# 32. SPLIT FIJO
# ============================================================

# No se asigna un split fijo train/validation/test.
#
# BEH utiliza cross-validation mediante cv_fold.
# TAL queda identificado por dataset_role="external_test".
#
# Por eso split permanece vacío deliberadamente.

manifest[
    "split"
] = ""


# ============================================================
# 33. ORDEN DE COLUMNAS
# ============================================================

column_order = [

    "patient_id",
    "center",

    "relative_path",
    "filename",
    "extension",

    "fibroscan_s",
    "target_binary",

    "steatosis_stage",
    "fibroscan_f",

    "cap_score",
    "e_score",

    "qc_label",
    "qc_status",
    "qc_method",

    "include_final",
    "exclusion_reason",

    "included_images_patient",
    "patient_eligible",

    "dataset_role",

    "cv_fold",
    "split",
]

manifest = manifest[
    column_order
]


# ============================================================
# 34. ORDENAR
# ============================================================

manifest = (
    manifest
    .sort_values(
        [
            "center",
            "patient_id",
            "relative_path",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 35. VALIDACIONES LÓGICAS
# ============================================================

labels_per_patient = (
    manifest
    .groupby(
        "patient_id"
    )[
        "target_binary"
    ]
    .nunique()
)

if labels_per_patient.max() != 1:

    print(
        "\nPacientes con múltiples targets:"
    )

    print(
        labels_per_patient[
            labels_per_patient > 1
        ]
    )

    raise ValueError(
        "Un paciente tiene más de "
        "una clase binaria."
    )

grades_per_patient = (
    manifest
    .groupby(
        "patient_id"
    )[
        "fibroscan_s"
    ]
    .nunique()
)

if grades_per_patient.max() != 1:

    print(
        "\nPacientes con múltiples "
        "Fibroscan S:"
    )

    print(
        grades_per_patient[
            grades_per_patient > 1
        ]
    )

    raise ValueError(
        "Un paciente tiene más de "
        "un Fibroscan S."
    )


# ============================================================
# 36. VALIDAR TARGET BINARIO
# ============================================================

expected_target = (
    manifest[
        "fibroscan_s"
    ]
    .apply(
        lambda x:
            0
            if int(x) in {
                0,
                1,
            }
            else 1
    )
)

target_consistent = (
    manifest[
        "target_binary"
    ]
    .astype(int)
    .equals(
        expected_target.astype(int)
    )
)

if not target_consistent:

    raise ValueError(
        "target_binary no coincide "
        "con Fibroscan S."
    )


# ============================================================
# 37. RESUMEN
# ============================================================

print(
    "\n========================================"
)

print(
    "BEHSOF MANIFEST"
)

print(
    "========================================"
)

print(
    "\nFilas totales:",
    len(manifest)
)

print(
    "\nImágenes por centro:"
)

print(
    manifest[
        "center"
    ]
    .value_counts()
)

print(
    "\nQC:"
)

print(
    manifest[
        "qc_status"
    ]
    .value_counts()
)

print(
    "\nIncluidas:"
)

print(
    manifest[
        "include_final"
    ]
    .value_counts()
)

print(
    "\nImágenes incluidas por centro:"
)

print(
    manifest.loc[
        manifest[
            "include_final"
        ],
        "center",
    ]
    .value_counts()
)

print(
    "\nPacientes originales:",
    manifest[
        "patient_id"
    ].nunique()
)

eligible_patients = (
    manifest.loc[
        manifest[
            "patient_eligible"
        ],
        "patient_id",
    ]
    .nunique()
)

print(
    "Pacientes elegibles:",
    eligible_patients
)


# ============================================================
# 38. PACIENTES EXCLUIDOS COMPLETAMENTE
# ============================================================

excluded_patients = (
    manifest.loc[
        ~manifest[
            "patient_eligible"
        ],
        [
            "patient_id",
            "center",
            "fibroscan_s",
            "target_binary",
        ]
    ]
    .drop_duplicates()
)

print(
    "\nPacientes excluidos completamente:"
)

if len(
    excluded_patients
) == 0:

    print(
        "Ninguno"
    )

else:

    print(
        excluded_patients.to_string(
            index=False
        )
    )


# ============================================================
# 39. TABLA A NIVEL PACIENTE
# ============================================================

patient_table = (
    manifest.loc[
        manifest[
            "patient_eligible"
        ],
        [
            "patient_id",
            "center",
            "fibroscan_s",
            "target_binary",
        ]
    ]
    .drop_duplicates()
)

print(
    "\nDistribución de pacientes "
    "elegibles por clase:"
)

print(
    patient_table[
        "target_binary"
    ]
    .value_counts()
    .sort_index()
)

print(
    "\nPor centro y clase:"
)

print(
    pd.crosstab(
        patient_table[
            "center"
        ],
        patient_table[
            "target_binary"
        ]
    )
)

beh_patient_table = (
    patient_table[
        patient_table[
            "center"
        ] == "BEH"
    ]
)

tal_patient_table = (
    patient_table[
        patient_table[
            "center"
        ] == "TAL"
    ]
)


# ============================================================
# 40. RESUMEN DE FOLDS
# ============================================================

print(
    "\nDistribución de folds BEH "
    "a nivel paciente:"
)

fold_summary = (
    beh_patients
    .groupby(
        [
            "cv_fold",
            "target_binary",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

fold_summary[
    "total"
] = (
    fold_summary.sum(
        axis=1
    )
)

print(
    fold_summary
)


# ============================================================
# 41. ASSERTIONS FINALES
# ============================================================

print(
    "\n========================================"
)

print(
    "COMPROBACIONES FINALES"
)

print(
    "========================================"
)

checks = {

    "1601 imágenes originales":
        len(
            manifest
        ) == 1601,

    "113 pacientes originales":
        manifest[
            "patient_id"
        ].nunique() == 113,

    "1388 imágenes BEH":
        (
            manifest[
                "center"
            ] == "BEH"
        ).sum() == 1388,

    "213 imágenes TAL":
        (
            manifest[
                "center"
            ] == "TAL"
        ).sum() == 213,

    "95 pacientes BEH":
        manifest.loc[
            manifest[
                "center"
            ] == "BEH",
            "patient_id",
        ].nunique() == 95,

    "18 pacientes TAL originales":
        manifest.loc[
            manifest[
                "center"
            ] == "TAL",
            "patient_id",
        ].nunique() == 18,

    "1566 imágenes incluidas":
        manifest[
            "include_final"
        ].sum() == 1566,

    "35 imágenes excluidas":
        (
            ~manifest[
                "include_final"
            ]
        ).sum() == 35,

    "1388 BEH incluidas":
        manifest.loc[
            (
                manifest[
                    "center"
                ] == "BEH"
            )
            &
            (
                manifest[
                    "include_final"
                ]
            )
        ].shape[0] == 1388,

    "178 TAL incluidas":
        manifest.loc[
            (
                manifest[
                    "center"
                ] == "TAL"
            )
            &
            (
                manifest[
                    "include_final"
                ]
            )
        ].shape[0] == 178,

    "26 dual-panel excluidas":
        (
            manifest[
                "qc_label"
            ] == "D"
        ).sum() == 26,

    "6 low-information excluidas":
        (
            manifest[
                "qc_label"
            ] == "L"
        ).sum() == 6,

    "3 no-US excluidas":
        (
            manifest[
                "qc_label"
            ] == "X"
        ).sum() == 3,

    "112 pacientes elegibles":
        eligible_patients == 112,

    "17 pacientes TAL elegibles":
        manifest.loc[
            (
                manifest[
                    "center"
                ] == "TAL"
            )
            &
            (
                manifest[
                    "patient_eligible"
                ]
            ),
            "patient_id",
        ].nunique() == 17,

    "TAL07001 no elegible":
        not manifest.loc[
            manifest[
                "patient_id"
            ] == "TAL07001",
            "patient_eligible",
        ].any(),

    "TAL07001 tiene 0 imágenes incluidas":
        manifest.loc[
            manifest[
                "patient_id"
            ] == "TAL07001",
            "include_final",
        ].sum() == 0,

    "Todas las filas tienen target":
        manifest[
            "target_binary"
        ]
        .notna()
        .all(),

    "Target binario coherente":
        target_consistent,

    "Una única clase por paciente":
        labels_per_patient.max() == 1,

    "Un único Fibroscan S por paciente":
        grades_per_patient.max() == 1,

    "Todas las filas tienen dataset_role":
        manifest[
            "dataset_role"
        ]
        .notna()
        .all(),

    "Todas las filas tienen qc_method":
        (
            manifest[
                "qc_method"
            ]
            .astype(str)
            .str.len()
            >
            0
        ).all(),

    "39 pacientes elegibles clase 0":
        (
            patient_table[
                "target_binary"
            ] == 0
        ).sum() == 39,

    "73 pacientes elegibles clase 1":
        (
            patient_table[
                "target_binary"
            ] == 1
        ).sum() == 73,

    "BEH tiene 29 pacientes clase 0":
        (
            beh_patient_table[
                "target_binary"
            ] == 0
        ).sum() == 29,

    "BEH tiene 66 pacientes clase 1":
        (
            beh_patient_table[
                "target_binary"
            ] == 1
        ).sum() == 66,

    "TAL tiene 10 pacientes clase 0":
        (
            tal_patient_table[
                "target_binary"
            ] == 0
        ).sum() == 10,

    "TAL tiene 7 pacientes clase 1":
        (
            tal_patient_table[
                "target_binary"
            ] == 1
        ).sum() == 7,

    "95 pacientes BEH con cv_fold":
        beh_patients[
            "cv_fold"
        ]
        .notna()
        .sum() == 95,

    "TAL sin cv_fold":
        manifest.loc[
            tal_mask,
            "cv_fold"
        ]
        .isna()
        .all(),

    "Cada paciente BEH pertenece a un solo fold":
        (
            folds_per_beh_patient == 1
        ).all(),

    "Cada fold tiene 19 pacientes":
        (
            beh_patients[
                "cv_fold"
            ]
            .value_counts()
            .sort_index()
            == 19
        ).all(),

    "Los folds cubren exactamente 95 pacientes":
        beh_patients[
            "cv_fold"
        ]
        .value_counts()
        .sum() == 95,

    "split permanece vacío":
        (
            manifest[
                "split"
            ]
            .astype(str)
            .str.len()
            == 0
        ).all(),
}

for description, passed in checks.items():

    symbol = (
        "OK"
        if passed
        else "ERROR"
    )

    print(
        f"[{symbol}] {description}"
    )


# ============================================================
# 42. DETENER SI ALGÚN CHECK FALLA
# ============================================================

if not all(
    checks.values()
):

    failed = [
        description
        for description, passed
        in checks.items()
        if not passed
    ]

    print(
        "\nComprobaciones fallidas:"
    )

    for item in failed:
        print(
            "-",
            item
        )

    raise ValueError(
        "\nEl manifest no superó todas "
        "las comprobaciones. "
        "No se generará BEHSOF_manifest.csv."
    )


# ============================================================
# 43. GUARDAR MANIFEST SOLO DESPUÉS DE VALIDAR
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

# Escritura atómica:
# primero se escribe un archivo temporal y luego se reemplaza OUTPUT.
# Así se reduce el riesgo de dejar un CSV parcialmente escrito si
# el proceso se interrumpe durante el guardado.
tmp_output = OUTPUT.with_name(
    f"{OUTPUT.stem}.tmp{OUTPUT.suffix}"
)

manifest.to_csv(
    tmp_output,
    index=False,
    encoding="utf-8-sig"
)

tmp_output.replace(
    OUTPUT
)


# ============================================================
# 44. GUARDAR REPORTE AUXILIAR DE FOLDS
# ============================================================

if EXPORT_PATIENT_FOLDS:

    patient_folds_report = (
        beh_patients[
            [
                "patient_id",
                "target_binary",
                "cv_fold",
            ]
        ]
        .copy()
    )

    patient_folds_report[
        "cv_seed"
    ] = CV_RANDOM_STATE

    patient_folds_report[
        "cv_strategy"
    ] = (
        f"StratifiedKFold_patient_level_{N_SPLITS}fold"
    )

    PATIENT_FOLDS_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    patient_folds_report.to_csv(
        PATIENT_FOLDS_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 45. RESULTADO FINAL
# ============================================================

print(
    "\n========================================"
)

print(
    "MANIFEST VALIDADO CORRECTAMENTE"
)

print(
    "========================================"
)

print(
    "\nArchivo generado:"
)

print(
    OUTPUT
)

print(
    "\nCV:"
)

print(
    f"- Estrategia: {N_SPLITS}-fold StratifiedKFold "
    "a nivel paciente"
)

print(
    f"- random_state: {CV_RANDOM_STATE}"
)

print(
    "- BEH: cv_fold 1-5"
)

print(
    "- TAL: cv_fold vacío"
)

print(
    "- split fijo: vacío"
)

if EXPORT_PATIENT_FOLDS:

    print(
        "\nReporte auxiliar generado:"
    )

    print(
        PATIENT_FOLDS_OUTPUT
    )
