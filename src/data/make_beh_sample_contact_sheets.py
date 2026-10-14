import math
import random

import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont

from src.settings import (
    BEHSOF_IMAGES_DIR,
    BEHSOF_DEMO_CSV,
    BEH_SAMPLE_CONTACT_SHEETS_DIR,
    BEH_SAMPLE_REVIEW_CSV,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = BEHSOF_IMAGES_DIR

DEMO_CSV = BEHSOF_DEMO_CSV

OUTPUT_DIR = (
    BEH_SAMPLE_CONTACT_SHEETS_DIR
)

REVIEW_CSV = (
    BEH_SAMPLE_REVIEW_CSV
)


RANDOM_SEED = 42

TARGET_PER_GRADE = 15


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# DISEÑO DEL CONTACT SHEET
# ============================================================

COLS = 4
ROWS = 5

IMAGES_PER_SHEET = (
    COLS * ROWS
)

THUMB_W = 300
THUMB_H = 225

TEXT_H = 70

CELL_W = THUMB_W
CELL_H = THUMB_H + TEXT_H

MARGIN = 20
PADDING = 10


# ============================================================
# VALIDAR ENTRADAS
# ============================================================

if not ROOT.exists():

    raise FileNotFoundError(
        "No existe la carpeta de imágenes BEHSOF:\n"
        f"{ROOT}"
    )


if not ROOT.is_dir():

    raise NotADirectoryError(
        "La ruta de imágenes BEHSOF no es una carpeta:\n"
        f"{ROOT}"
    )


if not DEMO_CSV.exists():

    raise FileNotFoundError(
        "No existe Demo_data.csv:\n"
        f"{DEMO_CSV}"
    )


if not DEMO_CSV.is_file():

    raise FileNotFoundError(
        "La ruta de Demo_data.csv no corresponde "
        "a un archivo:\n"
        f"{DEMO_CSV}"
    )


# ============================================================
# CREAR DIRECTORIOS DE SALIDA
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REVIEW_CSV.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# PROTECCIÓN CONTRA SOBRESCRITURA
# ============================================================

existing_sheets = sorted(
    OUTPUT_DIR.glob(
        "BEH_sample_sheet_*.jpg"
    )
)


if existing_sheets:

    raise FileExistsError(
        "Ya existen contact sheets de muestra BEH en:\n"
        f"{OUTPUT_DIR}\n\n"
        "No se regenerarán automáticamente para preservar "
        "la trazabilidad de la revisión manual.\n\n"
        "Si realmente deseas repetir la revisión desde cero, "
        "mueve o elimina deliberadamente esos archivos."
    )


if REVIEW_CSV.exists():

    raise FileExistsError(
        "El archivo de revisión de la muestra BEH ya existe:\n"
        f"{REVIEW_CSV}\n\n"
        "No se sobrescribirá automáticamente porque puede "
        "contener decisiones manuales de QC.\n\n"
        "Si realmente deseas regenerarlo, realiza primero "
        "una copia de seguridad y elimínalo manualmente."
    )


# ============================================================
# FUENTE
# ============================================================

try:

    font = ImageFont.truetype(
        "arial.ttf",
        16,
    )

    font_small = ImageFont.truetype(
        "arial.ttf",
        13,
    )

except Exception:

    font = ImageFont.load_default()
    font_small = ImageFont.load_default()


# ============================================================
# NORMALIZAR ID
# ============================================================

def normalize_patient_id(value):

    value = str(
        value
    ).strip()

    value = value.replace(
        "\\",
        "/",
    )

    value = value.rstrip(
        "/"
    )

    value = value.split(
        "/"
    )[-1]

    return value


# ============================================================
# CARGAR ETIQUETAS
# ============================================================

demo = pd.read_csv(
    DEMO_CSV
)


required_columns = [
    "Patient ID",
    "Fibroscan S",
]


for column in required_columns:

    if column not in demo.columns:

        raise ValueError(
            "Demo_data.csv no contiene "
            f"la columna requerida: {column}"
        )


demo[
    "patient_id"
] = (
    demo[
        "Patient ID"
    ]
    .apply(
        normalize_patient_id
    )
)


demo = demo[
    demo[
        "patient_id"
    ]
    .str.startswith(
        "BEH"
    )
].copy()


# Normalizamos por si Fibroscan S viene como float.
demo[
    "fibroscan_s"
] = pd.to_numeric(
    demo[
        "Fibroscan S"
    ],
    errors="coerce",
)


demo = demo[
    demo[
        "fibroscan_s"
    ]
    .isin(
        [
            0,
            1,
            2,
            3,
        ]
    )
].copy()


labels = dict(
    zip(
        demo[
            "patient_id"
        ],
        demo[
            "fibroscan_s"
        ].astype(
            int
        ),
    )
)


# ============================================================
# RECOLECTAR IMÁGENES BEH
# ============================================================

rows = []


for patient_folder in sorted(
    ROOT.iterdir()
):

    if not patient_folder.is_dir():
        continue

    patient_id = (
        patient_folder.name
    )

    if not patient_id.startswith(
        "BEH"
    ):
        continue

    if patient_id not in labels:
        continue

    grade = labels[
        patient_id
    ]

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

        rows.append({
            "patient_id":
                patient_id,

            "fibroscan_s":
                grade,

            "path":
                path,

            "filename":
                path.name,

            "relative_path":
                str(
                    path.relative_to(
                        ROOT
                    )
                ),
        })


df = pd.DataFrame(
    rows
)


if df.empty:

    raise ValueError(
        "No se encontraron imágenes BEH válidas."
    )


print(
    "\nImágenes BEH encontradas:",
    len(df),
)

print(
    "\nPacientes BEH:",
    df[
        "patient_id"
    ].nunique(),
)

print(
    "\nDistribución de imágenes por Fibroscan S:"
)

print(
    df[
        "fibroscan_s"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# MUESTREO ESTRATIFICADO
# ============================================================

random.seed(
    RANDOM_SEED
)


def sample_grade(
    grade_df,
    target_n,
):
    """
    Prioriza pacientes diferentes.

    Primero escoge una imagen aleatoria por paciente.
    Si todavía faltan imágenes, completa desde el resto.
    """

    if grade_df.empty:

        raise ValueError(
            "No hay imágenes disponibles "
            "para uno de los grados solicitados."
        )


    if len(
        grade_df
    ) < target_n:

        raise ValueError(
            "No existen suficientes imágenes "
            f"para seleccionar {target_n} casos."
        )


    selected_indices = []


    patient_ids = list(
        grade_df[
            "patient_id"
        ].unique()
    )


    random.shuffle(
        patient_ids
    )


    # --------------------------------------------
    # Primera pasada:
    # una imagen por paciente
    # --------------------------------------------

    for patient_id in patient_ids:

        patient_images = grade_df[
            grade_df[
                "patient_id"
            ] == patient_id
        ]

        index = random.choice(
            list(
                patient_images.index
            )
        )

        selected_indices.append(
            index
        )

        if (
            len(
                selected_indices
            )
            >= target_n
        ):
            break


    # --------------------------------------------
    # Segunda pasada:
    # completar si faltan imágenes
    # --------------------------------------------

    if (
        len(
            selected_indices
        )
        < target_n
    ):

        remaining = [
            index
            for index
            in grade_df.index
            if index
            not in selected_indices
        ]

        random.shuffle(
            remaining
        )

        needed = (
            target_n
            - len(
                selected_indices
            )
        )

        selected_indices.extend(
            remaining[
                :needed
            ]
        )


    return grade_df.loc[
        selected_indices
    ]


samples = []


for grade in [
    0,
    1,
    2,
    3,
]:

    grade_df = df[
        df[
            "fibroscan_s"
        ] == grade
    ]

    sample = sample_grade(
        grade_df,
        TARGET_PER_GRADE,
    )

    samples.append(
        sample
    )


sample_df = pd.concat(
    samples,
    ignore_index=True,
)


# Mezclar visualmente el orden para no revisar
# 15 imágenes consecutivas del mismo grado.
sample_df = sample_df.sample(
    frac=1,
    random_state=RANDOM_SEED,
).reset_index(
    drop=True
)


sample_df[
    "sample_index"
] = (
    sample_df.index
    + 1
)


print(
    "\n======================================"
)

print(
    "MUESTRA SELECCIONADA"
)

print(
    "======================================"
)

print(
    "\nTotal:",
    len(sample_df),
)

print(
    "\nPor grado:"
)

print(
    sample_df[
        "fibroscan_s"
    ]
    .value_counts()
    .sort_index()
)

print(
    "\nPacientes diferentes por grado:"
)

print(
    sample_df.groupby(
        "fibroscan_s"
    )[
        "patient_id"
    ].nunique()
)


# ============================================================
# MINIATURA
# ============================================================

def make_thumbnail(path):

    try:

        with Image.open(
            path
        ) as source_img:

            img = source_img.convert(
                "RGB"
            )

        thumb = ImageOps.contain(
            img,
            (
                THUMB_W,
                THUMB_H,
            )
        )

        background = Image.new(
            "RGB",
            (
                THUMB_W,
                THUMB_H,
            ),
            "black",
        )

        x = (
            THUMB_W
            - thumb.width
        ) // 2

        y = (
            THUMB_H
            - thumb.height
        ) // 2

        background.paste(
            thumb,
            (
                x,
                y,
            )
        )

        return background

    except Exception:

        error_img = Image.new(
            "RGB",
            (
                THUMB_W,
                THUMB_H,
            ),
            "white",
        )

        draw = ImageDraw.Draw(
            error_img
        )

        draw.text(
            (
                10,
                10,
            ),
            "ERROR",
            fill="black",
            font=font,
        )

        return error_img


# ============================================================
# CONTACT SHEETS
# ============================================================

num_sheets = math.ceil(
    len(sample_df)
    / IMAGES_PER_SHEET
)


for sheet_index in range(
    num_sheets
):

    start = (
        sheet_index
        * IMAGES_PER_SHEET
    )

    end = min(
        start
        + IMAGES_PER_SHEET,
        len(sample_df),
    )

    batch = sample_df.iloc[
        start:end
    ]

    sheet_width = (
        MARGIN * 2
        + COLS * (
            CELL_W
            + PADDING
        )
    )

    sheet_height = (
        MARGIN * 2
        + ROWS * (
            CELL_H
            + PADDING
        )
    )

    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    for local_index, (
        _,
        item,
    ) in enumerate(
        batch.iterrows()
    ):

        row = (
            local_index
            // COLS
        )

        col = (
            local_index
            % COLS
        )

        x = (
            MARGIN
            + col * (
                CELL_W
                + PADDING
            )
        )

        y = (
            MARGIN
            + row * (
                CELL_H
                + PADDING
            )
        )

        thumb = make_thumbnail(
            item[
                "path"
            ]
        )

        sheet.paste(
            thumb,
            (
                x,
                y,
            )
        )

        draw.rectangle(
            [
                x,
                y,
                x + THUMB_W,
                y + THUMB_H,
            ],
            outline="black",
            width=1,
        )

        label_1 = (
            f"{int(item['sample_index']):03d} | "
            f"{item['patient_id']}"
        )

        filename = str(
            item[
                "filename"
            ]
        )

        if len(
            filename
        ) > 36:

            filename = (
                filename[:33]
                + "..."
            )

        draw.text(
            (
                x + 4,
                y + THUMB_H + 4,
            ),
            label_1,
            fill="black",
            font=font,
        )

        draw.text(
            (
                x + 4,
                y + THUMB_H + 29,
            ),
            filename,
            fill="black",
            font=font_small,
        )


    output_path = (
        OUTPUT_DIR
        / f"BEH_sample_sheet_{sheet_index + 1:02d}.jpg"
    )

    sheet.save(
        output_path,
        quality=92,
    )

    print(
        "Generado:",
        output_path
    )


# ============================================================
# CSV PARA REVISIÓN
# ============================================================

review_df = sample_df[
    [
        "sample_index",
        "patient_id",
        "fibroscan_s",
        "filename",
        "relative_path",
    ]
].copy()


review_df[
    "qc_label"
] = ""

review_df[
    "notes"
] = ""


review_df.to_csv(
    REVIEW_CSV,
    index=False,
    encoding="utf-8-sig",
)


print(
    "\nCSV generado:",
    REVIEW_CSV
)

print(
    "\nProceso terminado."
)