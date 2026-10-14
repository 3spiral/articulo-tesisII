import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont

from src.settings import (
    BEHSOF_IMAGES_DIR,
    BEH_OUTLIER_REVIEW_DIR,
    BEH_OUTLIERS_REVIEW_CSV,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = BEHSOF_IMAGES_DIR

OUTPUT_DIR = (
    BEH_OUTLIER_REVIEW_DIR
)

REVIEW_CSV = (
    BEH_OUTLIERS_REVIEW_CSV
)


OUTLIERS = [
    (
        "BEH86221",
        "Image2.jpg",
        "LOW_ENTROPY",
    ),
    (
        "BEH86221",
        "Image14.jpg",
        "LOW_ENTROPY + LOW_CONTENT",
    ),
    (
        "BEH92111",
        "Image14.jpg",
        "LOW_ENTROPY",
    ),
    (
        "BEH86221",
        "Image10.jpg",
        "LOW_ENTROPY + LOW_CONTENT",
    ),
    (
        "BEH86221",
        "Image1.jpg",
        "LOW_ENTROPY",
    ),

    (
        "BEH74231",
        "Image6.jpg",
        "LOW_CONTENT",
    ),
    (
        "BEH77332",
        "Image3.jpg",
        "LOW_CONTENT",
    ),
    (
        "BEH16110",
        "Image14.jpg",
        "LOW_CONTENT",
    ),

    (
        "BEH93231",
        "Image5.jpg",
        "CENTER_DROP",
    ),
    (
        "BEH30111",
        "Image11.jpg",
        "CENTER_DROP",
    ),
    (
        "BEH04020",
        "Image6.jpg",
        "CENTER_DROP",
    ),
    (
        "BEH60111",
        "Image13.jpg",
        "CENTER_DROP",
    ),
    (
        "BEH53231",
        "Image2.jpg",
        "CENTER_DROP",
    ),
]


# ============================================================
# DISEÑO
# ============================================================

COLS = 3

THUMB_W = 360
THUMB_H = 270
TEXT_H = 75

MARGIN = 20
PADDING = 12

CELL_W = THUMB_W
CELL_H = THUMB_H + TEXT_H


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
# DEFINIR ARTEFACTO VISUAL
# ============================================================

OUTPUT_IMAGE = (
    OUTPUT_DIR
    / "BEH_outliers_contact_sheet.jpg"
)


# ============================================================
# PROTECCIÓN CONTRA SOBRESCRITURA
# ============================================================

if OUTPUT_IMAGE.exists():

    raise FileExistsError(
        "Ya existe el contact sheet de outliers BEH:\n"
        f"{OUTPUT_IMAGE}\n\n"
        "No se sobrescribirá automáticamente para preservar "
        "la trazabilidad de la revisión manual.\n\n"
        "Si realmente deseas regenerarlo, mueve o elimina "
        "deliberadamente el archivo existente."
    )


if REVIEW_CSV.exists():

    raise FileExistsError(
        "El archivo de revisión de outliers BEH ya existe:\n"
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
        17,
    )

    font_small = ImageFont.truetype(
        "arial.ttf",
        13,
    )

except Exception:

    font = ImageFont.load_default()
    font_small = ImageFont.load_default()


# ============================================================
# BUSCAR ARCHIVOS
# ============================================================

items = []


for (
    patient_id,
    filename,
    reason,
) in OUTLIERS:

    patient_folder = (
        ROOT
        / patient_id
    )

    if not patient_folder.exists():

        print(
            "CARPETA NO ENCONTRADA:",
            patient_id,
        )

        continue

    matches = [
        p
        for p in patient_folder.rglob("*")
        if (
            p.is_file()
            and
            p.name.lower()
            == filename.lower()
        )
    ]


    if len(
        matches
    ) == 0:

        print(
            "NO ENCONTRADO:",
            patient_id,
            filename,
        )

        continue


    if len(
        matches
    ) > 1:

        raise ValueError(
            "Se encontró más de un archivo "
            "con el mismo nombre para:\n"
            f"{patient_id} / {filename}"
        )


    path = matches[
        0
    ]


    items.append({
        "patient_id":
            patient_id,

        "filename":
            filename,

        "path":
            path,

        "reason":
            reason,
    })


print(
    f"Encontradas: "
    f"{len(items)}/{len(OUTLIERS)}"
)


if len(
    items
) != len(
    OUTLIERS
):

    raise ValueError(
        "No se localizaron todos los outliers esperados. "
        "No se generará el contact sheet."
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
# CONTACT SHEET
# ============================================================

ROWS = (
    len(items)
    + COLS
    - 1
) // COLS


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


for i, item in enumerate(
    items
):

    row = (
        i
        // COLS
    )

    col = (
        i
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

    index = (
        i
        + 1
    )

    draw.text(
        (
            x + 4,
            y + THUMB_H + 4,
        ),
        f"{index:02d} | "
        f"{item['patient_id']}",
        fill="black",
        font=font,
    )

    draw.text(
        (
            x + 4,
            y + THUMB_H + 28,
        ),
        item[
            "filename"
        ],
        fill="black",
        font=font_small,
    )

    draw.text(
        (
            x + 4,
            y + THUMB_H + 48,
        ),
        item[
            "reason"
        ],
        fill="black",
        font=font_small,
    )


sheet.save(
    OUTPUT_IMAGE,
    quality=94,
)


# ============================================================
# CSV DE REVISIÓN
# ============================================================

review_rows = []


for i, item in enumerate(
    items,
    start=1,
):

    review_rows.append({
        "index":
            i,

        "patient_id":
            item[
                "patient_id"
            ],

        "filename":
            item[
                "filename"
            ],

        "relative_path":
            str(
                item[
                    "path"
                ].relative_to(
                    ROOT
                )
            ),

        "outlier_reason":
            item[
                "reason"
            ],

        "qc_label":
            "",

        "notes":
            "",
    })


review_df = pd.DataFrame(
    review_rows
)


review_df.to_csv(
    REVIEW_CSV,
    index=False,
    encoding="utf-8-sig",
)


print(
    "\nContact sheet:",
    OUTPUT_IMAGE
)

print(
    "CSV:",
    REVIEW_CSV
)

print(
    "\nProceso terminado."
)