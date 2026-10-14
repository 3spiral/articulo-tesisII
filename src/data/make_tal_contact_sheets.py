import csv
import math

from PIL import Image, ImageOps, ImageDraw, ImageFont

from src.settings import (
    BEHSOF_IMAGES_DIR,
    TAL_CONTACT_SHEETS_DIR,
    TAL_CONTACT_SHEET_INDEX,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = BEHSOF_IMAGES_DIR

OUTPUT_DIR = TAL_CONTACT_SHEETS_DIR

INDEX_CSV = TAL_CONTACT_SHEET_INDEX


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# Diseño
COLS = 4
ROWS = 8

IMAGES_PER_SHEET = (
    COLS * ROWS
)

THUMB_W = 280
THUMB_H = 210

TEXT_H = 55

CELL_W = THUMB_W
CELL_H = THUMB_H + TEXT_H

MARGIN = 20
PADDING = 8


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
    exist_ok=True
)

INDEX_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PROTECCIÓN CONTRA SOBRESCRITURA
# ============================================================

existing_sheets = sorted(
    OUTPUT_DIR.glob(
        "sheet_*.jpg"
    )
)


if existing_sheets:

    raise FileExistsError(
        "Ya existen contact sheets TAL en:\n"
        f"{OUTPUT_DIR}\n\n"
        "No se regenerarán automáticamente para preservar "
        "la trazabilidad de la revisión manual.\n\n"
        "Si realmente deseas regenerarlos, mueve o elimina "
        "deliberadamente los contact sheets existentes."
    )


if INDEX_CSV.exists():

    raise FileExistsError(
        "El archivo de revisión TAL ya existe:\n"
        f"{INDEX_CSV}\n\n"
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
        16
    )

    font_small = ImageFont.truetype(
        "arial.ttf",
        13
    )

except Exception:

    font = ImageFont.load_default()
    font_small = ImageFont.load_default()


# ============================================================
# RECOLECTAR IMÁGENES TAL
# ============================================================

items = []


for patient_folder in sorted(
    ROOT.iterdir()
):

    if not patient_folder.is_dir():
        continue

    patient_id = (
        patient_folder.name
    )

    if not patient_id.startswith(
        "TAL"
    ):
        continue

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

        items.append({
            "patient_id":
                patient_id,

            "path":
                path,

            "relative_path":
                str(
                    path.relative_to(
                        ROOT
                    )
                ),
        })


if not items:

    raise ValueError(
        "No se encontraron imágenes TAL."
    )


print(
    "Imágenes TAL encontradas:",
    len(items)
)


# ============================================================
# FUNCIÓN MINIATURA
# ============================================================

def make_thumbnail(path):

    try:

        with Image.open(
            path
        ) as source_img:

            img = source_img.convert(
                "RGB"
            )

        # Mantiene proporción y rellena
        # sin deformar la ecografía.
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
# CREAR PÁGINAS
# ============================================================

num_sheets = math.ceil(
    len(items)
    / IMAGES_PER_SHEET
)


print(
    "Contact sheets:",
    num_sheets
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
        len(items),
    )

    batch = items[
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

    for local_index, item in enumerate(
        batch
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
            item["path"]
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

        global_index = (
            start
            + local_index
            + 1
        )

        label_1 = (
            f"{global_index:03d} | "
            f"{item['patient_id']}"
        )

        filename = (
            item["path"].name
        )

        if len(
            filename
        ) > 34:

            filename = (
                filename[:31]
                + "..."
            )

        draw.text(
            (
                x + 3,
                y + THUMB_H + 4,
            ),
            label_1,
            fill="black",
            font=font,
        )

        draw.text(
            (
                x + 3,
                y + THUMB_H + 26,
            ),
            filename,
            fill="black",
            font=font_small,
        )

    output_path = (
        OUTPUT_DIR
        / f"sheet_{sheet_index + 1:02d}.jpg"
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
# GENERAR ÍNDICE CSV
# ============================================================

with open(
    INDEX_CSV,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.writer(
        f
    )

    writer.writerow([
        "index",
        "patient_id",
        "filename",
        "relative_path",
        "sheet_number",
        "qc_label",
        "notes",
    ])

    for i, item in enumerate(
        items,
        start=1,
    ):

        sheet_number = (
            (i - 1)
            // IMAGES_PER_SHEET
            + 1
        )

        writer.writerow([
            i,
            item["patient_id"],
            item["path"].name,
            item["relative_path"],
            sheet_number,
            "",
            "",
        ])


print(
    "\nÍndice generado:",
    INDEX_CSV
)

print(
    "\nProceso terminado."
)