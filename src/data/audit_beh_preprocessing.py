"""Este Script genera una auditoría visual reproducible del preprocesamiento BEH.

El script solo lee imágenes originales seleccionadas por el manifiesto. Las
hojas de contacto se escriben en ``artifacts/qc`` y la evidencia documental en
``metadata/qc/preprocessing_audit``. No modifica el dataset ni los reportes de
control de calidad existentes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.settings import BEHSOF_IMAGES_DIR, BEHSOF_MANIFEST_CSV, PROJECT_ROOT


RANDOM_SEED = 42
PATIENTS_PER_CLASS = 20

CROP_LEFT = 128
CROP_TOP = 0
CROP_RIGHT = 896
CROP_BOTTOM = 768
MASK_TOP_END = 48
MASK_BOTTOM_START = 720
MASK_BOTTOM_LEFT_RIGHT = 64
MASK_BOTTOM_LEFT_TOP = 688

EXPECTED_SOURCE_SIZE = (1024, 768)
VISUAL_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "qc"
METADATA_OUTPUT_ROOT = PROJECT_ROOT / "metadata" / "qc" / "preprocessing_audit"

CARD_WIDTH = 620
CARD_HEIGHT = 335
SHEET_COLUMNS = 2
SHEET_ROWS = 2
CARDS_PER_SHEET = SHEET_COLUMNS * SHEET_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera hojas de contacto para auditar el recorte BEH.",
    )
    parser.add_argument(
        "--patients-per-class",
        type=int,
        default=PATIENTS_PER_CLASS,
        help="Pacientes distintos seleccionados por clase (por defecto: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Semilla de la selección determinística (por defecto: 42).",
    )
    parser.add_argument(
        "--candidate",
        choices=("v1", "v2"),
        default="v2",
        help="Regla candidata que se desea auditar (por defecto: v2).",
    )
    return parser.parse_args()


def load_font(
    size: int,
) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def parse_boolean(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def load_eligible_manifest() -> pd.DataFrame:
    if not BEHSOF_MANIFEST_CSV.is_file():
        raise FileNotFoundError(
            f"No existe el manifiesto BEHSOF: {BEHSOF_MANIFEST_CSV}"
        )

    manifest = pd.read_csv(BEHSOF_MANIFEST_CSV, dtype={"patient_id": str})
    required = {
        "patient_id",
        "center",
        "relative_path",
        "target_binary",
        "include_final",
        "patient_eligible",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(
            "El manifiesto no contiene las columnas requeridas: "
            + ", ".join(missing)
        )

    eligible = manifest.loc[
        manifest["center"].astype(str).str.upper().eq("BEH")
        & parse_boolean(manifest["include_final"])
        & parse_boolean(manifest["patient_eligible"])
    ].copy()

    if eligible.empty:
        raise ValueError("El manifiesto no contiene imágenes BEH elegibles.")

    eligible["target_binary"] = pd.to_numeric(
        eligible["target_binary"], errors="raise"
    ).astype(int)
    invalid_targets = sorted(set(eligible["target_binary"]) - {0, 1})
    if invalid_targets:
        raise ValueError(f"Se encontraron targets no binarios: {invalid_targets}")

    targets_per_patient = eligible.groupby("patient_id")["target_binary"].nunique()
    inconsistent = targets_per_patient[targets_per_patient.ne(1)].index.tolist()
    if inconsistent:
        raise ValueError(
            "Hay pacientes con más de un target: " + ", ".join(inconsistent)
        )

    return eligible


def select_sample(
    eligible: pd.DataFrame,
    patients_per_class: int,
    seed: int,
) -> pd.DataFrame:
    if patients_per_class < 1:
        raise ValueError("--patients-per-class debe ser mayor que cero.")

    rng = random.Random(seed)
    selected_rows: list[pd.Series] = []

    patient_targets = (
        eligible[["patient_id", "target_binary"]]
        .drop_duplicates()
        .sort_values(["target_binary", "patient_id"])
    )

    for target in (0, 1):
        patients = patient_targets.loc[
            patient_targets["target_binary"].eq(target), "patient_id"
        ].tolist()
        if len(patients) < patients_per_class:
            raise ValueError(
                f"La clase {target} solo tiene {len(patients)} pacientes; "
                f"no se pueden seleccionar {patients_per_class}."
            )

        for patient_id in sorted(rng.sample(patients, patients_per_class)):
            patient_rows = (
                eligible.loc[eligible["patient_id"].eq(patient_id)]
                .sort_values("relative_path")
                .reset_index(drop=True)
            )
            selected_rows.append(
                patient_rows.iloc[rng.randrange(len(patient_rows))]
            )

    sample = pd.DataFrame(selected_rows).reset_index(drop=True)
    sample.insert(0, "audit_index", range(1, len(sample) + 1))
    return sample


def resolve_source(relative_path: str) -> Path:
    normalized = str(relative_path).replace("\\", "/")
    source = (BEHSOF_IMAGES_DIR / Path(normalized)).resolve()
    try:
        source.relative_to(BEHSOF_IMAGES_DIR.resolve())
    except ValueError as error:
        raise ValueError(
            f"La ruta sale de la carpeta de imágenes BEHSOF: {relative_path}"
        ) from error
    return source


def apply_candidate_preprocessing(
    image: Image.Image,
    candidate: str,
) -> Image.Image:
    processed = image.crop(
        (CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM)
    ).copy()
    draw = ImageDraw.Draw(processed)
    draw.rectangle(
        (0, 0, processed.width, MASK_TOP_END - 1),
        fill="black",
    )
    draw.rectangle(
        (0, MASK_BOTTOM_START, processed.width, processed.height),
        fill="black",
    )
    if candidate == "v2":
        draw.rectangle(
            (
                0,
                MASK_BOTTOM_LEFT_TOP,
                MASK_BOTTOM_LEFT_RIGHT - 1,
                MASK_BOTTOM_START - 1,
            ),
            fill="black",
        )
    return processed


def make_card(
    image: Image.Image,
    processed: Image.Image,
    audit_index: int,
    patient_id: str,
    target: int,
    relative_path: str,
    candidate: str,
) -> Image.Image:
    card = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), "white")
    draw = ImageDraw.Draw(card)
    font = load_font(16)
    small_font = load_font(13)

    original_overlay = image.copy()
    overlay_draw = ImageDraw.Draw(original_overlay)
    overlay_draw.rectangle(
        (CROP_LEFT, CROP_TOP, CROP_RIGHT - 1, CROP_BOTTOM - 1),
        outline=(255, 0, 0),
        width=4,
    )
    overlay_draw.line(
        (CROP_LEFT, MASK_TOP_END, CROP_RIGHT - 1, MASK_TOP_END),
        fill=(255, 220, 0),
        width=4,
    )
    if candidate == "v2":
        overlay_draw.rectangle(
            (
                CROP_LEFT,
                MASK_BOTTOM_LEFT_TOP,
                CROP_LEFT + MASK_BOTTOM_LEFT_RIGHT - 1,
                MASK_BOTTOM_START - 1,
            ),
            outline=(255, 0, 255),
            width=4,
        )
    overlay_draw.line(
        (CROP_LEFT, MASK_BOTTOM_START, CROP_RIGHT - 1, MASK_BOTTOM_START),
        fill=(255, 220, 0),
        width=4,
    )

    original_thumb = ImageOps.contain(original_overlay, (320, 240))
    processed_thumb = ImageOps.contain(processed, (240, 240))
    card.paste(original_thumb, (10, 65))
    card.paste(processed_thumb, (350, 65))

    title = f"#{audit_index:02d} | {patient_id} | clase {target}"
    draw.text((10, 8), title, fill="black", font=font)
    draw.text(
        (10, 31),
        str(relative_path),
        fill=(55, 55, 55),
        font=small_font,
    )
    draw.text(
        (10, 48),
        "Rojo: recorte | Amarillo: máscara | Magenta: máscara v2",
        fill=(55, 55, 55),
        font=small_font,
    )
    draw.text((10, 310), "Original + límites", fill="black", font=small_font)
    draw.text((350, 310), "Resultado candidato", fill="black", font=small_font)
    return card


def save_contact_sheets(
    cards: list[Image.Image],
    output_dir: Path,
) -> list[str]:
    filenames: list[str] = []
    for offset in range(0, len(cards), CARDS_PER_SHEET):
        batch = cards[offset : offset + CARDS_PER_SHEET]
        sheet = Image.new(
            "RGB",
            (CARD_WIDTH * SHEET_COLUMNS, CARD_HEIGHT * SHEET_ROWS),
            (225, 225, 225),
        )
        for position, card in enumerate(batch):
            x = (position % SHEET_COLUMNS) * CARD_WIDTH
            y = (position // SHEET_COLUMNS) * CARD_HEIGHT
            sheet.paste(card, (x, y))

        sheet_number = offset // CARDS_PER_SHEET + 1
        filename = f"preprocessing_audit_sheet_{sheet_number:02d}.jpg"
        sheet.save(output_dir / filename, quality=94, subsampling=0)
        filenames.append(filename)
    return filenames


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    output_dir = VISUAL_OUTPUT_ROOT / f"preprocessing_audit_{args.candidate}"
    review_csv = METADATA_OUTPUT_ROOT / f"{args.candidate}_review.csv"
    parameters_json = METADATA_OUTPUT_ROOT / f"{args.candidate}_parameters.json"

    if output_dir.exists() or review_csv.exists() or parameters_json.exists():
        raise FileExistsError(
            "Ya existe una salida visual o documental para esta candidata.\n"
            f"Salida visual: {output_dir}\n"
            f"Ficha de revisión: {review_csv}\n"
            f"Parámetros: {parameters_json}\n"
            "No se sobrescribirá ninguna evidencia. Muévela o elimínala "
            "deliberadamente si deseas generar una auditoría nueva."
        )

    eligible = load_eligible_manifest()
    sample = select_sample(eligible, args.patients_per_class, args.seed)

    loaded: list[tuple[pd.Series, Image.Image, Image.Image]] = []
    for _, row in sample.iterrows():
        source = resolve_source(row["relative_path"])
        if not source.is_file():
            raise FileNotFoundError(f"No existe la imagen seleccionada: {source}")

        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if image.size != EXPECTED_SOURCE_SIZE:
            raise ValueError(
                f"Resolución inesperada en {source}: {image.size}; "
                f"se esperaba {EXPECTED_SOURCE_SIZE}."
            )

        processed = apply_candidate_preprocessing(image, args.candidate)
        loaded.append((row, image, processed))

    output_dir.mkdir(parents=True)
    METADATA_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    cards: list[Image.Image] = []
    review_rows: list[dict[str, object]] = []
    for row, image, processed in loaded:
        audit_index = int(row["audit_index"])
        cards.append(
            make_card(
                image=image,
                processed=processed,
                audit_index=audit_index,
                patient_id=str(row["patient_id"]),
                target=int(row["target_binary"]),
                relative_path=str(row["relative_path"]),
                candidate=args.candidate,
            )
        )
        review_rows.append(
            {
                "audit_index": audit_index,
                "patient_id": row["patient_id"],
                "target_binary": int(row["target_binary"]),
                "relative_path": row["relative_path"],
                "source_width": image.width,
                "source_height": image.height,
                "processed_width": processed.width,
                "processed_height": processed.height,
                "peripheral_interface_removed": "",
                "internal_annotation_present": "",
                "anatomy_preserved": "",
                "review_status": "",
                "notes": "",
            }
        )

    sheet_filenames = save_contact_sheets(cards, output_dir)
    pd.DataFrame(review_rows).to_csv(
        review_csv,
        index=False,
        encoding="utf-8",
    )

    parameters = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": args.candidate,
        "manifest": str(BEHSOF_MANIFEST_CSV.relative_to(PROJECT_ROOT)),
        "manifest_sha256": sha256(BEHSOF_MANIFEST_CSV),
        "eligibility_filter": {
            "center": "BEH",
            "include_final": True,
            "patient_eligible": True,
        },
        "eligible_images": int(len(eligible)),
        "eligible_patients": int(eligible["patient_id"].nunique()),
        "selection": {
            "seed": args.seed,
            "patients_per_class": args.patients_per_class,
            "images_per_patient": 1,
            "sample_images": int(len(sample)),
            "sample_patients": int(sample["patient_id"].nunique()),
        },
        "candidate_transform": {
            "crop_box": [CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM],
            "top_mask_rows": [0, MASK_TOP_END],
            "bottom_mask_rows": [MASK_BOTTOM_START, CROP_BOTTOM],
            "bottom_left_mask": (
                [
                    0,
                    MASK_BOTTOM_LEFT_TOP,
                    MASK_BOTTOM_LEFT_RIGHT,
                    MASK_BOTTOM_START,
                ]
                if args.candidate == "v2"
                else None
            ),
            "processed_size": [CROP_RIGHT - CROP_LEFT, CROP_BOTTOM - CROP_TOP],
        },
        "contact_sheets": sheet_filenames,
    }
    parameters_json.write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Hojas de contacto generadas en: {output_dir}")
    print(f"Imágenes revisables: {len(review_rows)}")
    print(f"Hojas de contacto: {len(sheet_filenames)}")
    print(f"Ficha de revisión: {review_csv}")
    print(f"Parámetros: {parameters_json}")


if __name__ == "__main__":
    main()
