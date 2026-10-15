"""Genera una auditoría visual reproducible del aumento afín de BEH.

Reutiliza exactamente las 40 imágenes de la auditoría v2. Solo lee ``raw``;
las hojas de contacto se guardan en ``artifacts/qc`` y la ficha de revisión y
los parámetros en ``metadata/qc/input_pipeline_audit``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont

from src.data.preprocess_behsof import (
    OUTPUT_SIZE,
    TRAIN_AUGMENTATION_PROBABILITY,
    TRAIN_ROTATION_DEGREES,
    TRAIN_SCALE_RANGE,
    TRAIN_TRANSLATION_FRACTION,
    build_affine_augmentation,
    prepare_beh_image,
)
from src.settings import BEHSOF_IMAGES_DIR, PROJECT_ROOT


DEFAULT_SEED = 42
SOURCE_REVIEW = (
    PROJECT_ROOT / "metadata" / "qc" / "preprocessing_audit" / "v2_review.csv"
)
VISUAL_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "qc" / "input_pipeline_audit"
METADATA_OUTPUT_DIR = (
    PROJECT_ROOT / "metadata" / "qc" / "input_pipeline_audit"
)
REVIEW_CSV = METADATA_OUTPUT_DIR / "augmentation_review.csv"
PARAMETERS_JSON = METADATA_OUTPUT_DIR / "augmentation_parameters.json"

CARD_WIDTH = 640
CARD_HEIGHT = 310
SHEET_COLUMNS = 2
SHEET_ROWS = 2
CARDS_PER_SHEET = SHEET_COLUMNS * SHEET_ROWS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audita visualmente el aumento afín confirmado para BEH."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Semilla de la secuencia de aumentos (por defecto: 42).",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_audit_sample() -> pd.DataFrame:
    if not SOURCE_REVIEW.is_file():
        raise FileNotFoundError(f"No existe la revisión v2: {SOURCE_REVIEW}")

    sample = pd.read_csv(SOURCE_REVIEW, dtype={"patient_id": str})
    required = {
        "audit_index",
        "patient_id",
        "target_binary",
        "relative_path",
        "review_status",
    }
    missing = sorted(required.difference(sample.columns))
    if missing:
        raise ValueError(
            "La revisión v2 no contiene las columnas requeridas: "
            + ", ".join(missing)
        )

    if len(sample) != 40 or sample["patient_id"].nunique() != 40:
        raise ValueError(
            "La auditoría del pipeline exige las 40 imágenes de 40 pacientes de v2."
        )
    if not sample["review_status"].astype(str).str.lower().eq("pass").all():
        raise ValueError("La ficha v2 contiene imágenes que no fueron aprobadas.")

    return sample.sort_values("audit_index").reset_index(drop=True)


def make_card(
    original: Image.Image,
    augmented: Image.Image,
    audit_index: int,
    patient_id: str,
    target: int,
    relative_path: str,
) -> Image.Image:
    card = Image.new("RGB", (CARD_WIDTH, CARD_HEIGHT), "white")
    draw = ImageDraw.Draw(card)
    font = load_font(16)
    small_font = load_font(13)

    card.paste(original, (45, 58))
    card.paste(augmented, (371, 58))
    draw.text(
        (10, 8),
        f"#{audit_index:02d} | {patient_id} | clase {target}",
        fill="black",
        font=font,
    )
    draw.text((10, 31), relative_path, fill=(55, 55, 55), font=small_font)
    draw.text((45, 286), "Base determinística", fill="black", font=small_font)
    draw.text((371, 286), "Afín aplicada", fill="black", font=small_font)
    return card


def save_contact_sheets(cards: list[Image.Image]) -> list[str]:
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

        number = offset // CARDS_PER_SHEET + 1
        filename = f"input_pipeline_audit_sheet_{number:02d}.jpg"
        sheet.save(VISUAL_OUTPUT_DIR / filename, quality=94, subsampling=0)
        filenames.append(filename)
    return filenames


def main() -> None:
    args = parse_args()
    if VISUAL_OUTPUT_DIR.exists() or REVIEW_CSV.exists() or PARAMETERS_JSON.exists():
        raise FileExistsError(
            "Ya existe una salida de la auditoría del pipeline. No se sobrescribirá.\n"
            f"Salida visual: {VISUAL_OUTPUT_DIR}\n"
            f"Ficha de revisión: {REVIEW_CSV}\n"
            f"Parámetros: {PARAMETERS_JSON}"
        )

    sample = load_audit_sample()
    affine = build_affine_augmentation()
    torch.manual_seed(args.seed)

    cards: list[Image.Image] = []
    review_rows: list[dict[str, object]] = []
    for row in sample.itertuples(index=False):
        source = resolve_source(str(row.relative_path))
        if not source.is_file():
            raise FileNotFoundError(f"No existe la imagen seleccionada: {source}")

        base = prepare_beh_image(source)
        augmented = affine(base)
        audit_index = int(row.audit_index)
        cards.append(
            make_card(
                original=base,
                augmented=augmented,
                audit_index=audit_index,
                patient_id=str(row.patient_id),
                target=int(row.target_binary),
                relative_path=str(row.relative_path),
            )
        )
        review_rows.append(
            {
                "audit_index": audit_index,
                "patient_id": row.patient_id,
                "target_binary": int(row.target_binary),
                "relative_path": row.relative_path,
                "width": augmented.width,
                "height": augmented.height,
                "anatomy_preserved": "",
                "artifacts_acceptable": "",
                "review_status": "",
                "notes": "",
            }
        )

    VISUAL_OUTPUT_DIR.mkdir(parents=True)
    METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet_filenames = save_contact_sheets(cards)
    pd.DataFrame(review_rows).to_csv(REVIEW_CSV, index=False, encoding="utf-8")

    parameters = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "visual_audit_of_affine_when_forced_to_apply",
        "source_review": str(SOURCE_REVIEW.relative_to(PROJECT_ROOT)),
        "source_review_sha256": sha256(SOURCE_REVIEW),
        "selection": {
            "source": "same_40_images_as_preprocessing_v2_audit",
            "images": int(len(sample)),
            "patients": int(sample["patient_id"].nunique()),
            "seed": args.seed,
        },
        "base_pipeline": {
            "output_size": list(OUTPUT_SIZE),
            "resize_interpolation": "bicubic",
            "mode": "RGB",
        },
        "affine_audited": {
            "forced_application_for_visual_audit": True,
            "training_probability": TRAIN_AUGMENTATION_PROBABILITY,
            "degrees": [-TRAIN_ROTATION_DEGREES, TRAIN_ROTATION_DEGREES],
            "translate_fraction": list(TRAIN_TRANSLATION_FRACTION),
            "scale": list(TRAIN_SCALE_RANGE),
            "shear": None,
            "interpolation": "bilinear",
            "fill": 0,
        },
        "software": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
        },
        "contact_sheets": sheet_filenames,
        "acceptance_rule": (
            "pass only if anatomy_preserved=yes and artifacts_acceptable=yes"
        ),
    }
    PARAMETERS_JSON.write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Hojas de contacto: {VISUAL_OUTPUT_DIR}")
    print(f"Imágenes revisables: {len(review_rows)}")
    print(f"Ficha de revisión: {REVIEW_CSV}")
    print(f"Parámetros: {PARAMETERS_JSON}")


if __name__ == "__main__":
    main()
