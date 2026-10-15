"""Preprocesamiento y transformaciones de entrada para las ecografías BEH.

El módulo carga cada imagen en RGB, elimina la interfaz periférica mediante
una regla geométrica fija, redimensiona a 224 × 224 y construye los pipelines
de entrenamiento y validación. El entrenamiento incorpora un aumento afín
conservador; la validación permanece determinística. Ambos pipelines convierten
la salida a ``float32`` y aplican la normalización de ImageNet.

Las imágenes originales se leen sin modificarlas y todas las operaciones se
realizan en memoria.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import TypeAlias

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import InterpolationMode, v2


ImageInput: TypeAlias = Image.Image | str | PathLike[str]

EXPECTED_SOURCE_SIZE = (1024, 768)
OUTPUT_SIZE = (224, 224)

CROP_BOX = (128, 0, 896, 768)
MASK_TOP_END = 48
MASK_BOTTOM_START = 720
MASK_BOTTOM_LEFT_BOX = (0, 688, 64, 720)

TRAIN_AUGMENTATION_PROBABILITY = 0.5
TRAIN_ROTATION_DEGREES = 5.0
TRAIN_TRANSLATION_FRACTION = (0.03, 0.03)
TRAIN_SCALE_RANGE = (0.95, 1.05)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb_image(source: ImageInput) -> Image.Image:
    """Carga una imagen, corrige EXIF y devuelve una copia RGB independiente."""

    if isinstance(source, Image.Image):
        return ImageOps.exif_transpose(source).convert("RGB")

    path = Path(source)
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def clean_beh_interface(image: Image.Image) -> Image.Image:
    """Aplica la regla geométrica v2 confirmada para las imágenes BEH."""

    if image.size != EXPECTED_SOURCE_SIZE:
        raise ValueError(
            f"Resolución BEH inesperada: {image.size}; "
            f"se esperaba {EXPECTED_SOURCE_SIZE}."
        )

    processed = image.crop(CROP_BOX).copy()
    draw = ImageDraw.Draw(processed)
    draw.rectangle((0, 0, processed.width, MASK_TOP_END - 1), fill="black")
    draw.rectangle(
        (0, MASK_BOTTOM_START, processed.width, processed.height),
        fill="black",
    )
    left, top, right, bottom = MASK_BOTTOM_LEFT_BOX
    draw.rectangle((left, top, right - 1, bottom - 1), fill="black")
    return processed


def prepare_beh_image(source: ImageInput) -> Image.Image:
    """Ejecuta la parte determinística común y devuelve una imagen RGB 224×224."""

    image = load_rgb_image(source)
    cleaned = clean_beh_interface(image)
    return cleaned.resize(OUTPUT_SIZE, resample=Image.Resampling.BICUBIC)


def build_affine_augmentation() -> v2.RandomAffine:
    """Construye la transformación afín sin su probabilidad de aplicación."""

    return v2.RandomAffine(
        degrees=TRAIN_ROTATION_DEGREES,
        translate=TRAIN_TRANSLATION_FRACTION,
        scale=TRAIN_SCALE_RANGE,
        shear=None,
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )


def build_train_spatial_transform() -> v2.RandomApply:
    """Aplica la transformación afín con probabilidad 0.5 solo en entrenamiento."""

    return v2.RandomApply(
        [build_affine_augmentation()],
        p=TRAIN_AUGMENTATION_PROBABILITY,
    )


def build_tensor_transform() -> v2.Compose:
    """Convierte a float32, escala a [0, 1] y normaliza con ImageNet."""

    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_train_transform() -> v2.Compose:
    """Pipeline completo destinado exclusivamente a muestras de entrenamiento."""

    return v2.Compose(
        [
            prepare_beh_image,
            build_train_spatial_transform(),
            build_tensor_transform(),
        ]
    )


def build_validation_transform() -> v2.Compose:
    """Pipeline determinístico para validación y evaluación."""

    return v2.Compose(
        [
            prepare_beh_image,
            build_tensor_transform(),
        ]
    )
