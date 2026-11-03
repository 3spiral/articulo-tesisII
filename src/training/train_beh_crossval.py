"""Entrenamiento preespecificado del experimento corregido del artículo.

Cada ejecución corresponde a una arquitectura, protocolo, repetición y fold.
Las asignaciones y semillas se leen exclusivamente de los productos canónicos
de las fases 1 y 2. El script no modifica las imágenes originales ni selecciona
checkpoints o configuraciones a partir del rendimiento de validación.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)

from src.data.preprocess_behsof import build_train_transform, build_validation_transform
from src.settings import BEHSOF_IMAGES_DIR, DATA_ROOT, PROJECT_ROOT


ANALYTIC_COHORT_CSV = PROJECT_ROOT / "metadata" / "BEH_analytic_cohort.csv"
IMAGE_FOLDS_CSV = PROJECT_ROOT / "metadata" / "folds" / "BEH_image_folds.csv"
PATIENT_FOLDS_CSV = PROJECT_ROOT / "metadata" / "folds" / "BEH_patient_folds.csv"
SEED_SCHEDULE_CSV = (
    PROJECT_ROOT / "metadata" / "qc" / "partition_folds" / "seed_schedule.csv"
)
PARTITION_PARAMETERS_JSON = (
    PROJECT_ROOT / "metadata" / "qc" / "partition_folds" / "parameters.json"
)

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "training"
DEFAULT_CHECKPOINT_ROOT = DATA_ROOT / "models" / "article"
DEFAULT_WEIGHTS_CACHE = DATA_ROOT / "models" / "torch_cache"

ARCHITECTURES = ("efficientnet_b0", "resnet50")
PROTOCOLS = ("image_level", "patient_level")
N_REPETITIONS = 5
N_FOLDS = 5
EXPECTED_IMAGES = 1_371
EXPECTED_PATIENTS = 93

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
RESNET50_PILOT_BATCH_SIZES = (32, 16, 8)
FINAL_BATCH_SIZES = {"efficientnet_b0": 32, "resnet50": 32}
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_NUM_WORKERS = 0

WEIGHT_NAMES = {
    "efficientnet_b0": "EfficientNet_B0_Weights.IMAGENET1K_V1",
    "resnet50": "ResNet50_Weights.IMAGENET1K_V2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_boolean(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    parsed = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if parsed.isna().any():
        values = sorted(normalized.loc[parsed.isna()].unique())
        raise ValueError(f"Valores booleanos no reconocidos: {values}")
    return parsed.astype(bool)


class BehDataset(Dataset):
    """Resuelve rutas BEH y transforma las imágenes en memoria."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: Path,
        transform: Callable,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.image_root = image_root.resolve()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve_image(self, relative_path: str) -> Path:
        path = (self.image_root / relative_path).resolve()
        if not path.is_relative_to(self.image_root):
            raise ValueError(f"Ruta fuera de la raíz BEH: {relative_path}")
        if not path.is_file():
            raise FileNotFoundError(f"No existe la imagen BEH: {path}")
        return path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        row = self.frame.iloc[index]
        path = self._resolve_image(str(row["relative_path"]))
        with Image.open(path) as opened:
            tensor = self.transform(opened)
        label = torch.tensor(float(row["target_binary"]), dtype=torch.float32)
        return tensor, label, str(row["relative_path"]), str(row["patient_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena las arquitecturas preespecificadas en los folds corregidos."
    )
    parser.add_argument(
        "--architecture", choices=(*ARCHITECTURES, "both"), default="both"
    )
    parser.add_argument("--protocol", choices=(*PROTOCOLS, "both"), default="both")
    parser.add_argument(
        "--repetition", choices=("1", "2", "3", "4", "5", "all"), default="all"
    )
    parser.add_argument(
        "--fold", choices=("1", "2", "3", "4", "5", "all"), default="all"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Solo permite 32 en ejecuciones finales. En un piloto de ResNet50 "
            "admite la contingencia preespecificada 32, 16 u 8."
        ),
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--weights-cache", type=Path, default=DEFAULT_WEIGHTS_CACHE)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Ejecuta una sola combinación durante una época con fines técnicos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar productos de una ejecución con la misma identidad.",
    )
    return parser.parse_args()


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_analytic_cohort(path: Path = ANALYTIC_COHORT_CSV) -> pd.DataFrame:
    cohort = pd.read_csv(path, dtype={"patient_id": str, "sha256": str})
    required = {
        "relative_path",
        "patient_id",
        "target_binary",
        "sha256",
        "analytic_include",
    }
    missing = sorted(required.difference(cohort.columns))
    if missing:
        raise ValueError("Faltan columnas en la cohorte analítica: " + ", ".join(missing))
    cohort = cohort.loc[parse_boolean(cohort["analytic_include"])].copy()
    cohort["relative_path"] = cohort["relative_path"].astype(str).str.replace(
        "\\", "/", regex=False
    )
    cohort["patient_id"] = cohort["patient_id"].astype(str)
    cohort["target_binary"] = pd.to_numeric(
        cohort["target_binary"], errors="raise"
    ).astype(int)
    cohort["sha256"] = cohort["sha256"].astype(str)
    cohort = cohort.sort_values("relative_path").reset_index(drop=True)
    if len(cohort) != EXPECTED_IMAGES:
        raise ValueError(f"La cohorte incluida no contiene {EXPECTED_IMAGES} imágenes.")
    if cohort["patient_id"].nunique() != EXPECTED_PATIENTS:
        raise ValueError(f"La cohorte incluida no contiene {EXPECTED_PATIENTS} pacientes.")
    if cohort["relative_path"].duplicated().any() or cohort["sha256"].duplicated().any():
        raise ValueError("La cohorte incluida contiene rutas o hashes duplicados.")
    if cohort.groupby("patient_id")["target_binary"].nunique().max() != 1:
        raise ValueError("Un paciente analítico tiene más de una clase.")
    return cohort[["relative_path", "patient_id", "target_binary", "sha256"]]


def load_seed_schedule(path: Path = SEED_SCHEDULE_CSV) -> pd.DataFrame:
    schedule = pd.read_csv(path)
    required = {
        "repetition",
        "root_seed",
        "fold",
        "image_partition_seed",
        "patient_partition_seed",
        "training_seed",
    }
    missing = sorted(required.difference(schedule.columns))
    if missing:
        raise ValueError("Faltan columnas en el calendario de semillas: " + ", ".join(missing))
    for column in required:
        schedule[column] = pd.to_numeric(schedule[column], errors="raise").astype(int)
    if len(schedule) != N_REPETITIONS * N_FOLDS:
        raise ValueError("El calendario no contiene las 25 combinaciones esperadas.")
    if schedule[["repetition", "fold"]].duplicated().any():
        raise ValueError("El calendario repite una combinación de repetición y fold.")
    return schedule.sort_values(["repetition", "fold"]).reset_index(drop=True)


def _validate_common_fold_columns(table: pd.DataFrame, protocol: str) -> pd.DataFrame:
    required = {
        "repetition",
        "root_seed",
        "partition_seed",
        "patient_id",
        "target_binary",
        "cv_fold",
        "training_seed",
        "cv_strategy",
    }
    required.update(
        {"relative_path", "sha256"}
        if protocol == "image_level"
        else {"image_count"}
    )
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError("Faltan columnas en los folds: " + ", ".join(missing))
    for column in (
        "repetition",
        "root_seed",
        "partition_seed",
        "target_binary",
        "cv_fold",
        "training_seed",
    ):
        table[column] = pd.to_numeric(table[column], errors="raise").astype(int)
    table["patient_id"] = table["patient_id"].astype(str)
    if set(table["repetition"].unique()) != set(range(1, N_REPETITIONS + 1)):
        raise ValueError("Los folds no contienen exactamente las cinco repeticiones.")
    if set(table["cv_fold"].unique()) != set(range(1, N_FOLDS + 1)):
        raise ValueError("Los folds no contienen exactamente los cinco folds.")
    identity_columns = (
        ["repetition", "relative_path"]
        if protocol == "image_level"
        else ["repetition", "patient_id"]
    )
    if table.duplicated(identity_columns).any():
        raise ValueError("Los folds contienen asignaciones duplicadas.")
    return table


def load_fold_assignments(
    protocol: str,
    cohort: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Protocolo no reconocido: {protocol}")
    cohort = load_analytic_cohort() if cohort is None else cohort.copy()
    path = IMAGE_FOLDS_CSV if protocol == "image_level" else PATIENT_FOLDS_CSV
    table = _validate_common_fold_columns(
        pd.read_csv(path, dtype={"patient_id": str, "sha256": str}), protocol
    )
    expected_strategy = (
        "StratifiedKFold_image_level_5fold"
        if protocol == "image_level"
        else "StratifiedKFold_patient_level_5fold"
    )
    if not table["cv_strategy"].eq(expected_strategy).all():
        raise ValueError(f"Estrategia inesperada en {path.name}.")

    if protocol == "image_level":
        expected_rows = EXPECTED_IMAGES * N_REPETITIONS
        if len(table) != expected_rows:
            raise ValueError(f"Se esperaban {expected_rows} asignaciones por imagen.")
        checked = table.merge(
            cohort,
            on="relative_path",
            how="outer",
            validate="many_to_one",
            suffixes=("", "_cohort"),
            indicator=True,
        )
        if not checked["_merge"].eq("both").all():
            raise ValueError("Los folds por imagen no cubren exactamente la cohorte.")
        for column in ("patient_id", "target_binary", "sha256"):
            if not checked[column].eq(checked[f"{column}_cohort"]).all():
                raise ValueError(f"{column} difiere entre cohorte y folds por imagen.")
        assignments = table.copy()
    else:
        expected_rows = EXPECTED_PATIENTS * N_REPETITIONS
        if len(table) != expected_rows:
            raise ValueError(f"Se esperaban {expected_rows} asignaciones por paciente.")
        patient_truth = (
            cohort.groupby(["patient_id", "target_binary"], as_index=False)
            .size()
            .rename(columns={"size": "image_count"})
        )
        checked = table.merge(
            patient_truth,
            on="patient_id",
            how="outer",
            validate="many_to_one",
            suffixes=("", "_cohort"),
            indicator=True,
        )
        if not checked["_merge"].eq("both").all():
            raise ValueError("Los folds por paciente no cubren exactamente la cohorte.")
        for column in ("target_binary", "image_count"):
            if not checked[column].eq(checked[f"{column}_cohort"]).all():
                raise ValueError(f"{column} difiere entre cohorte y folds por paciente.")
        assignments = cohort.merge(
            table,
            on="patient_id",
            how="inner",
            validate="many_to_many",
            suffixes=("", "_fold"),
        )
        if not assignments["target_binary"].eq(
            assignments["target_binary_fold"]
        ).all():
            raise ValueError("La clase difiere al expandir folds por paciente.")
        assignments = assignments.drop(columns=["target_binary_fold", "image_count"])

    schedule = load_seed_schedule()
    seed_columns = schedule[
        [
            "repetition",
            "fold",
            "root_seed",
            "image_partition_seed",
            "patient_partition_seed",
            "training_seed",
        ]
    ].rename(columns={"fold": "cv_fold", "root_seed": "root_seed_schedule"})
    checked_seeds = assignments.merge(
        seed_columns,
        on=["repetition", "cv_fold", "training_seed"],
        how="left",
        validate="many_to_one",
    )
    if checked_seeds["root_seed_schedule"].isna().any():
        raise ValueError("Una asignación no coincide con el calendario de semillas.")
    expected_partition = (
        checked_seeds["image_partition_seed"]
        if protocol == "image_level"
        else checked_seeds["patient_partition_seed"]
    )
    if not checked_seeds["root_seed"].eq(checked_seeds["root_seed_schedule"]).all():
        raise ValueError("La semilla raíz difiere del calendario.")
    if not checked_seeds["partition_seed"].eq(expected_partition).all():
        raise ValueError("La semilla de partición difiere del calendario.")
    return assignments.sort_values(["repetition", "cv_fold", "relative_path"]).reset_index(
        drop=True
    )


def build_run_frames(
    protocol: str,
    repetition: int,
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if repetition not in range(1, N_REPETITIONS + 1):
        raise ValueError(f"Repetición fuera del intervalo 1-5: {repetition}")
    if fold not in range(1, N_FOLDS + 1):
        raise ValueError(f"Fold fuera del intervalo 1-5: {fold}")
    table = load_fold_assignments(protocol)
    repetition_rows = table.loc[table["repetition"].eq(repetition)].copy()
    if len(repetition_rows) != EXPECTED_IMAGES:
        raise ValueError("La repetición no contiene exactamente la cohorte analítica.")
    if repetition_rows["relative_path"].nunique() != EXPECTED_IMAGES:
        raise ValueError("La repetición contiene rutas repetidas u omitidas.")
    validation = repetition_rows.loc[repetition_rows["cv_fold"].eq(fold)].copy()
    training = repetition_rows.loc[~repetition_rows["cv_fold"].eq(fold)].copy()
    if set(training["relative_path"]).intersection(validation["relative_path"]):
        raise ValueError("Entrenamiento y validación comparten rutas.")
    if set(training["sha256"]).intersection(validation["sha256"]):
        raise ValueError("Entrenamiento y validación comparten hashes exactos.")
    patient_overlap = set(training["patient_id"]).intersection(validation["patient_id"])
    if protocol == "patient_level" and patient_overlap:
        raise ValueError("La partición por paciente comparte identidades.")
    if protocol == "image_level" and not patient_overlap:
        raise ValueError("La partición por imagen no presenta el solapamiento esperado.")
    return training.reset_index(drop=True), validation.reset_index(drop=True)


def validate_run_configuration(
    *, architecture: str, epochs: int, batch_size: int, pilot: bool
) -> None:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Arquitectura no reconocida: {architecture}")
    expected_epochs = 1 if pilot else DEFAULT_EPOCHS
    if epochs != expected_epochs:
        raise ValueError(
            f"epochs={epochs} difiere del valor preespecificado {expected_epochs}."
        )
    if pilot and architecture == "resnet50":
        if batch_size not in RESNET50_PILOT_BATCH_SIZES:
            raise ValueError("El piloto ResNet50 solo admite lotes 32, 16 u 8.")
    elif batch_size != FINAL_BATCH_SIZES[architecture]:
        raise ValueError(
            f"batch_size={batch_size} difiere del valor preespecificado "
            f"{FINAL_BATCH_SIZES[architecture]}."
        )


def build_model(architecture: str, pretrained: bool = True) -> nn.Module:
    if architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif architecture == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        raise ValueError(f"Arquitectura no reconocida: {architecture}")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Todas las capas deben permanecer ajustables.")
    return model


def make_loaders(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    training_dataset = BehDataset(training, BEHSOF_IMAGES_DIR, build_train_transform())
    validation_dataset = BehDataset(
        validation, BEHSOF_IMAGES_DIR, build_validation_transform()
    )
    common = {
        "batch_size": batch_size,
        "num_workers": DEFAULT_NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
    }
    training_loader = DataLoader(
        training_dataset, shuffle=True, generator=generator, **common
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return training_loader, validation_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * len(labels)
        total_items += len(labels)
    if total_items == 0:
        raise RuntimeError("El conjunto de entrenamiento está vacío.")
    return total_loss / total_items


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for images, labels, paths, patients in loader:
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels_device)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            total_loss += float(loss.item()) * len(labels)
            total_items += len(labels)
            for path, patient, target, probability in zip(
                paths, patients, labels.numpy(), probabilities, strict=True
            ):
                rows.append(
                    {
                        "relative_path": str(path),
                        "patient_id": str(patient),
                        "target_binary": int(target),
                        "probability": float(probability),
                    }
                )
    if total_items == 0:
        raise RuntimeError("El conjunto de validación está vacío.")
    return total_loss / total_items, pd.DataFrame(rows)


def _ensure_new_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Ya existen productos de esta ejecución. Use --overwrite solo "
            f"después de revisarlos:\n{formatted}"
        )


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def run_training(
    *,
    architecture: str,
    protocol: str,
    repetition: int,
    fold: int,
    epochs: int,
    batch_size: int,
    results_root: Path,
    checkpoint_root: Path,
    weights_cache: Path,
    pilot: bool,
    overwrite: bool,
) -> dict[str, object]:
    validate_run_configuration(
        architecture=architecture,
        epochs=epochs,
        batch_size=batch_size,
        pilot=pilot,
    )
    training, validation = build_run_frames(protocol, repetition, fold)
    identity = validation[["root_seed", "training_seed", "partition_seed"]].drop_duplicates()
    if len(identity) != 1:
        raise RuntimeError("El fold no posee una identidad de semillas única.")
    root_seed = int(identity.iloc[0]["root_seed"])
    training_seed = int(identity.iloc[0]["training_seed"])
    partition_seed = int(identity.iloc[0]["partition_seed"])
    set_reproducibility(training_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_cache = weights_cache.resolve()
    weights_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(weights_cache)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    category = "pilot" if pilot else "final"
    run_dir = (
        results_root.resolve()
        / category
        / architecture
        / protocol
        / f"repetition_{repetition}"
        / f"fold_{fold}"
    )
    checkpoint_dir = (
        checkpoint_root.resolve()
        / category
        / architecture
        / protocol
        / f"repetition_{repetition}"
    )
    history_path = run_dir / "history.csv"
    predictions_path = run_dir / "predictions.csv"
    metadata_path = run_dir / "run_metadata.json"
    checkpoint_path = checkpoint_dir / f"fold_{fold}_final.pt"
    _ensure_new_outputs(
        [history_path, predictions_path, metadata_path, checkpoint_path], overwrite
    )

    training_loader, validation_loader = make_loaders(
        training, validation, batch_size, training_seed
    )
    model = build_model(architecture, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )

    history: list[dict[str, object]] = []
    predictions = pd.DataFrame()
    run_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_epoch(model, training_loader, criterion, optimizer, device)
        validation_loss, predictions = validate(
            model, validation_loader, criterion, device
        )
        elapsed = time.perf_counter() - epoch_start
        history.append(
            {
                "architecture": architecture,
                "protocol": protocol,
                "repetition": repetition,
                "fold": fold,
                "root_seed": root_seed,
                "training_seed": training_seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"{architecture} | {protocol} | repetition {repetition} | "
            f"fold {fold} | epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={validation_loss:.6f} | "
            f"{elapsed:.1f} s",
            flush=True,
        )

    total_seconds = time.perf_counter() - run_start
    if len(predictions) != len(validation):
        raise RuntimeError("El número de predicciones no coincide con la validación.")
    if predictions["relative_path"].duplicated().any():
        raise RuntimeError("La validación produjo rutas duplicadas.")
    if set(predictions["relative_path"]) != set(validation["relative_path"]):
        raise RuntimeError("Las predicciones no corresponden exactamente al fold.")
    if not np.isfinite(predictions["probability"]).all():
        raise RuntimeError("La validación produjo probabilidades no finitas.")
    if not predictions["probability"].between(0.0, 1.0).all():
        raise RuntimeError("La validación produjo probabilidades fuera de [0, 1].")
    predictions = predictions.merge(
        validation[["relative_path", "sha256"]],
        on="relative_path",
        how="left",
        validate="one_to_one",
    )
    if predictions["sha256"].isna().any():
        raise RuntimeError("No se pudo vincular el hash de una predicción.")

    for position, (column, value) in enumerate(
        (
            ("architecture", architecture),
            ("protocol", protocol),
            ("repetition", repetition),
            ("fold", fold),
            ("root_seed", root_seed),
            ("training_seed", training_seed),
        )
    ):
        predictions.insert(position, column, value)

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False, lineterminator="\n")
    predictions.to_csv(predictions_path, index=False, lineterminator="\n")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": architecture,
            "protocol": protocol,
            "repetition": repetition,
            "fold": fold,
            "epoch": epochs,
            "root_seed": root_seed,
            "training_seed": training_seed,
        },
        checkpoint_path,
    )

    fold_source = IMAGE_FOLDS_CSV if protocol == "image_level" else PATIENT_FOLDS_CSV
    metadata: dict[str, object] = {
        "run_status": "complete",
        "architecture": architecture,
        "protocol": protocol,
        "repetition": repetition,
        "fold": fold,
        "pilot": pilot,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss_unweighted",
        "pretrained_weights": WEIGHT_NAMES[architecture],
        "all_layers_trainable": True,
        "precision": "float32",
        "mixed_precision": False,
        "gradient_accumulation_steps": 1,
        "scheduler": None,
        "early_stopping": False,
        "checkpoint_selection": "fixed_final_epoch",
        "root_seed": root_seed,
        "partition_seed": partition_seed,
        "training_seed": training_seed,
        "num_workers": DEFAULT_NUM_WORKERS,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "analytic_cohort": ANALYTIC_COHORT_CSV.relative_to(PROJECT_ROOT).as_posix(),
        "analytic_cohort_sha256": sha256(ANALYTIC_COHORT_CSV),
        "fold_source": fold_source.relative_to(PROJECT_ROOT).as_posix(),
        "fold_source_sha256": sha256(fold_source),
        "seed_schedule": SEED_SCHEDULE_CSV.relative_to(PROJECT_ROOT).as_posix(),
        "seed_schedule_sha256": sha256(SEED_SCHEDULE_CSV),
        "partition_parameters": PARTITION_PARAMETERS_JSON.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "partition_parameters_sha256": sha256(PARTITION_PARAMETERS_JSON),
        "weights_cache": _relative_or_absolute(weights_cache, DATA_ROOT),
        "checkpoint": _relative_or_absolute(checkpoint_path, DATA_ROOT),
        "checkpoint_sha256": sha256(checkpoint_path),
        "history_sha256": sha256(history_path),
        "predictions_sha256": sha256(predictions_path),
        "peak_gpu_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        ),
        "training_images": len(training),
        "validation_images": len(validation),
        "training_patients": int(training["patient_id"].nunique()),
        "validation_patients": int(validation["patient_id"].nunique()),
        "overlapping_patients": len(
            set(training["patient_id"]).intersection(validation["patient_id"])
        ),
        "total_seconds": total_seconds,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return metadata


def _selected(value: str, options: tuple[str, ...]) -> tuple[str, ...]:
    return options if value in {"all", "both"} else (value,)


def main() -> None:
    args = parse_args()
    architectures = _selected(args.architecture, ARCHITECTURES)
    protocols = _selected(args.protocol, PROTOCOLS)
    repetitions = (
        tuple(range(1, N_REPETITIONS + 1))
        if args.repetition == "all"
        else (int(args.repetition),)
    )
    folds = (
        tuple(range(1, N_FOLDS + 1))
        if args.fold == "all"
        else (int(args.fold),)
    )
    if args.pilot and any(
        len(values) != 1 for values in (architectures, protocols, repetitions, folds)
    ):
        raise ValueError(
            "Un piloto debe identificar exactamente una arquitectura, protocolo, "
            "repetición y fold."
        )
    for architecture in architectures:
        batch_size = (
            FINAL_BATCH_SIZES[architecture]
            if args.batch_size is None
            else args.batch_size
        )
        for protocol in protocols:
            for repetition in repetitions:
                for fold in folds:
                    run_training(
                        architecture=architecture,
                        protocol=protocol,
                        repetition=repetition,
                        fold=fold,
                        epochs=args.epochs,
                        batch_size=batch_size,
                        results_root=args.results_root,
                        checkpoint_root=args.checkpoint_root,
                        weights_cache=args.weights_cache,
                        pilot=args.pilot,
                        overwrite=args.overwrite,
                    )


if __name__ == "__main__":
    main()
