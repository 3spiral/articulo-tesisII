"""Entrena EfficientNet-B0 con particiones por imagen o por paciente.

Cada ejecución usa como validación un único fold y produce probabilidades
fuera de muestra para todas sus imágenes. El script nunca modifica las
ecografías originales ni selecciona checkpoints a partir del rendimiento de
validación: conserva únicamente el estado de la época final fijada.
"""

from __future__ import annotations

import argparse
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
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.data.build_beh_image_folds import (
    PATIENT_STRATEGY,
    STRATEGY as IMAGE_STRATEGY,
    load_eligible_images,
    sha256,
)
from src.data.preprocess_behsof import build_train_transform, build_validation_transform
from src.settings import (
    BEH_PATIENT_FOLDS_CSV,
    BEHSOF_IMAGES_DIR,
    BEHSOF_MANIFEST_CSV,
    DATA_ROOT,
    PROJECT_ROOT,
)


IMAGE_FOLDS_CSV = PROJECT_ROOT / "metadata" / "BEH_image_folds.csv"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "training"
DEFAULT_CHECKPOINT_ROOT = DATA_ROOT / "models" / "article"
DEFAULT_WEIGHTS_CACHE = DATA_ROOT / "models" / "torch_cache"

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_SEED = 42
N_FOLDS = 5


class BehDataset(Dataset):
    """Dataset que resuelve rutas BEH y transforma las imágenes en memoria."""

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
        description="Entrena EfficientNet-B0 en los folds BEH fijados."
    )
    parser.add_argument(
        "--protocol",
        choices=("image", "patient", "both"),
        default="both",
    )
    parser.add_argument(
        "--fold",
        choices=("1", "2", "3", "4", "5", "all"),
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--weights-cache", type=Path, default=DEFAULT_WEIGHTS_CACHE)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Ejecuta una época técnica y separa sus productos de los resultados finales.",
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


def _load_fold_table(protocol: str) -> pd.DataFrame:
    if protocol == "image":
        table = pd.read_csv(IMAGE_FOLDS_CSV, dtype={"patient_id": str})
        required = [
            "relative_path",
            "patient_id",
            "target_binary",
            "cv_fold",
            "cv_seed",
            "cv_strategy",
        ]
        missing = sorted(set(required).difference(table.columns))
        if missing:
            raise ValueError("Faltan columnas en BEH_image_folds.csv: " + ", ".join(missing))
        table = table[required].copy()
        table["target_binary"] = pd.to_numeric(
            table["target_binary"], errors="raise"
        ).astype(int)
        table["cv_fold"] = pd.to_numeric(table["cv_fold"], errors="raise").astype(int)
        table["cv_seed"] = pd.to_numeric(table["cv_seed"], errors="raise").astype(int)
        if not table["cv_seed"].eq(DEFAULT_SEED).all():
            raise ValueError("BEH_image_folds.csv no utiliza exclusivamente la semilla 42.")
        if not table["cv_strategy"].eq(IMAGE_STRATEGY).all():
            raise ValueError("BEH_image_folds.csv declara una estrategia inesperada.")

        eligible = load_eligible_images(BEHSOF_MANIFEST_CSV)[
            ["relative_path", "patient_id", "target_binary"]
        ]
        checked = table.merge(
            eligible,
            on="relative_path",
            how="outer",
            validate="one_to_one",
            suffixes=("", "_manifest"),
            indicator=True,
        )
        if not checked["_merge"].eq("both").all():
            raise ValueError("Las rutas de BEH_image_folds.csv difieren del manifiesto elegible.")
        if not checked["patient_id"].eq(checked["patient_id_manifest"]).all():
            raise ValueError("Los pacientes de BEH_image_folds.csv difieren del manifiesto.")
        if not checked["target_binary"].eq(checked["target_binary_manifest"]).all():
            raise ValueError("Las etiquetas de BEH_image_folds.csv difieren del manifiesto.")
        return table[["relative_path", "patient_id", "target_binary", "cv_fold"]]

    patient_folds = pd.read_csv(BEH_PATIENT_FOLDS_CSV, dtype={"patient_id": str})
    required = ["patient_id", "target_binary", "cv_fold", "cv_seed", "cv_strategy"]
    missing = sorted(set(required).difference(patient_folds.columns))
    if missing:
        raise ValueError("Faltan columnas en BEH_patient_folds.csv: " + ", ".join(missing))
    patient_folds = patient_folds[required].copy()
    patient_folds["target_binary"] = pd.to_numeric(
        patient_folds["target_binary"], errors="raise"
    ).astype(int)
    patient_folds["cv_fold"] = pd.to_numeric(
        patient_folds["cv_fold"], errors="raise"
    ).astype(int)
    patient_folds["cv_seed"] = pd.to_numeric(
        patient_folds["cv_seed"], errors="raise"
    ).astype(int)
    if len(patient_folds) != 95 or patient_folds["patient_id"].nunique() != 95:
        raise ValueError("BEH_patient_folds.csv no representa exactamente 95 pacientes.")
    if not patient_folds["cv_seed"].eq(DEFAULT_SEED).all():
        raise ValueError("BEH_patient_folds.csv no utiliza exclusivamente la semilla 42.")
    if not patient_folds["cv_strategy"].eq(PATIENT_STRATEGY).all():
        raise ValueError("BEH_patient_folds.csv declara una estrategia inesperada.")

    eligible = load_eligible_images(BEHSOF_MANIFEST_CSV)
    eligible = eligible.drop(columns=["cv_fold"])
    table = eligible[["relative_path", "patient_id", "target_binary"]].merge(
        patient_folds[["patient_id", "target_binary", "cv_fold"]],
        on="patient_id",
        how="left",
        validate="many_to_one",
        suffixes=("", "_fold"),
    )
    if table["cv_fold"].isna().any():
        raise ValueError("Hay pacientes elegibles sin fold por paciente.")
    if not table["target_binary"].eq(table["target_binary_fold"]).all():
        raise ValueError("La clase del manifiesto difiere de BEH_patient_folds.csv.")
    return table[["relative_path", "patient_id", "target_binary", "cv_fold"]].copy()


def build_run_frames(protocol: str, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve entrenamiento y validación, verificando la separación prevista."""

    if protocol not in {"image", "patient"}:
        raise ValueError(f"Protocolo desconocido: {protocol}")
    if fold not in range(1, N_FOLDS + 1):
        raise ValueError(f"Fold fuera del intervalo 1-5: {fold}")
    table = _load_fold_table(protocol)
    table["cv_fold"] = pd.to_numeric(table["cv_fold"], errors="raise").astype(int)
    table["target_binary"] = pd.to_numeric(
        table["target_binary"], errors="raise"
    ).astype(int)
    table = table.sort_values("relative_path", kind="stable").reset_index(drop=True)

    if len(table) != 1_388 or table["relative_path"].nunique() != 1_388:
        raise ValueError("La tabla de folds no representa exactamente 1,388 imágenes únicas.")
    if set(table["cv_fold"].unique()) != set(range(1, N_FOLDS + 1)):
        raise ValueError("La tabla no contiene exactamente los folds 1-5.")

    validation = table.loc[table["cv_fold"].eq(fold)].copy()
    training = table.loc[~table["cv_fold"].eq(fold)].copy()
    if set(training["relative_path"]).intersection(validation["relative_path"]):
        raise ValueError("Una imagen aparece simultáneamente en entrenamiento y validación.")

    patient_overlap = set(training["patient_id"]).intersection(validation["patient_id"])
    if protocol == "patient" and patient_overlap:
        raise ValueError("La partición por paciente presenta solapamiento de pacientes.")
    if protocol == "image" and not patient_overlap:
        raise ValueError("La partición por imagen no presenta el solapamiento esperado.")
    return training, validation


def validate_image_sources(image_root: Path = BEHSOF_IMAGES_DIR) -> None:
    """Comprueba que toda la muestra elegible exista sin abrir ni modificar imágenes."""

    resolved_root = image_root.resolve()
    eligible = load_eligible_images(BEHSOF_MANIFEST_CSV)
    missing: list[str] = []
    for relative_path in eligible["relative_path"]:
        path = (resolved_root / str(relative_path)).resolve()
        if not path.is_relative_to(resolved_root):
            raise ValueError(f"Ruta fuera de la raíz BEH: {relative_path}")
        if not path.is_file():
            missing.append(str(relative_path))
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Faltan {len(missing)} imágenes elegibles. Primeras rutas: {preview}"
        )


def validate_run_configuration(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    pilot: bool,
) -> None:
    """Impide que una ejecución formal se aparte de la configuración registrada."""

    expected_epochs = 1 if pilot else DEFAULT_EPOCHS
    observed = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
    }
    expected = {
        "epochs": expected_epochs,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "seed": DEFAULT_SEED,
    }
    mismatches = [
        f"{key}={observed[key]!r} (esperado {expected[key]!r})"
        for key in expected
        if observed[key] != expected[key]
    ]
    if mismatches:
        category = "piloto" if pilot else "experimento final"
        raise ValueError(
            f"La configuración del {category} difiere del protocolo fijado: "
            + "; ".join(mismatches)
        )


def build_model(pretrained: bool = True) -> nn.Module:
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def make_loaders(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    training_dataset = BehDataset(training, BEHSOF_IMAGES_DIR, build_train_transform())
    validation_dataset = BehDataset(
        validation, BEHSOF_IMAGES_DIR, build_validation_transform()
    )
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    training_loader = DataLoader(
        training_dataset,
        shuffle=True,
        generator=generator,
        **common,
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
    loss_sum = 0.0
    sample_count = 0
    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * labels.size(0)
        sample_count += labels.size(0)
    return loss_sum / sample_count


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, pd.DataFrame]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    rows: list[dict[str, str | int | float]] = []
    for images, labels, relative_paths, patient_ids in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, labels_device)
        probabilities = torch.sigmoid(logits).cpu().tolist()
        loss_sum += float(loss) * labels.size(0)
        sample_count += labels.size(0)
        for relative_path, patient_id, target, probability in zip(
            relative_paths,
            patient_ids,
            labels.tolist(),
            probabilities,
            strict=True,
        ):
            rows.append(
                {
                    "relative_path": relative_path,
                    "patient_id": patient_id,
                    "target_binary": int(target),
                    "probability": float(probability),
                }
            )
    return loss_sum / sample_count, pd.DataFrame(rows)


def _ensure_new_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Ya existen productos para esta ejecución. Revise o use --overwrite:\n"
            + formatted
        )


def run_training(
    protocol: str,
    fold: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    num_workers: int,
    results_root: Path,
    checkpoint_root: Path,
    weights_cache: Path,
    pilot: bool,
    overwrite: bool,
) -> dict[str, object]:
    validate_run_configuration(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        pilot=pilot,
    )
    set_reproducibility(seed)
    training, validation = build_run_frames(protocol, fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_cache = weights_cache.resolve()
    weights_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(weights_cache)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    category = "pilot" if pilot else "final"
    run_dir = results_root.resolve() / category / protocol / f"fold_{fold}"
    checkpoint_dir = checkpoint_root.resolve() / category / protocol
    history_path = run_dir / "history.csv"
    predictions_path = run_dir / "predictions.csv"
    metadata_path = run_dir / "run_metadata.json"
    checkpoint_path = checkpoint_dir / f"fold_{fold}_final.pt"
    _ensure_new_outputs(
        [history_path, predictions_path, metadata_path, checkpoint_path], overwrite
    )

    training_loader, validation_loader = make_loaders(
        training, validation, batch_size, num_workers, seed
    )
    model = build_model(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history: list[dict[str, int | float | str]] = []
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
                "protocol": protocol,
                "fold": fold,
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "elapsed_seconds": elapsed,
            }
        )
        print(
            f"{protocol} fold {fold} | epoch {epoch}/{epochs} | "
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
        raise RuntimeError("Las predicciones no corresponden exactamente al fold validado.")
    if not np.isfinite(predictions["probability"]).all():
        raise RuntimeError("La validación produjo probabilidades no finitas.")
    if not predictions["probability"].between(0.0, 1.0).all():
        raise RuntimeError("La validación produjo probabilidades fuera de [0, 1].")
    predictions.insert(0, "seed", seed)
    predictions.insert(0, "fold", fold)
    predictions.insert(0, "protocol", protocol)

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False, lineterminator="\n")
    predictions.to_csv(predictions_path, index=False, lineterminator="\n")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "protocol": protocol,
            "fold": fold,
            "epoch": epochs,
            "seed": seed,
        },
        checkpoint_path,
    )

    fold_source = IMAGE_FOLDS_CSV if protocol == "image" else BEH_PATIENT_FOLDS_CSV
    try:
        weights_cache_record = weights_cache.relative_to(DATA_ROOT).as_posix()
    except ValueError:
        weights_cache_record = str(weights_cache)
    metadata: dict[str, object] = {
        "run_status": "complete",
        "protocol": protocol,
        "fold": fold,
        "pilot": pilot,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss_unweighted",
        "architecture": "EfficientNet-B0",
        "pretrained_weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "checkpoint_selection": "fixed_final_epoch",
        "seed": seed,
        "num_workers": num_workers,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "manifest_sha256": sha256(BEHSOF_MANIFEST_CSV),
        "fold_source": fold_source.relative_to(PROJECT_ROOT).as_posix(),
        "fold_source_sha256": sha256(fold_source),
        "weights_cache": weights_cache_record,
        "peak_gpu_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "peak_gpu_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        ),
        "training_images": len(training),
        "validation_images": len(validation),
        "training_patients": int(training["patient_id"].nunique()),
        "validation_patients": int(validation["patient_id"].nunique()),
        "overlapping_patients": len(
            set(training["patient_id"]).intersection(validation["patient_id"])
        ),
        "elapsed_seconds": total_seconds,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return metadata


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs y batch-size deben ser positivos; num-workers no puede ser negativo.")
    validate_run_configuration(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        pilot=args.pilot,
    )
    validate_image_sources()
    protocols = ("image", "patient") if args.protocol == "both" else (args.protocol,)
    folds = range(1, N_FOLDS + 1) if args.fold == "all" else (int(args.fold),)
    for protocol in protocols:
        for fold in folds:
            run_training(
                protocol=protocol,
                fold=fold,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
                num_workers=args.num_workers,
                results_root=args.results_root,
                checkpoint_root=args.checkpoint_root,
                weights_cache=args.weights_cache,
                pilot=args.pilot,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
