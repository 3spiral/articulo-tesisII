"""Valida, agrega y compara las predicciones fuera de muestra de BEH."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from src.data.build_beh_image_folds import load_eligible_images, sha256
from src.settings import (
    BEH_PATIENT_FOLDS_CSV,
    BEHSOF_MANIFEST_CSV,
    PROJECT_ROOT,
)
from src.training.train_beh_crossval import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    IMAGE_FOLDS_CSV,
    build_run_frames,
)


DEFAULT_PREDICTIONS_ROOT = PROJECT_ROOT / "results" / "training" / "final"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"
PROTOCOLS = ("image", "patient")
N_FOLDS = 5
THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42


def validate_run_metadata(
    metadata: dict[str, object], protocol: str, fold: int
) -> None:
    """Rechaza ejecuciones incompletas, piloto o ajenas al protocolo fijado."""

    fold_source = IMAGE_FOLDS_CSV if protocol == "image" else BEH_PATIENT_FOLDS_CSV
    expected: dict[str, object] = {
        "run_status": "complete",
        "protocol": protocol,
        "fold": fold,
        "pilot": False,
        "epochs": DEFAULT_EPOCHS,
        "batch_size": DEFAULT_BATCH_SIZE,
        "learning_rate": DEFAULT_LEARNING_RATE,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss_unweighted",
        "architecture": "EfficientNet-B0",
        "pretrained_weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "checkpoint_selection": "fixed_final_epoch",
        "seed": DEFAULT_SEED,
        "manifest_sha256": sha256(BEHSOF_MANIFEST_CSV),
        "fold_source": fold_source.relative_to(PROJECT_ROOT).as_posix(),
        "fold_source_sha256": sha256(fold_source),
    }
    missing = sorted(set(expected).difference(metadata))
    if missing:
        raise ValueError(
            f"Metadatos incompletos para {protocol} fold {fold}: {', '.join(missing)}"
        )
    mismatches = [
        f"{key}={metadata[key]!r} (esperado {value!r})"
        for key, value in expected.items()
        if metadata[key] != value
    ]
    if mismatches:
        raise ValueError(
            f"La ejecución {protocol} fold {fold} no coincide con el protocolo: "
            + "; ".join(mismatches)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa y compara las predicciones finales de ambos protocolos."
    )
    parser.add_argument(
        "--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_evaluation_configuration(iterations: int, seed: int) -> None:
    """Protege la configuración inferencial preespecificada en D-018."""

    if iterations != BOOTSTRAP_ITERATIONS or seed != BOOTSTRAP_SEED:
        raise ValueError(
            "La evaluación final exige 10,000 remuestreos bootstrap y semilla 42."
        )


def load_protocol_predictions(protocol: str, predictions_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold in range(1, N_FOLDS + 1):
        run_dir = predictions_root / protocol / f"fold_{fold}"
        path = run_dir / "predictions.csv"
        metadata_path = run_dir / "run_metadata.json"
        if not path.exists():
            raise FileNotFoundError(f"Falta el archivo de predicciones: {path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Falta el archivo de metadatos: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"Los metadatos no son un objeto JSON: {metadata_path}")
        validate_run_metadata(metadata, protocol, fold)

        frame = pd.read_csv(path, dtype={"patient_id": str})
        required = {
            "protocol",
            "fold",
            "seed",
            "relative_path",
            "patient_id",
            "target_binary",
            "probability",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Faltan columnas en {path}: {', '.join(missing)}")
        frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
        frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
        if not frame["protocol"].eq(protocol).all() or not frame["fold"].eq(fold).all():
            raise ValueError(f"Identidad de protocolo o fold incoherente en {path}.")
        if not frame["seed"].eq(DEFAULT_SEED).all():
            raise ValueError(f"Semilla incoherente en {path}.")

        _, expected_validation = build_run_frames(protocol, fold)
        if frame["relative_path"].duplicated().any():
            raise ValueError(f"Hay rutas duplicadas en {path}.")
        if set(frame["relative_path"]) != set(expected_validation["relative_path"]):
            raise ValueError(f"Las rutas de {path} no corresponden al fold esperado.")
        frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True)
    predictions["target_binary"] = pd.to_numeric(
        predictions["target_binary"], errors="raise"
    ).astype(int)
    predictions["probability"] = pd.to_numeric(
        predictions["probability"], errors="raise"
    ).astype(float)
    if len(predictions) != 1_388 or predictions["relative_path"].nunique() != 1_388:
        raise ValueError(f"{protocol}: no hay exactamente 1,388 predicciones únicas.")
    if not np.isfinite(predictions["probability"]).all():
        raise ValueError(f"{protocol}: existen probabilidades no finitas.")
    if not predictions["probability"].between(0.0, 1.0).all():
        raise ValueError(f"{protocol}: existen probabilidades fuera de [0, 1].")

    eligible = load_eligible_images(BEHSOF_MANIFEST_CSV)[
        ["relative_path", "patient_id", "target_binary"]
    ]
    checked = predictions.merge(
        eligible,
        on="relative_path",
        how="outer",
        validate="one_to_one",
        suffixes=("", "_manifest"),
        indicator=True,
    )
    if not checked["_merge"].eq("both").all():
        raise ValueError(f"{protocol}: las rutas no coinciden con la muestra elegible.")
    if not checked["patient_id"].eq(checked["patient_id_manifest"]).all():
        raise ValueError(f"{protocol}: patient_id no coincide con el manifiesto.")
    if not checked["target_binary"].eq(checked["target_binary_manifest"]).all():
        raise ValueError(f"{protocol}: target_binary no coincide con el manifiesto.")
    return predictions.sort_values("relative_path", kind="stable").reset_index(drop=True)


def aggregate_by_patient(
    predictions: pd.DataFrame, protocol: str, expected_patients: int = 95
) -> pd.DataFrame:
    if predictions.groupby("patient_id")["target_binary"].nunique().max() != 1:
        raise ValueError("Un paciente presenta más de una clase objetivo.")
    aggregated = (
        predictions.groupby("patient_id", as_index=False, sort=True)
        .agg(
            target_binary=("target_binary", "first"),
            probability=("probability", "mean"),
            image_count=("relative_path", "size"),
        )
        .assign(protocol=protocol)
    )
    if len(aggregated) != expected_patients:
        raise ValueError(
            f"{protocol}: se esperaban {expected_patients} pacientes; "
            f"se obtuvieron {len(aggregated)}."
        )
    return aggregated[["protocol", "patient_id", "target_binary", "image_count", "probability"]]


def calculate_metrics(frame: pd.DataFrame, threshold: float = THRESHOLD) -> dict[str, float | int]:
    targets = frame["target_binary"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "auroc": float(roc_auc_score(targets, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "threshold": threshold,
    }


def paired_stratified_bootstrap_delta(
    paired: pd.DataFrame,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    if iterations < 1:
        raise ValueError("El número de remuestreos debe ser positivo.")
    rng = np.random.default_rng(seed)
    targets = paired["target_binary"].to_numpy(dtype=int)
    image_probabilities = paired["probability_image"].to_numpy(dtype=float)
    patient_probabilities = paired["probability_patient"].to_numpy(dtype=float)
    class_indices = [np.flatnonzero(targets == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("Ambas clases deben estar presentes para calcular AUROC.")

    observed = float(
        roc_auc_score(targets, image_probabilities)
        - roc_auc_score(targets, patient_probabilities)
    )
    deltas = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        sampled_targets = targets[sampled]
        deltas[iteration] = roc_auc_score(
            sampled_targets, image_probabilities[sampled]
        ) - roc_auc_score(sampled_targets, patient_probabilities[sampled])
    lower, upper = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_auroc_image_minus_patient": observed,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "bootstrap_method": "paired_stratified_patient_level_percentile",
    }


def build_paired_patient_table(
    image_patients: pd.DataFrame, patient_patients: pd.DataFrame
) -> pd.DataFrame:
    paired = image_patients.merge(
        patient_patients,
        on="patient_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_image", "_patient"),
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Los protocolos no contienen exactamente los mismos pacientes.")
    if not paired["target_binary_image"].eq(paired["target_binary_patient"]).all():
        raise ValueError("Las clases objetivo difieren entre protocolos.")
    if not paired["image_count_image"].eq(paired["image_count_patient"]).all():
        raise ValueError("El número de imágenes por paciente difiere entre protocolos.")
    return paired.rename(columns={"target_binary_image": "target_binary"})[
        [
            "patient_id",
            "target_binary",
            "image_count_image",
            "probability_image",
            "probability_patient",
        ]
    ].rename(columns={"image_count_image": "image_count"})


def ensure_outputs_are_writable(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError("Ya existen resultados de evaluación:\n" + formatted)


def main() -> None:
    args = parse_args()
    validate_evaluation_configuration(
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    predictions_root = args.predictions_root.resolve()
    output_dir = args.output_dir.resolve()
    image_outputs_path = output_dir / "image_predictions.csv"
    patient_outputs_path = output_dir / "patient_predictions.csv"
    paired_path = output_dir / "paired_patient_predictions.csv"
    metrics_path = output_dir / "metrics.csv"
    comparison_path = output_dir / "auroc_comparison.json"
    output_paths = [
        image_outputs_path,
        patient_outputs_path,
        paired_path,
        metrics_path,
        comparison_path,
    ]
    ensure_outputs_are_writable(output_paths, args.overwrite)

    image_predictions = load_protocol_predictions("image", predictions_root)
    patient_predictions = load_protocol_predictions("patient", predictions_root)
    image_patients = aggregate_by_patient(image_predictions, "image")
    patient_patients = aggregate_by_patient(patient_predictions, "patient")
    paired = build_paired_patient_table(image_patients, patient_patients)

    metric_rows = []
    for protocol, frame in (("image", image_patients), ("patient", patient_patients)):
        metric_rows.append({"protocol": protocol, **calculate_metrics(frame)})
    comparison = paired_stratified_bootstrap_delta(
        paired,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_predictions.to_csv(image_outputs_path, index=False, lineterminator="\n")
    patient_predictions.to_csv(patient_outputs_path, index=False, lineterminator="\n")
    paired.to_csv(paired_path, index=False, lineterminator="\n")
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False, lineterminator="\n")
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
