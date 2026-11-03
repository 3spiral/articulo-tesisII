"""Evalúa las predicciones OOF del experimento corregido del artículo.

La evaluación principal se realiza por paciente. El análisis por imagen y los
AUROC por fold son complementarios. El bootstrap remuestrea pacientes de forma
pareada y estratificada, sin reentrenar modelos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from src.settings import PROJECT_ROOT
from src.training.train_beh_crossval import (
    ANALYTIC_COHORT_CSV,
    ARCHITECTURES,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_WEIGHT_DECAY,
    FINAL_BATCH_SIZES,
    IMAGE_FOLDS_CSV,
    N_FOLDS,
    N_REPETITIONS,
    PARTITION_PARAMETERS_JSON,
    PATIENT_FOLDS_CSV,
    PROTOCOLS,
    SEED_SCHEDULE_CSV,
    WEIGHT_NAMES,
    build_run_frames,
    load_analytic_cohort,
    sha256,
)


DEFAULT_PREDICTIONS_ROOT = PROJECT_ROOT / "results" / "training" / "final"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"
THRESHOLD = 0.5
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20_261_030


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalúa las 100 ejecuciones preespecificadas del artículo."
    )
    parser.add_argument(
        "--predictions-root", type=Path, default=DEFAULT_PREDICTIONS_ROOT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS
    )
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_evaluation_configuration(iterations: int, seed: int) -> None:
    if iterations != BOOTSTRAP_ITERATIONS or seed != BOOTSTRAP_SEED:
        raise ValueError(
            "La evaluación final exige 10,000 remuestreos y raíz 20261030."
        )


def _expected_run_dir(
    predictions_root: Path,
    architecture: str,
    protocol: str,
    repetition: int,
    fold: int,
) -> Path:
    return (
        predictions_root
        / architecture
        / protocol
        / f"repetition_{repetition}"
        / f"fold_{fold}"
    )


def validate_run_metadata(
    metadata: dict[str, object],
    *,
    architecture: str,
    protocol: str,
    repetition: int,
    fold: int,
    predictions_path: Path,
    root_seed: int,
    partition_seed: int,
    training_seed: int,
) -> None:
    fold_source = IMAGE_FOLDS_CSV if protocol == "image_level" else PATIENT_FOLDS_CSV
    expected = {
        "run_status": "complete",
        "architecture": architecture,
        "protocol": protocol,
        "repetition": repetition,
        "fold": fold,
        "root_seed": root_seed,
        "partition_seed": partition_seed,
        "training_seed": training_seed,
        "pilot": False,
        "epochs": DEFAULT_EPOCHS,
        "batch_size": FINAL_BATCH_SIZES[architecture],
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
        "num_workers": DEFAULT_NUM_WORKERS,
        "analytic_cohort_sha256": sha256(ANALYTIC_COHORT_CSV),
        "fold_source_sha256": sha256(fold_source),
        "seed_schedule_sha256": sha256(SEED_SCHEDULE_CSV),
        "partition_parameters_sha256": sha256(PARTITION_PARAMETERS_JSON),
        "predictions_sha256": sha256(predictions_path),
    }
    missing = sorted(set(expected).difference(metadata))
    if missing:
        raise ValueError("Metadatos incompletos: " + ", ".join(missing))
    mismatches = [
        f"{key}={metadata[key]!r} (esperado {value!r})"
        for key, value in expected.items()
        if metadata[key] != value
    ]
    if mismatches:
        raise ValueError("Metadatos incompatibles: " + "; ".join(mismatches))


def load_all_predictions(predictions_root: Path) -> pd.DataFrame:
    predictions_root = predictions_root.resolve()
    frames: list[pd.DataFrame] = []
    required = {
        "architecture",
        "protocol",
        "repetition",
        "fold",
        "root_seed",
        "training_seed",
        "relative_path",
        "patient_id",
        "target_binary",
        "sha256",
        "probability",
    }
    for architecture in ARCHITECTURES:
        for protocol in PROTOCOLS:
            for repetition in range(1, N_REPETITIONS + 1):
                for fold in range(1, N_FOLDS + 1):
                    run_dir = _expected_run_dir(
                        predictions_root, architecture, protocol, repetition, fold
                    )
                    predictions_path = run_dir / "predictions.csv"
                    metadata_path = run_dir / "run_metadata.json"
                    if not predictions_path.is_file() or not metadata_path.is_file():
                        raise FileNotFoundError(f"Ejecución incompleta: {run_dir}")
                    _, expected_validation = build_run_frames(
                        protocol, repetition, fold
                    )
                    seed_identity = expected_validation[
                        ["root_seed", "partition_seed", "training_seed"]
                    ].drop_duplicates()
                    if len(seed_identity) != 1:
                        raise ValueError("El fold esperado no posee semillas únicas.")
                    expected_root_seed = int(seed_identity.iloc[0]["root_seed"])
                    expected_partition_seed = int(
                        seed_identity.iloc[0]["partition_seed"]
                    )
                    expected_training_seed = int(
                        seed_identity.iloc[0]["training_seed"]
                    )
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    validate_run_metadata(
                        metadata,
                        architecture=architecture,
                        protocol=protocol,
                        repetition=repetition,
                        fold=fold,
                        predictions_path=predictions_path,
                        root_seed=expected_root_seed,
                        partition_seed=expected_partition_seed,
                        training_seed=expected_training_seed,
                    )
                    frame = pd.read_csv(predictions_path, dtype={"patient_id": str})
                    missing = sorted(required.difference(frame.columns))
                    if missing:
                        raise ValueError(
                            f"Faltan columnas en {predictions_path}: " + ", ".join(missing)
                        )
                    identities = {
                        "architecture": architecture,
                        "protocol": protocol,
                        "repetition": repetition,
                        "fold": fold,
                    }
                    for column, value in identities.items():
                        if not frame[column].eq(value).all():
                            raise ValueError(f"Identidad incoherente en {predictions_path}.")
                    frame["target_binary"] = pd.to_numeric(
                        frame["target_binary"], errors="raise"
                    ).astype(int)
                    frame["probability"] = pd.to_numeric(
                        frame["probability"], errors="raise"
                    ).astype(float)
                    if not np.isfinite(frame["probability"]).all() or not frame[
                        "probability"
                    ].between(0.0, 1.0).all():
                        raise ValueError(f"Probabilidades inválidas en {predictions_path}.")
                    checked = frame.merge(
                        expected_validation[
                            [
                                "relative_path",
                                "patient_id",
                                "target_binary",
                                "sha256",
                                "root_seed",
                                "training_seed",
                            ]
                        ],
                        on="relative_path",
                        how="outer",
                        validate="one_to_one",
                        suffixes=("", "_expected"),
                        indicator=True,
                    )
                    if not checked["_merge"].eq("both").all():
                        raise ValueError(
                            f"Las rutas de {predictions_path} no corresponden al fold."
                        )
                    for column in (
                        "patient_id",
                        "target_binary",
                        "sha256",
                        "root_seed",
                        "training_seed",
                    ):
                        if not checked[column].eq(checked[f"{column}_expected"]).all():
                            raise ValueError(
                                f"{column} difiere del fold en {predictions_path}."
                            )
                    frames.append(frame[list(required)].copy())
    combined = pd.concat(frames, ignore_index=True)
    expected_rows = len(ARCHITECTURES) * len(PROTOCOLS) * N_REPETITIONS * 1_371
    if len(combined) != expected_rows:
        raise ValueError(f"Se esperaban {expected_rows} predicciones OOF.")
    return combined


def aggregate_by_patient(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.groupby("patient_id")["target_binary"].nunique().max() != 1:
        raise ValueError("Un paciente tiene más de una etiqueta en las predicciones.")
    return (
        frame.groupby(["patient_id", "target_binary"], as_index=False)["probability"]
        .mean()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )


def calculate_metrics(
    frame: pd.DataFrame, threshold: float = THRESHOLD
) -> dict[str, float | int]:
    targets = frame["target_binary"].to_numpy(dtype=int)
    probabilities = frame["probability"].to_numpy(dtype=float)
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("Se necesitan ambas clases para calcular las métricas.")
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
    }


def build_paired_patient_table(
    image_level: pd.DataFrame, patient_level: pd.DataFrame
) -> pd.DataFrame:
    left = aggregate_by_patient(image_level).rename(
        columns={"probability": "probability_image_level"}
    )
    right = aggregate_by_patient(patient_level).rename(
        columns={"probability": "probability_patient_level"}
    )
    paired = left.merge(
        right,
        on="patient_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_image", "_patient"),
        indicator=True,
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Los protocolos no contienen los mismos pacientes.")
    if not paired["target_binary_image"].eq(paired["target_binary_patient"]).all():
        raise ValueError("Las etiquetas difieren entre protocolos.")
    return paired.rename(columns={"target_binary_image": "target_binary"})[
        [
            "patient_id",
            "target_binary",
            "probability_image_level",
            "probability_patient_level",
        ]
    ]


def compute_metric_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    patient_predictions: list[pd.DataFrame] = []
    for (architecture, protocol, repetition), group in predictions.groupby(
        ["architecture", "protocol", "repetition"], sort=True
    ):
        if len(group) != 1_371 or group["relative_path"].nunique() != 1_371:
            raise ValueError("Una repetición no posee cobertura OOF completa.")
        patient = aggregate_by_patient(group)
        if len(patient) != 93:
            raise ValueError("La agregación no contiene 93 pacientes.")
        patient_metrics = calculate_metrics(patient)
        patient_rows.append(
            {
                "architecture": architecture,
                "protocol": protocol,
                "repetition": int(repetition),
                "analysis_level": "patient",
                **patient_metrics,
            }
        )
        image_rows.append(
            {
                "architecture": architecture,
                "protocol": protocol,
                "repetition": int(repetition),
                "analysis_level": "image_sensitivity",
                "auroc": float(
                    roc_auc_score(group["target_binary"], group["probability"])
                ),
            }
        )
        patient.insert(0, "repetition", int(repetition))
        patient.insert(0, "protocol", protocol)
        patient.insert(0, "architecture", architecture)
        patient_predictions.append(patient)
        for fold, fold_group in group.groupby("fold", sort=True):
            fold_rows.append(
                {
                    "architecture": architecture,
                    "protocol": protocol,
                    "repetition": int(repetition),
                    "fold": int(fold),
                    "auroc": float(
                        roc_auc_score(
                            fold_group["target_binary"], fold_group["probability"]
                        )
                    ),
                    "images": len(fold_group),
                    "patients": int(fold_group["patient_id"].nunique()),
                }
            )
    return (
        pd.DataFrame(patient_rows),
        pd.DataFrame(image_rows),
        pd.DataFrame(fold_rows),
        pd.concat(patient_predictions, ignore_index=True),
    )


def delta_by_repetition(
    patient_metrics: pd.DataFrame, image_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for level, table in (
        ("patient", patient_metrics),
        ("image_sensitivity", image_metrics),
    ):
        pivot = table.pivot(
            index=["architecture", "repetition"], columns="protocol", values="auroc"
        ).reset_index()
        if set(PROTOCOLS).difference(pivot.columns):
            raise ValueError("Falta un protocolo para calcular diferencias emparejadas.")
        pivot["analysis_level"] = level
        pivot["delta_auroc"] = pivot["image_level"] - pivot["patient_level"]
        rows.extend(pivot.to_dict("records"))
    return pd.DataFrame(rows)[
        [
            "architecture",
            "analysis_level",
            "repetition",
            "image_level",
            "patient_level",
            "delta_auroc",
        ]
    ].sort_values(["architecture", "analysis_level", "repetition"])


def _describe(values: Iterable[float]) -> dict[str, float]:
    series = pd.Series(list(values), dtype=float)
    return {
        "mean": float(series.mean()),
        "standard_deviation": float(series.std(ddof=1)),
        "median": float(series.median()),
        "minimum": float(series.min()),
        "maximum": float(series.max()),
    }


def summarize_metrics(patient_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_names = ("auroc", "balanced_accuracy", "sensitivity", "specificity")
    for (architecture, protocol), group in patient_metrics.groupby(
        ["architecture", "protocol"], sort=True
    ):
        for metric in metric_names:
            rows.append(
                {
                    "architecture": architecture,
                    "protocol": protocol,
                    "metric": metric,
                    **_describe(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def summarize_fold_diagnostics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (architecture, protocol), group in fold_metrics.groupby(
        ["architecture", "protocol"], sort=True
    ):
        values = group["auroc"]
        rows.append(
            {
                "architecture": architecture,
                "protocol": protocol,
                "fold_auroc_median": float(values.median()),
                "fold_auroc_q1": float(values.quantile(0.25)),
                "fold_auroc_q3": float(values.quantile(0.75)),
                "fold_auroc_minimum": float(values.min()),
                "fold_auroc_maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _prediction_wide_tables(
    predictions: pd.DataFrame, patient_predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_wide = patient_predictions.pivot(
        index=["patient_id", "target_binary"],
        columns=["architecture", "protocol", "repetition"],
        values="probability",
    ).sort_index()
    image_wide = predictions.pivot(
        index=["relative_path", "patient_id", "target_binary"],
        columns=["architecture", "protocol", "repetition"],
        values="probability",
    ).sort_index()
    expected_columns = len(ARCHITECTURES) * len(PROTOCOLS) * N_REPETITIONS
    if patient_wide.shape != (93, expected_columns):
        raise ValueError("La tabla ancha por paciente está incompleta.")
    if image_wide.shape != (1_371, expected_columns):
        raise ValueError("La tabla ancha por imagen está incompleta.")
    return patient_wide, image_wide


def paired_stratified_bootstrap(
    predictions: pd.DataFrame,
    patient_predictions: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    if iterations < 1:
        raise ValueError("iterations debe ser positivo.")
    patient_wide, image_wide = _prediction_wide_tables(
        predictions, patient_predictions
    )
    patient_ids = patient_wide.index.get_level_values("patient_id").to_numpy()
    targets = patient_wide.index.get_level_values("target_binary").to_numpy(dtype=int)
    class_positions = {value: np.flatnonzero(targets == value) for value in (0, 1)}
    if any(len(value) == 0 for value in class_positions.values()):
        raise ValueError("El bootstrap requiere ambas clases.")
    image_patient_ids = image_wide.index.get_level_values("patient_id").to_numpy()
    patient_to_image_positions = {
        patient: np.flatnonzero(image_patient_ids == patient) for patient in patient_ids
    }
    image_targets = image_wide.index.get_level_values("target_binary").to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    values = {
        (architecture, level): np.empty(iterations, dtype=float)
        for architecture in ARCHITECTURES
        for level in ("patient", "image_sensitivity")
    }
    for iteration in range(iterations):
        sampled_positions = np.concatenate(
            [
                rng.choice(positions, size=len(positions), replace=True)
                for positions in class_positions.values()
            ]
        )
        sampled_targets = targets[sampled_positions]
        sampled_patients = patient_ids[sampled_positions]
        sampled_image_positions = np.concatenate(
            [patient_to_image_positions[patient] for patient in sampled_patients]
        )
        sampled_image_targets = image_targets[sampled_image_positions]
        for architecture in ARCHITECTURES:
            patient_deltas: list[float] = []
            image_deltas: list[float] = []
            for repetition in range(1, N_REPETITIONS + 1):
                patient_image = patient_wide[
                    (architecture, "image_level", repetition)
                ].to_numpy()[sampled_positions]
                patient_patient = patient_wide[
                    (architecture, "patient_level", repetition)
                ].to_numpy()[sampled_positions]
                patient_deltas.append(
                    float(roc_auc_score(sampled_targets, patient_image))
                    - float(roc_auc_score(sampled_targets, patient_patient))
                )
                image_image = image_wide[
                    (architecture, "image_level", repetition)
                ].to_numpy()[sampled_image_positions]
                image_patient = image_wide[
                    (architecture, "patient_level", repetition)
                ].to_numpy()[sampled_image_positions]
                image_deltas.append(
                    float(roc_auc_score(sampled_image_targets, image_image))
                    - float(roc_auc_score(sampled_image_targets, image_patient))
                )
            values[(architecture, "patient")][iteration] = float(
                np.mean(patient_deltas)
            )
            values[(architecture, "image_sensitivity")][iteration] = float(
                np.mean(image_deltas)
            )
    rows: list[dict[str, object]] = []
    for (architecture, level), samples in values.items():
        rows.append(
            {
                "architecture": architecture,
                "analysis_level": level,
                "bootstrap_iterations": iterations,
                "bootstrap_seed": seed,
                "bootstrap_method": "paired_stratified_patient_percentile",
                "ci_lower_2_5": float(np.percentile(samples, 2.5)),
                "ci_upper_97_5": float(np.percentile(samples, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def build_delta_summary(
    deltas: pd.DataFrame, bootstrap: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (architecture, level), group in deltas.groupby(
        ["architecture", "analysis_level"], sort=True
    ):
        rows.append(
            {
                "architecture": architecture,
                "analysis_level": level,
                **_describe(group["delta_auroc"]),
            }
        )
    summary = pd.DataFrame(rows)
    return summary.merge(
        bootstrap,
        on=["architecture", "analysis_level"],
        how="left",
        validate="one_to_one",
    )


def ensure_outputs_are_writable(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Ya existen productos de evaluación. Use --overwrite solo después "
            f"de revisarlos:\n{formatted}"
        )


def _write_outputs(
    output_dir: Path,
    tables: dict[str, pd.DataFrame],
    *,
    predictions_root: Path,
    iterations: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False, lineterminator="\n")
        output_paths[name] = path
    parameters = {
        "analysis_contract": {
            "primary_unit": "patient",
            "primary_metric": "AUROC",
            "threshold": THRESHOLD,
            "image_level_role": "methodological_sensitivity",
            "fold_auroc_role": "descriptive_diagnostic",
            "bootstrap_scope": "conditional_on_trained_models_and_five_repetitions",
        },
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
        "predictions_root": str(predictions_root.resolve()),
        "inputs": {
            "analytic_cohort_sha256": sha256(ANALYTIC_COHORT_CSV),
            "image_folds_sha256": sha256(IMAGE_FOLDS_CSV),
            "patient_folds_sha256": sha256(PATIENT_FOLDS_CSV),
            "seed_schedule_sha256": sha256(SEED_SCHEDULE_CSV),
            "partition_parameters_sha256": sha256(PARTITION_PARAMETERS_JSON),
        },
        "outputs": {
            path.relative_to(output_dir).as_posix(): sha256(path)
            for path in output_paths.values()
        },
    }
    (output_dir / "parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    args = parse_args()
    validate_evaluation_configuration(args.bootstrap_iterations, args.bootstrap_seed)
    output_dir = args.output_dir.resolve()
    names = (
        "patient_metrics",
        "image_sensitivity_metrics",
        "fold_diagnostics",
        "confusion_matrices",
        "delta_by_repetition",
        "metric_summary",
        "fold_diagnostics_summary",
        "delta_summary",
    )
    paths = [output_dir / f"{name}.csv" for name in names]
    paths.append(output_dir / "parameters.json")
    ensure_outputs_are_writable(paths, args.overwrite)

    predictions = load_all_predictions(args.predictions_root)
    patient_metrics, image_metrics, fold_metrics, patient_predictions = (
        compute_metric_tables(predictions)
    )
    deltas = delta_by_repetition(patient_metrics, image_metrics)
    bootstrap = paired_stratified_bootstrap(
        predictions,
        patient_predictions,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    delta_summary = build_delta_summary(deltas, bootstrap)
    confusion = patient_metrics[
        ["architecture", "protocol", "repetition", "tn", "fp", "fn", "tp"]
    ].copy()
    tables = {
        "patient_metrics": patient_metrics,
        "image_sensitivity_metrics": image_metrics,
        "fold_diagnostics": fold_metrics,
        "confusion_matrices": confusion,
        "delta_by_repetition": deltas,
        "metric_summary": summarize_metrics(patient_metrics),
        "fold_diagnostics_summary": summarize_fold_diagnostics(fold_metrics),
        "delta_summary": delta_summary,
    }
    _write_outputs(
        output_dir,
        tables,
        predictions_root=args.predictions_root,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
