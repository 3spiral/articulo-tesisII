"""Genera y audita los folds por imagen del experimento principal BEH.

El script trabaja exclusivamente con el manifiesto versionado. No abre ni
modifica imágenes, no altera ``BEHSOF_manifest.csv`` y conserva por separado
la asignación deliberadamente susceptible a solapamiento de pacientes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

from src.settings import BEH_PATIENT_FOLDS_CSV, PROJECT_ROOT


N_SPLITS = 5
RANDOM_STATE = 42
STRATEGY = "StratifiedKFold_image_level_5fold"
PATIENT_STRATEGY = "StratifiedKFold_patient_level_5fold"
EXPECTED_IMAGES = 1_388
EXPECTED_PATIENTS = 95
EXPECTED_CLASS_COUNTS = {0: 421, 1: 967}

DEFAULT_MANIFEST = PROJECT_ROOT / "metadata" / "BEHSOF_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "metadata" / "BEH_image_folds.csv"
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "metadata" / "qc" / "image_level_folds"

OUTPUT_COLUMNS = [
    "relative_path",
    "patient_id",
    "target_binary",
    "cv_fold",
    "cv_seed",
    "cv_strategy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera y audita cinco folds estratificados por imagen para BEH."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar productos existentes después de recalcularlos.",
    )
    return parser.parse_args()


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
        values = sorted(normalized[parsed.isna()].unique())
        raise ValueError(f"Valores booleanos no reconocidos: {values}")
    return parsed.astype(bool)


def load_eligible_images(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    required = {
        "patient_id",
        "center",
        "relative_path",
        "target_binary",
        "include_final",
        "patient_eligible",
        "cv_fold",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError("Faltan columnas en el manifiesto: " + ", ".join(missing))

    eligible = manifest.loc[
        manifest["center"].astype(str).str.upper().eq("BEH")
        & parse_boolean(manifest["include_final"])
        & parse_boolean(manifest["patient_eligible"])
    ].copy()
    eligible["target_binary"] = pd.to_numeric(
        eligible["target_binary"], errors="raise"
    ).astype(int)
    eligible["cv_fold"] = pd.to_numeric(eligible["cv_fold"], errors="raise").astype(int)
    eligible["relative_path"] = eligible["relative_path"].astype(str)
    eligible["patient_id"] = eligible["patient_id"].astype(str)

    if eligible["relative_path"].duplicated().any():
        raise ValueError("La muestra elegible contiene relative_path duplicados.")
    if set(eligible["target_binary"].unique()) != {0, 1}:
        raise ValueError("target_binary debe contener exactamente las clases 0 y 1.")
    if eligible.groupby("patient_id")["target_binary"].nunique().max() != 1:
        raise ValueError("Un paciente elegible tiene más de una clase objetivo.")
    if len(eligible) != EXPECTED_IMAGES:
        raise ValueError(f"Se esperaban {EXPECTED_IMAGES} imágenes; se obtuvieron {len(eligible)}.")
    if eligible["patient_id"].nunique() != EXPECTED_PATIENTS:
        raise ValueError(
            f"Se esperaban {EXPECTED_PATIENTS} pacientes; se obtuvieron "
            f"{eligible['patient_id'].nunique()}."
        )
    if eligible["target_binary"].value_counts().sort_index().to_dict() != EXPECTED_CLASS_COUNTS:
        raise ValueError("La distribución de clases no coincide con el protocolo confirmado.")

    return eligible.sort_values("relative_path", kind="stable").reset_index(drop=True)


def assign_image_folds(eligible: pd.DataFrame) -> pd.DataFrame:
    assigned = eligible[["relative_path", "patient_id", "target_binary"]].copy()
    assigned["cv_fold"] = 0
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    dummy_features = np.zeros((len(assigned), 1), dtype=np.uint8)
    for fold, (_, validation_indices) in enumerate(
        splitter.split(dummy_features, assigned["target_binary"]), start=1
    ):
        assigned.loc[validation_indices, "cv_fold"] = fold

    assigned["cv_seed"] = RANDOM_STATE
    assigned["cv_strategy"] = STRATEGY
    assigned = assigned[OUTPUT_COLUMNS]
    validate_image_folds(assigned)
    return assigned


def load_patient_fold_reference(
    patient_folds_path: Path, eligible: pd.DataFrame
) -> pd.DataFrame:
    """Valida la partición por paciente usada realmente por el entrenamiento."""

    patient_folds = pd.read_csv(patient_folds_path, dtype={"patient_id": str})
    required = ["patient_id", "target_binary", "cv_fold", "cv_seed", "cv_strategy"]
    missing = sorted(set(required).difference(patient_folds.columns))
    if missing:
        raise ValueError(
            "Faltan columnas en la partición por paciente: " + ", ".join(missing)
        )
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

    if len(patient_folds) != EXPECTED_PATIENTS:
        raise ValueError("La partición por paciente no contiene exactamente 95 filas.")
    if patient_folds["patient_id"].duplicated().any():
        raise ValueError("La partición por paciente contiene identificadores duplicados.")
    if set(patient_folds["cv_fold"]) != set(range(1, N_SPLITS + 1)):
        raise ValueError("La partición por paciente no contiene exactamente los folds 1-5.")
    if not patient_folds["cv_seed"].eq(RANDOM_STATE).all():
        raise ValueError("La partición por paciente no utiliza la semilla 42 fijada.")
    if not patient_folds["cv_strategy"].eq(PATIENT_STRATEGY).all():
        raise ValueError("La estrategia declarada en la partición por paciente es inesperada.")

    patient_reference = eligible[
        ["relative_path", "patient_id", "target_binary", "cv_fold"]
    ].merge(
        patient_folds,
        on="patient_id",
        how="outer",
        validate="many_to_one",
        suffixes=("_manifest", "_patient_file"),
        indicator=True,
    )
    if not patient_reference["_merge"].eq("both").all():
        raise ValueError("Los pacientes del manifiesto y de BEH_patient_folds.csv difieren.")
    if not patient_reference["target_binary_manifest"].eq(
        patient_reference["target_binary_patient_file"]
    ).all():
        raise ValueError("Las etiquetas difieren entre el manifiesto y los folds por paciente.")
    if not patient_reference["cv_fold_manifest"].eq(
        patient_reference["cv_fold_patient_file"]
    ).all():
        raise ValueError("Los folds por paciente difieren entre el manifiesto y su archivo dedicado.")

    return patient_reference.rename(
        columns={
            "target_binary_manifest": "target_binary",
            "cv_fold_patient_file": "patient_level_fold",
        }
    )[["relative_path", "patient_id", "target_binary", "patient_level_fold"]]


def validate_image_folds(assigned: pd.DataFrame) -> None:
    if len(assigned) != EXPECTED_IMAGES:
        raise ValueError("La asignación no cubre las 1,388 imágenes esperadas.")
    if assigned["relative_path"].duplicated().any():
        raise ValueError("Una imagen aparece más de una vez en la asignación.")
    if set(assigned["cv_fold"].unique()) != set(range(1, N_SPLITS + 1)):
        raise ValueError("Los folds generados no son exactamente 1-5.")
    fold_sizes = sorted(assigned.groupby("cv_fold").size().tolist())
    if fold_sizes != [277, 277, 278, 278, 278]:
        raise ValueError(f"Tamaños inesperados de folds: {fold_sizes}")
    class_by_fold = assigned.groupby(["cv_fold", "target_binary"]).size().unstack(fill_value=0)
    if sorted(class_by_fold[0].tolist()) != [84, 84, 84, 84, 85]:
        raise ValueError("Distribución inesperada de imágenes de clase 0 entre folds.")
    if sorted(class_by_fold[1].tolist()) != [193, 193, 193, 194, 194]:
        raise ValueError("Distribución inesperada de imágenes de clase 1 entre folds.")


def make_fold_audit(
    eligible: pd.DataFrame,
    image_folds: pd.DataFrame,
    patient_folds_path: Path = BEH_PATIENT_FOLDS_CSV,
) -> pd.DataFrame:
    patient_folds = load_patient_fold_reference(patient_folds_path, eligible)
    rows: list[dict[str, int | str]] = []

    for protocol, fold_column, data in (
        ("image_level", "cv_fold", image_folds),
        ("patient_level", "patient_level_fold", patient_folds),
    ):
        for fold in range(1, N_SPLITS + 1):
            validation = data.loc[data[fold_column].eq(fold)]
            training = data.loc[~data[fold_column].eq(fold)]
            validation_patients = set(validation["patient_id"])
            training_patients = set(training["patient_id"])
            overlap = validation_patients.intersection(training_patients)
            rows.append(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "training_images": len(training),
                    "validation_images": len(validation),
                    "validation_class_0_images": int(validation["target_binary"].eq(0).sum()),
                    "validation_class_1_images": int(validation["target_binary"].eq(1).sum()),
                    "training_patients": len(training_patients),
                    "validation_patients": len(validation_patients),
                    "overlapping_patients": len(overlap),
                }
            )

    audit = pd.DataFrame(rows)
    image_rows = audit.loc[audit["protocol"].eq("image_level")]
    patient_rows = audit.loc[audit["protocol"].eq("patient_level")]
    if not image_rows["overlapping_patients"].eq(image_rows["validation_patients"]).all():
        raise ValueError("No todos los pacientes de validación por imagen aparecen en entrenamiento.")
    if not patient_rows["overlapping_patients"].eq(0).all():
        raise ValueError("La partición de referencia por paciente presenta solapamiento.")
    return audit


def ensure_writable_targets(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Ya existen productos de esta etapa. Use --overwrite solo después de revisarlos:\n"
            + formatted
        )


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()
    audit_dir = args.audit_dir.resolve()
    audit_path = audit_dir / "fold_audit.csv"
    parameters_path = audit_dir / "parameters.json"
    ensure_writable_targets(
        [output_path, audit_path, parameters_path], overwrite=args.overwrite
    )

    eligible = load_eligible_images(manifest_path)
    image_folds = assign_image_folds(eligible)
    audit = make_fold_audit(eligible, image_folds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    image_folds.to_csv(output_path, index=False, lineterminator="\n")
    audit.to_csv(audit_path, index=False, lineterminator="\n")
    parameters = {
        "manifest": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "patient_folds": BEH_PATIENT_FOLDS_CSV.relative_to(PROJECT_ROOT).as_posix(),
        "patient_folds_sha256": sha256(BEH_PATIENT_FOLDS_CSV),
        "eligible_images": len(eligible),
        "eligible_patients": int(eligible["patient_id"].nunique()),
        "class_counts": {
            str(key): int(value)
            for key, value in eligible["target_binary"].value_counts().sort_index().items()
        },
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "strategy": STRATEGY,
        "sort_key": "relative_path",
        "filters": {
            "center": "BEH",
            "include_final": True,
            "patient_eligible": True,
        },
        "scikit_learn_version": sklearn.__version__,
    }
    parameters_path.write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"Asignación guardada: {output_path}")
    print(f"Auditoría guardada: {audit_path}")
    print(f"Parámetros guardados: {parameters_path}")
    print("\nResumen por protocolo y fold:")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
