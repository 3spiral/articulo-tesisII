"""Construye y audita cinco repeticiones de folds por imagen y paciente.

La entrada es la cohorte analítica corregida. La implementación separa las
corrientes aleatorias de partición y entrenamiento, conserva una programación
emparejada y produce la auditoría descriptiva del solapamiento real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

from src.data.build_beh_analytic_cohort import natural_path_key, parse_boolean
from src.settings import PROJECT_ROOT


ROOT_SEEDS = (42, 43, 44, 45, 46)
N_REPETITIONS = len(ROOT_SEEDS)
N_SPLITS = 5
EXPECTED_IMAGES = 1_371
EXPECTED_PATIENTS = 93
EXPECTED_CLASS_IMAGES = {0: 420, 1: 951}
EXPECTED_CLASS_PATIENTS = {0: 29, 1: 64}

IMAGE_PROTOCOL = "image_level"
PATIENT_PROTOCOL = "patient_level"
IMAGE_STRATEGY = "StratifiedKFold_image_level_5fold"
PATIENT_STRATEGY = "StratifiedKFold_patient_level_5fold"

DEFAULT_COHORT = PROJECT_ROOT / "metadata" / "BEH_analytic_cohort.csv"
DEFAULT_FOLDS_DIR = PROJECT_ROOT / "metadata" / "folds"
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "metadata" / "qc" / "partition_folds"

IMAGE_OUTPUT_COLUMNS = [
    "repetition",
    "root_seed",
    "partition_seed",
    "relative_path",
    "patient_id",
    "target_binary",
    "sha256",
    "cv_fold",
    "training_seed",
    "cv_strategy",
]

PATIENT_OUTPUT_COLUMNS = [
    "repetition",
    "root_seed",
    "partition_seed",
    "patient_id",
    "target_binary",
    "image_count",
    "cv_fold",
    "training_seed",
    "cv_strategy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera cinco repeticiones de folds por imagen y paciente para BEH."
        )
    )
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permite reemplazar productos existentes después de revisarlos.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spawn_key_text(sequence: np.random.SeedSequence) -> str:
    return ".".join(str(value) for value in sequence.spawn_key)


def derived_uint32(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def derive_seed_schedule() -> pd.DataFrame:
    rows: list[dict[str, int | str]] = []
    for repetition, root_seed in enumerate(ROOT_SEEDS, start=1):
        root = np.random.SeedSequence(root_seed)
        image_sequence, patient_sequence, training_sequence = root.spawn(3)
        image_seed = derived_uint32(image_sequence)
        patient_seed = derived_uint32(patient_sequence)
        training_children = training_sequence.spawn(N_SPLITS)
        for fold, training_child in enumerate(training_children, start=1):
            rows.append(
                {
                    "repetition": repetition,
                    "root_seed": root_seed,
                    "image_partition_seed": image_seed,
                    "image_partition_spawn_key": spawn_key_text(image_sequence),
                    "patient_partition_seed": patient_seed,
                    "patient_partition_spawn_key": spawn_key_text(patient_sequence),
                    "fold": fold,
                    "training_seed": derived_uint32(training_child),
                    "training_spawn_key": spawn_key_text(training_child),
                }
            )
    schedule = pd.DataFrame(rows)
    if len(schedule) != N_REPETITIONS * N_SPLITS:
        raise ValueError("La programación de semillas no contiene 25 filas.")
    if schedule["training_seed"].duplicated().any():
        raise ValueError("La programación produjo semillas de entrenamiento repetidas.")
    partition_seeds = pd.concat(
        [schedule["image_partition_seed"], schedule["patient_partition_seed"]]
    ).unique()
    if len(partition_seeds) != N_REPETITIONS * 2:
        raise ValueError("La programación produjo semillas de partición repetidas.")
    return schedule


def load_analytic_cohort(path: Path) -> pd.DataFrame:
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
        raise ValueError("Faltan columnas en la cohorte: " + ", ".join(missing))

    cohort = cohort.loc[parse_boolean(cohort["analytic_include"])].copy()
    cohort["relative_path"] = (
        cohort["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    )
    cohort["patient_id"] = cohort["patient_id"].astype(str)
    cohort["sha256"] = cohort["sha256"].astype(str)
    cohort["target_binary"] = pd.to_numeric(
        cohort["target_binary"], errors="raise"
    ).astype(int)
    order = sorted(
        range(len(cohort)),
        key=lambda position: natural_path_key(cohort.iloc[position]["relative_path"]),
    )
    cohort = cohort.iloc[order].reset_index(drop=True)

    if len(cohort) != EXPECTED_IMAGES:
        raise ValueError(f"Se esperaban {EXPECTED_IMAGES} imágenes analíticas.")
    if cohort["patient_id"].nunique() != EXPECTED_PATIENTS:
        raise ValueError(f"Se esperaban {EXPECTED_PATIENTS} pacientes analíticos.")
    if cohort["relative_path"].duplicated().any():
        raise ValueError("La cohorte contiene rutas duplicadas.")
    if cohort["sha256"].duplicated().any():
        raise ValueError("La cohorte contiene hashes duplicados.")
    if cohort.groupby("patient_id")["target_binary"].nunique().max() != 1:
        raise ValueError("Un paciente tiene más de una etiqueta.")
    image_counts = cohort["target_binary"].value_counts().sort_index().to_dict()
    if image_counts != EXPECTED_CLASS_IMAGES:
        raise ValueError(f"Conteos inesperados de imágenes por clase: {image_counts}")
    patient_counts = (
        cohort[["patient_id", "target_binary"]]
        .drop_duplicates()["target_binary"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
    if patient_counts != EXPECTED_CLASS_PATIENTS:
        raise ValueError(f"Conteos inesperados de pacientes por clase: {patient_counts}")
    return cohort[["relative_path", "patient_id", "target_binary", "sha256"]]


def patient_table(cohort: pd.DataFrame) -> pd.DataFrame:
    patients = (
        cohort.groupby(["patient_id", "target_binary"], as_index=False)
        .size()
        .rename(columns={"size": "image_count"})
    )
    order = sorted(
        range(len(patients)),
        key=lambda position: natural_path_key(patients.iloc[position]["patient_id"]),
    )
    return patients.iloc[order].reset_index(drop=True)


def assign_folds(
    cohort: pd.DataFrame,
    schedule: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_outputs: list[pd.DataFrame] = []
    patient_outputs: list[pd.DataFrame] = []
    patients = patient_table(cohort)

    for repetition in range(1, N_REPETITIONS + 1):
        repetition_schedule = schedule.loc[schedule["repetition"].eq(repetition)]
        root_seed = int(repetition_schedule["root_seed"].iloc[0])
        image_seed = int(repetition_schedule["image_partition_seed"].iloc[0])
        patient_seed = int(repetition_schedule["patient_partition_seed"].iloc[0])
        training_by_fold = repetition_schedule.set_index("fold")["training_seed"]

        image_assignment = cohort.copy()
        image_assignment["cv_fold"] = 0
        image_splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=image_seed,
        )
        image_dummy = np.zeros((len(image_assignment), 1), dtype=np.uint8)
        for fold, (_, validation_indices) in enumerate(
            image_splitter.split(image_dummy, image_assignment["target_binary"]), start=1
        ):
            image_assignment.loc[validation_indices, "cv_fold"] = fold
        image_assignment.insert(0, "repetition", repetition)
        image_assignment.insert(1, "root_seed", root_seed)
        image_assignment.insert(2, "partition_seed", image_seed)
        image_assignment["training_seed"] = (
            image_assignment["cv_fold"].map(training_by_fold).astype("uint32")
        )
        image_assignment["cv_strategy"] = IMAGE_STRATEGY
        image_outputs.append(image_assignment[IMAGE_OUTPUT_COLUMNS])

        patient_assignment = patients.copy()
        patient_assignment["cv_fold"] = 0
        patient_splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=patient_seed,
        )
        patient_dummy = np.zeros((len(patient_assignment), 1), dtype=np.uint8)
        for fold, (_, validation_indices) in enumerate(
            patient_splitter.split(patient_dummy, patient_assignment["target_binary"]),
            start=1,
        ):
            patient_assignment.loc[validation_indices, "cv_fold"] = fold
        patient_assignment.insert(0, "repetition", repetition)
        patient_assignment.insert(1, "root_seed", root_seed)
        patient_assignment.insert(2, "partition_seed", patient_seed)
        patient_assignment["training_seed"] = (
            patient_assignment["cv_fold"].map(training_by_fold).astype("uint32")
        )
        patient_assignment["cv_strategy"] = PATIENT_STRATEGY
        patient_outputs.append(patient_assignment[PATIENT_OUTPUT_COLUMNS])

    image_folds = pd.concat(image_outputs, ignore_index=True)
    patient_folds = pd.concat(patient_outputs, ignore_index=True)
    validate_fold_tables(cohort, image_folds, patient_folds, schedule)
    return image_folds, patient_folds


def validate_fold_tables(
    cohort: pd.DataFrame,
    image_folds: pd.DataFrame,
    patient_folds: pd.DataFrame,
    schedule: pd.DataFrame,
) -> None:
    if len(image_folds) != EXPECTED_IMAGES * N_REPETITIONS:
        raise ValueError("La tabla por imagen no contiene 6,855 filas.")
    if len(patient_folds) != EXPECTED_PATIENTS * N_REPETITIONS:
        raise ValueError("La tabla por paciente no contiene 465 filas.")

    expected_paths = set(cohort["relative_path"])
    expected_patients = set(cohort["patient_id"])
    for repetition in range(1, N_REPETITIONS + 1):
        image_rep = image_folds.loc[image_folds["repetition"].eq(repetition)]
        patient_rep = patient_folds.loc[patient_folds["repetition"].eq(repetition)]
        if len(image_rep) != EXPECTED_IMAGES or set(image_rep["relative_path"]) != expected_paths:
            raise ValueError(f"Cobertura por imagen incorrecta en repetición {repetition}.")
        if image_rep["relative_path"].duplicated().any():
            raise ValueError(f"Rutas repetidas por imagen en repetición {repetition}.")
        if len(patient_rep) != EXPECTED_PATIENTS or set(patient_rep["patient_id"]) != expected_patients:
            raise ValueError(f"Cobertura por paciente incorrecta en repetición {repetition}.")
        if patient_rep["patient_id"].duplicated().any():
            raise ValueError(f"Pacientes repetidos en repetición {repetition}.")
        if set(image_rep["cv_fold"]) != set(range(1, N_SPLITS + 1)):
            raise ValueError("La tabla por imagen no contiene exactamente los folds 1-5.")
        if set(patient_rep["cv_fold"]) != set(range(1, N_SPLITS + 1)):
            raise ValueError("La tabla por paciente no contiene exactamente los folds 1-5.")

        expected_training = schedule.loc[
            schedule["repetition"].eq(repetition), ["fold", "training_seed"]
        ].set_index("fold")["training_seed"]
        for table in (image_rep, patient_rep):
            actual = table.groupby("cv_fold")["training_seed"].first().astype("uint32")
            if not actual.equals(expected_training.astype("uint32")):
                raise ValueError("La semilla de entrenamiento no coincide con el schedule.")

        patient_class_by_fold = (
            patient_rep.groupby(["cv_fold", "target_binary"]).size().unstack(fill_value=0)
        )
        if sorted(patient_class_by_fold[0].tolist()) != [5, 6, 6, 6, 6]:
            raise ValueError("Estratificación inesperada de pacientes S0-S1.")
        if sorted(patient_class_by_fold[1].tolist()) != [12, 13, 13, 13, 13]:
            raise ValueError("Estratificación inesperada de pacientes S2-S3.")


def expand_patient_assignments(
    cohort: pd.DataFrame,
    patient_folds: pd.DataFrame,
) -> pd.DataFrame:
    expanded = cohort.merge(
        patient_folds[
            [
                "repetition",
                "root_seed",
                "partition_seed",
                "patient_id",
                "target_binary",
                "cv_fold",
                "training_seed",
                "cv_strategy",
            ]
        ],
        on=["patient_id", "target_binary"],
        how="inner",
        validate="many_to_many",
    )
    return expanded[
        [
            "repetition",
            "root_seed",
            "partition_seed",
            "relative_path",
            "patient_id",
            "target_binary",
            "sha256",
            "cv_fold",
            "training_seed",
            "cv_strategy",
        ]
    ]


def numeric_summary(values: pd.Series, prefix: str) -> dict[str, float | int | None]:
    valid = values.dropna()
    if valid.empty:
        return {
            f"{prefix}_median": None,
            f"{prefix}_q1": None,
            f"{prefix}_q3": None,
            f"{prefix}_minimum": None,
            f"{prefix}_maximum": None,
        }
    return {
        f"{prefix}_median": float(valid.median()),
        f"{prefix}_q1": float(valid.quantile(0.25)),
        f"{prefix}_q3": float(valid.quantile(0.75)),
        f"{prefix}_minimum": float(valid.min()),
        f"{prefix}_maximum": float(valid.max()),
    }


def audit_partitions(
    cohort: pd.DataFrame,
    image_folds: pd.DataFrame,
    patient_folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_counts = cohort.groupby("patient_id").size().to_dict()
    expanded_patient = expand_patient_assignments(cohort, patient_folds)
    exposure_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for protocol, assignments in (
        (IMAGE_PROTOCOL, image_folds),
        (PATIENT_PROTOCOL, expanded_patient),
    ):
        for repetition in range(1, N_REPETITIONS + 1):
            repetition_rows = assignments.loc[
                assignments["repetition"].eq(repetition)
            ]
            for fold in range(1, N_SPLITS + 1):
                validation = repetition_rows.loc[repetition_rows["cv_fold"].eq(fold)]
                training = repetition_rows.loc[~repetition_rows["cv_fold"].eq(fold)]
                training_patients = set(training["patient_id"])
                validation_patients = set(validation["patient_id"])
                training_hashes = set(training["sha256"])
                validation_hashes = set(validation["sha256"])
                overlapping_patients = training_patients.intersection(validation_patients)
                overlapping_hashes = training_hashes.intersection(validation_hashes)
                training_patient_image_counts = training.groupby("patient_id").size().to_dict()

                fold_exposures: list[dict[str, Any]] = []
                for row in validation.itertuples(index=False):
                    total_images = int(patient_counts[row.patient_id])
                    sisters_in_training = int(
                        training_patient_image_counts.get(row.patient_id, 0)
                    )
                    ratio = (
                        sisters_in_training / (total_images - 1)
                        if total_images > 1
                        else None
                    )
                    exposure = {
                        "protocol": protocol,
                        "repetition": repetition,
                        "root_seed": int(row.root_seed),
                        "fold": fold,
                        "relative_path": row.relative_path,
                        "patient_id": row.patient_id,
                        "target_binary": int(row.target_binary),
                        "sha256": row.sha256,
                        "patient_total_images": total_images,
                        "sister_images_in_training": sisters_in_training,
                        "sister_exposure_ratio": ratio,
                        "patient_in_training": row.patient_id in training_patients,
                        "exposed_to_sister_image": sisters_in_training > 0,
                        "hash_in_training": row.sha256 in training_hashes,
                    }
                    exposure_rows.append(exposure)
                    fold_exposures.append(exposure)

                exposure_frame = pd.DataFrame(fold_exposures)
                c_summary = numeric_summary(
                    exposure_frame["sister_images_in_training"], "c_i"
                )
                r_summary = numeric_summary(
                    exposure_frame["sister_exposure_ratio"], "r_i"
                )
                summary_rows.append(
                    {
                        "protocol": protocol,
                        "repetition": repetition,
                        "root_seed": int(validation["root_seed"].iloc[0]),
                        "fold": fold,
                        "training_images": len(training),
                        "validation_images": len(validation),
                        "validation_class_0_images": int(
                            validation["target_binary"].eq(0).sum()
                        ),
                        "validation_class_1_images": int(
                            validation["target_binary"].eq(1).sum()
                        ),
                        "training_patients": len(training_patients),
                        "validation_patients": len(validation_patients),
                        "overlapping_patients": len(overlapping_patients),
                        "validation_patients_in_training_proportion": (
                            len(overlapping_patients) / len(validation_patients)
                        ),
                        "exposed_validation_images": int(
                            exposure_frame["exposed_to_sister_image"].sum()
                        ),
                        "exposed_validation_images_proportion": float(
                            exposure_frame["exposed_to_sister_image"].mean()
                        ),
                        "identifier_intersection_count": len(overlapping_patients),
                        "hash_intersection_count": len(overlapping_hashes),
                        **c_summary,
                        **r_summary,
                    }
                )

    exposure = pd.DataFrame(exposure_rows)
    summary = pd.DataFrame(summary_rows)
    validate_partition_audit(exposure, summary)
    return exposure, summary


def validate_partition_audit(exposure: pd.DataFrame, summary: pd.DataFrame) -> None:
    expected_exposure_rows = EXPECTED_IMAGES * N_REPETITIONS * 2
    if len(exposure) != expected_exposure_rows:
        raise ValueError(
            f"Se esperaban {expected_exposure_rows} filas de exposición; "
            f"se obtuvieron {len(exposure)}."
        )
    if len(summary) != N_REPETITIONS * N_SPLITS * 2:
        raise ValueError("El resumen no contiene las 50 combinaciones esperadas.")
    if summary["hash_intersection_count"].ne(0).any():
        raise ValueError("Existe contenido exacto compartido entre entrenamiento y validación.")
    if exposure["hash_in_training"].any():
        raise ValueError("Una imagen de validación tiene el mismo hash en entrenamiento.")

    patient_summary = summary.loc[summary["protocol"].eq(PATIENT_PROTOCOL)]
    patient_exposure = exposure.loc[exposure["protocol"].eq(PATIENT_PROTOCOL)]
    if patient_summary["overlapping_patients"].ne(0).any():
        raise ValueError("La partición por paciente comparte identificadores.")
    if patient_summary["exposed_validation_images"].ne(0).any():
        raise ValueError("La partición por paciente expone imágenes hermanas.")
    if patient_exposure["patient_in_training"].any():
        raise ValueError("Un paciente de validación aparece en entrenamiento.")
    if patient_exposure["sister_images_in_training"].ne(0).any():
        raise ValueError("Una imagen del protocolo por paciente tiene hermanas en entrenamiento.")

    coverage = exposure.groupby(["protocol", "repetition"])["relative_path"].agg(
        ["count", "nunique"]
    )
    if not coverage["count"].eq(EXPECTED_IMAGES).all():
        raise ValueError("La validación OOF no cubre 1,371 imágenes por repetición.")
    if not coverage["nunique"].eq(EXPECTED_IMAGES).all():
        raise ValueError("Una imagen se validó más de una vez en una repetición.")


def ensure_writable_targets(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "Ya existen productos de esta etapa. Use --overwrite solo después "
            "de revisarlos:\n" + formatted
        )


def relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_outputs(
    cohort_path: Path,
    folds_dir: Path,
    audit_dir: Path,
    schedule: pd.DataFrame,
    image_folds: pd.DataFrame,
    patient_folds: pd.DataFrame,
    exposure: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    image_path = folds_dir / "BEH_image_folds.csv"
    patient_path = folds_dir / "BEH_patient_folds.csv"
    schedule_path = audit_dir / "seed_schedule.csv"
    exposure_path = audit_dir / "image_exposure.csv"
    summary_path = audit_dir / "fold_summary.csv"
    parameters_path = audit_dir / "parameters.json"

    folds_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    image_folds.to_csv(image_path, index=False, lineterminator="\n")
    patient_folds.to_csv(patient_path, index=False, lineterminator="\n")
    schedule.to_csv(schedule_path, index=False, lineterminator="\n")
    exposure.to_csv(exposure_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")

    outputs = [image_path, patient_path, schedule_path, exposure_path, summary_path]
    parameters = {
        "analytic_cohort": relative_to_project(cohort_path),
        "analytic_cohort_sha256": sha256(cohort_path),
        "root_seeds": list(ROOT_SEEDS),
        "repetitions": N_REPETITIONS,
        "folds_per_repetition": N_SPLITS,
        "seed_derivation": {
            "implementation": "numpy.random.SeedSequence",
            "root_spawn_domains": [
                "image_partition",
                "patient_partition",
                "training",
            ],
            "training_children": "one per fold; reused across protocols and architectures",
        },
        "strategies": {
            IMAGE_PROTOCOL: IMAGE_STRATEGY,
            PATIENT_PROTOCOL: PATIENT_STRATEGY,
        },
        "cohort": {
            "images": EXPECTED_IMAGES,
            "patients": EXPECTED_PATIENTS,
            "class_images": {str(k): v for k, v in EXPECTED_CLASS_IMAGES.items()},
            "class_patients": {
                str(k): v for k, v in EXPECTED_CLASS_PATIENTS.items()
            },
        },
        "audit_contract": {
            "patient_protocol_identifier_overlap": 0,
            "patient_protocol_sister_exposure": 0,
            "all_protocols_hash_overlap": 0,
            "artificial_overlap_levels": False,
        },
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "outputs": {
            relative_to_project(path): sha256(path)
            for path in outputs
        },
    }
    parameters_path.write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    args = parse_args()
    cohort_path = args.cohort.resolve()
    folds_dir = args.folds_dir.resolve()
    audit_dir = args.audit_dir.resolve()
    targets = [
        folds_dir / "BEH_image_folds.csv",
        folds_dir / "BEH_patient_folds.csv",
        audit_dir / "seed_schedule.csv",
        audit_dir / "image_exposure.csv",
        audit_dir / "fold_summary.csv",
        audit_dir / "parameters.json",
    ]
    ensure_writable_targets(targets, args.overwrite)

    cohort = load_analytic_cohort(cohort_path)
    schedule = derive_seed_schedule()
    image_folds, patient_folds = assign_folds(cohort, schedule)
    exposure, summary = audit_partitions(cohort, image_folds, patient_folds)
    write_outputs(
        cohort_path,
        folds_dir,
        audit_dir,
        schedule,
        image_folds,
        patient_folds,
        exposure,
        summary,
    )
    print(f"Folds por imagen guardados: {folds_dir / 'BEH_image_folds.csv'}")
    print(f"Folds por paciente guardados: {folds_dir / 'BEH_patient_folds.csv'}")
    print(f"Auditoría guardada: {audit_dir}")


if __name__ == "__main__":
    main()
