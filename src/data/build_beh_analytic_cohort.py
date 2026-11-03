"""Materializa y audita la cohorte analítica corregida de BEH.

La etapa lee el manifiesto original y los archivos de ``raw`` en modo de solo
lectura. Calcula SHA-256, aplica la política de integridad preespecificada y
genera una capa derivada sin modificar ninguna de sus fuentes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.settings import BEHSOF_IMAGES_DIR, PROJECT_ROOT


EXPECTED_SOURCE_IMAGES = 1_388
EXPECTED_SOURCE_PATIENTS = 95
EXPECTED_SOURCE_CLASS_IMAGES = {0: 421, 1: 967}
EXPECTED_SOURCE_CLASS_PATIENTS = {0: 29, 1: 66}

EXPECTED_ANALYTIC_IMAGES = 1_371
EXPECTED_ANALYTIC_PATIENTS = 93
EXPECTED_ANALYTIC_CLASS_IMAGES = {0: 420, 1: 951}
EXPECTED_ANALYTIC_CLASS_PATIENTS = {0: 29, 1: 64}

AMBIGUOUS_PATIENTS = {"BEH42231", "BEH43131"}
EXPECTED_RETAINED_INTERNAL_DUPLICATE = "BEH62101/Image9.jpg"
EXPECTED_EXCLUDED_INTERNAL_DUPLICATE = "BEH62101/Image10.jpg"

CROSS_PATIENT_REASON = "cross_patient_exact_duplicate_identity_ambiguous"
WITHIN_PATIENT_REASON = "within_patient_exact_duplicate_redundant"

DEFAULT_MANIFEST = PROJECT_ROOT / "metadata" / "BEHSOF_manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "metadata" / "BEH_analytic_cohort.csv"
DEFAULT_AUDIT_DIR = PROJECT_ROOT / "metadata" / "qc" / "cohort_integrity"

OUTPUT_COLUMNS = [
    "source_manifest_row",
    "relative_path",
    "patient_id",
    "filename",
    "fibroscan_s",
    "target_binary",
    "sha256",
    "analytic_include",
    "analytic_exclusion_reason",
    "duplicate_scope",
    "duplicate_group_size",
    "retained_relative_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera y audita la cohorte analítica corregida de BEH."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--images-root", type=Path, default=BEHSOF_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def parse_boolean(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    parsed = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if parsed.isna().any():
        values = sorted(normalized[parsed.isna()].unique())
        raise ValueError(f"Valores booleanos no reconocidos: {values}")
    return parsed.astype(bool)


def natural_path_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """Devuelve una clave estable en la que Image9 precede a Image10."""

    normalized = str(value).replace("\\", "/").casefold()
    parts = re.split(r"(\d+)", normalized)
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    )


def natural_sort(frame: pd.DataFrame) -> pd.DataFrame:
    order = sorted(
        range(len(frame)),
        key=lambda position: natural_path_key(frame.iloc[position]["relative_path"]),
    )
    return frame.iloc[order].reset_index(drop=True)


def class_counts(frame: pd.DataFrame, unit: str) -> dict[int, int]:
    if unit == "images":
        counts = frame["target_binary"].value_counts().sort_index()
    elif unit == "patients":
        counts = (
            frame[["patient_id", "target_binary"]]
            .drop_duplicates()
            ["target_binary"]
            .value_counts()
            .sort_index()
        )
    else:
        raise ValueError(f"Unidad no reconocida: {unit}")
    return {int(key): int(value) for key, value in counts.items()}


def load_source_cohort(
    manifest_path: Path,
    *,
    enforce_protocol_counts: bool = True,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    required = {
        "patient_id",
        "center",
        "relative_path",
        "filename",
        "fibroscan_s",
        "target_binary",
        "include_final",
        "patient_eligible",
    }
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError("Faltan columnas en el manifiesto: " + ", ".join(missing))

    manifest.insert(0, "source_manifest_row", range(2, len(manifest) + 2))
    source = manifest.loc[
        manifest["center"].astype(str).str.upper().eq("BEH")
        & parse_boolean(manifest["include_final"])
        & parse_boolean(manifest["patient_eligible"])
    ].copy()
    source["patient_id"] = source["patient_id"].astype(str)
    source["relative_path"] = (
        source["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    )
    source["filename"] = source["filename"].astype(str)
    source["target_binary"] = pd.to_numeric(
        source["target_binary"], errors="raise"
    ).astype(int)

    if source.empty:
        raise ValueError("El manifiesto no contiene imágenes BEH elegibles.")
    if source["relative_path"].duplicated().any():
        raise ValueError("La fuente contiene rutas relativas duplicadas.")
    if set(source["target_binary"].unique()) != {0, 1}:
        raise ValueError("target_binary debe contener exactamente las clases 0 y 1.")
    if source.groupby("patient_id")["target_binary"].nunique().max() != 1:
        raise ValueError("Un paciente fuente tiene más de una clase objetivo.")

    source = natural_sort(source)
    if enforce_protocol_counts:
        if len(source) != EXPECTED_SOURCE_IMAGES:
            raise ValueError(
                f"Se esperaban {EXPECTED_SOURCE_IMAGES} imágenes fuente; "
                f"se obtuvieron {len(source)}."
            )
        if source["patient_id"].nunique() != EXPECTED_SOURCE_PATIENTS:
            raise ValueError(
                f"Se esperaban {EXPECTED_SOURCE_PATIENTS} pacientes fuente; "
                f"se obtuvieron {source['patient_id'].nunique()}."
            )
        if class_counts(source, "images") != EXPECTED_SOURCE_CLASS_IMAGES:
            raise ValueError("La distribución fuente de imágenes no es la esperada.")
        if class_counts(source, "patients") != EXPECTED_SOURCE_CLASS_PATIENTS:
            raise ValueError("La distribución fuente de pacientes no es la esperada.")
    return source


def hash_source_images(source: pd.DataFrame, images_root: Path) -> pd.DataFrame:
    root = images_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"No existe la raíz de imágenes BEHSOF: {root}")

    hashed = source.copy()
    hashes: list[str] = []
    for relative_path in hashed["relative_path"]:
        image_path = (root / relative_path).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError(f"La ruta sale de la raíz de imágenes: {relative_path}")
        if not image_path.is_file():
            raise FileNotFoundError(f"No existe la imagen fuente: {image_path}")
        hashes.append(sha256(image_path))
    hashed["sha256"] = hashes
    return hashed


def apply_integrity_policy(hashed_source: pd.DataFrame) -> pd.DataFrame:
    required = {"relative_path", "patient_id", "target_binary", "sha256"}
    missing = sorted(required.difference(hashed_source.columns))
    if missing:
        raise ValueError("Faltan columnas para aplicar la política: " + ", ".join(missing))

    analytic = natural_sort(hashed_source.copy())
    analytic["analytic_include"] = True
    analytic["analytic_exclusion_reason"] = ""
    analytic["duplicate_scope"] = ""
    analytic["duplicate_group_size"] = (
        analytic.groupby("sha256")["sha256"].transform("size").astype(int)
    )
    analytic["retained_relative_path"] = ""

    patients_per_hash = analytic.groupby("sha256")["patient_id"].nunique()
    cross_patient_hashes = set(patients_per_hash.loc[patients_per_hash.gt(1)].index)
    ambiguous_patients = set(
        analytic.loc[analytic["sha256"].isin(cross_patient_hashes), "patient_id"]
    )
    cross_patient_mask = analytic["patient_id"].isin(ambiguous_patients)
    analytic.loc[cross_patient_mask, "analytic_include"] = False
    analytic.loc[cross_patient_mask, "analytic_exclusion_reason"] = CROSS_PATIENT_REASON
    analytic.loc[cross_patient_mask, "duplicate_scope"] = "cross_patient"

    remaining = analytic.loc[~cross_patient_mask]
    for (_, _), group in remaining.groupby(["patient_id", "sha256"], sort=False):
        if len(group) < 2:
            continue
        ordered_indices = sorted(
            group.index,
            key=lambda index: natural_path_key(analytic.at[index, "relative_path"]),
        )
        retained_index = ordered_indices[0]
        excluded_indices = ordered_indices[1:]
        retained_path = analytic.at[retained_index, "relative_path"]
        analytic.loc[ordered_indices, "duplicate_scope"] = "within_patient"
        analytic.loc[ordered_indices, "duplicate_group_size"] = len(ordered_indices)
        analytic.loc[ordered_indices, "retained_relative_path"] = retained_path
        analytic.loc[excluded_indices, "analytic_include"] = False
        analytic.loc[
            excluded_indices, "analytic_exclusion_reason"
        ] = WITHIN_PATIENT_REASON

    return natural_sort(analytic)


def validate_protocol_cohort(analytic: pd.DataFrame) -> None:
    if len(analytic) != EXPECTED_SOURCE_IMAGES:
        raise ValueError("La capa derivada no conserva las 1,388 filas fuente.")
    if analytic["relative_path"].duplicated().any():
        raise ValueError("La capa derivada contiene rutas duplicadas.")
    if analytic["sha256"].isna().any() or analytic["sha256"].str.len().ne(64).any():
        raise ValueError("Hay hashes SHA-256 ausentes o inválidos.")

    included = analytic.loc[analytic["analytic_include"]].copy()
    excluded = analytic.loc[~analytic["analytic_include"]].copy()
    if len(included) != EXPECTED_ANALYTIC_IMAGES:
        raise ValueError(
            f"Se esperaban {EXPECTED_ANALYTIC_IMAGES} imágenes analíticas; "
            f"se obtuvieron {len(included)}."
        )
    if included["patient_id"].nunique() != EXPECTED_ANALYTIC_PATIENTS:
        raise ValueError(
            f"Se esperaban {EXPECTED_ANALYTIC_PATIENTS} pacientes analíticos; "
            f"se obtuvieron {included['patient_id'].nunique()}."
        )
    if class_counts(included, "images") != EXPECTED_ANALYTIC_CLASS_IMAGES:
        raise ValueError("La distribución analítica de imágenes no es la esperada.")
    if class_counts(included, "patients") != EXPECTED_ANALYTIC_CLASS_PATIENTS:
        raise ValueError("La distribución analítica de pacientes no es la esperada.")
    if included["sha256"].duplicated().any():
        duplicates = included.loc[included["sha256"].duplicated(False), "relative_path"]
        raise ValueError(
            "Persisten hashes duplicados en la cohorte incluida: "
            + ", ".join(duplicates)
        )
    if included.groupby("patient_id")["target_binary"].nunique().max() != 1:
        raise ValueError("Un paciente analítico tiene más de una clase objetivo.")

    excluded_patients = set(
        analytic.loc[
            analytic["analytic_exclusion_reason"].eq(CROSS_PATIENT_REASON),
            "patient_id",
        ]
    )
    if excluded_patients != AMBIGUOUS_PATIENTS:
        raise ValueError(
            "Los identificadores ambiguos detectados no coinciden con D-044: "
            f"{sorted(excluded_patients)}"
        )
    for patient_id in AMBIGUOUS_PATIENTS:
        patient_rows = analytic.loc[analytic["patient_id"].eq(patient_id)]
        if len(patient_rows) != 8 or patient_rows["analytic_include"].any():
            raise ValueError(f"La exclusión completa de {patient_id} no es correcta.")

    retained = analytic.loc[
        analytic["relative_path"].eq(EXPECTED_RETAINED_INTERNAL_DUPLICATE)
    ]
    dropped = analytic.loc[
        analytic["relative_path"].eq(EXPECTED_EXCLUDED_INTERNAL_DUPLICATE)
    ]
    if len(retained) != 1 or not bool(retained.iloc[0]["analytic_include"]):
        raise ValueError("No se conservó la copia interna fijada por D-059.")
    if len(dropped) != 1 or bool(dropped.iloc[0]["analytic_include"]):
        raise ValueError("No se excluyó la copia interna fijada por D-059.")
    if dropped.iloc[0]["analytic_exclusion_reason"] != WITHIN_PATIENT_REASON:
        raise ValueError("El motivo de exclusión de la copia interna es incorrecto.")

    expected_reason_counts = {CROSS_PATIENT_REASON: 16, WITHIN_PATIENT_REASON: 1}
    reason_counts = excluded["analytic_exclusion_reason"].value_counts().to_dict()
    if reason_counts != expected_reason_counts:
        raise ValueError(f"Conteos inesperados de exclusiones: {reason_counts}")


def make_patient_distribution(analytic: pd.DataFrame) -> pd.DataFrame:
    included = analytic.loc[analytic["analytic_include"]]
    distribution = (
        included.groupby(["patient_id", "target_binary"], as_index=False)
        .size()
        .rename(columns={"size": "image_count"})
    )
    order = sorted(
        range(len(distribution)),
        key=lambda position: natural_path_key(distribution.iloc[position]["patient_id"]),
    )
    return distribution.iloc[order].reset_index(drop=True)


def distribution_statistics(distribution: pd.DataFrame) -> dict[str, Any]:
    def summarize(values: pd.Series) -> dict[str, float | int]:
        return {
            "patients": int(len(values)),
            "images": int(values.sum()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "q1": float(values.quantile(0.25)),
            "q3": float(values.quantile(0.75)),
            "minimum": int(values.min()),
            "maximum": int(values.max()),
        }

    result: dict[str, Any] = {"overall": summarize(distribution["image_count"])}
    for target, group in distribution.groupby("target_binary"):
        result[f"class_{int(target)}"] = summarize(group["image_count"])
    return result


def duplicate_groups(analytic: pd.DataFrame) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for content_hash, group in analytic.groupby("sha256", sort=True):
        if len(group) < 2:
            continue
        groups.append(
            {
                "sha256": content_hash,
                "scope": (
                    "cross_patient"
                    if group["patient_id"].nunique() > 1
                    else "within_patient"
                ),
                "patients": sorted(group["patient_id"].unique()),
                "paths": sorted(group["relative_path"], key=natural_path_key),
            }
        )
    return groups


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
    analytic: pd.DataFrame,
    manifest_path: Path,
    output_path: Path,
    audit_dir: Path,
) -> None:
    exclusions_path = audit_dir / "exclusions.csv"
    distribution_path = audit_dir / "patient_image_distribution.csv"
    summary_path = audit_dir / "summary.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    cohort_output = analytic[OUTPUT_COLUMNS].copy()
    exclusions = cohort_output.loc[~cohort_output["analytic_include"]].copy()
    distribution = make_patient_distribution(analytic)

    cohort_output.to_csv(output_path, index=False, lineterminator="\n")
    exclusions.to_csv(exclusions_path, index=False, lineterminator="\n")
    distribution.to_csv(distribution_path, index=False, lineterminator="\n")

    included = analytic.loc[analytic["analytic_include"]]
    excluded = analytic.loc[~analytic["analytic_include"]]
    summary = {
        "source_manifest": relative_to_project(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "hash_algorithm": "SHA-256",
        "source_filters": {
            "center": "BEH",
            "include_final": True,
            "patient_eligible": True,
        },
        "selection_rule": (
            "Exclude every identity involved in an exact cross-patient duplicate; "
            "for exact duplicates within one patient, retain the first path in "
            "natural order."
        ),
        "flow": {
            "source_images": len(analytic),
            "excluded_images": len(excluded),
            "analytic_images": len(included),
            "source_patients": int(analytic["patient_id"].nunique()),
            "excluded_patients": len(AMBIGUOUS_PATIENTS),
            "analytic_patients": int(included["patient_id"].nunique()),
        },
        "source_class_counts": {
            "images": {str(k): v for k, v in class_counts(analytic, "images").items()},
            "patients": {
                str(k): v for k, v in class_counts(analytic, "patients").items()
            },
        },
        "analytic_class_counts": {
            "images": {str(k): v for k, v in class_counts(included, "images").items()},
            "patients": {
                str(k): v for k, v in class_counts(included, "patients").items()
            },
        },
        "exclusion_reason_counts": {
            str(key): int(value)
            for key, value in excluded["analytic_exclusion_reason"]
            .value_counts()
            .items()
        },
        "duplicate_groups": duplicate_groups(analytic),
        "images_per_patient": distribution_statistics(distribution),
        "outputs": {
            "analytic_cohort": relative_to_project(output_path),
            "analytic_cohort_sha256": sha256(output_path),
            "exclusions": relative_to_project(exclusions_path),
            "exclusions_sha256": sha256(exclusions_path),
            "patient_image_distribution": relative_to_project(distribution_path),
            "patient_image_distribution_sha256": sha256(distribution_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    images_root = args.images_root.resolve()
    output_path = args.output.resolve()
    audit_dir = args.audit_dir.resolve()
    exclusions_path = audit_dir / "exclusions.csv"
    distribution_path = audit_dir / "patient_image_distribution.csv"
    summary_path = audit_dir / "summary.json"
    ensure_writable_targets(
        [output_path, exclusions_path, distribution_path, summary_path],
        overwrite=args.overwrite,
    )

    source = load_source_cohort(manifest_path)
    hashed = hash_source_images(source, images_root)
    analytic = apply_integrity_policy(hashed)
    validate_protocol_cohort(analytic)
    write_outputs(analytic, manifest_path, output_path, audit_dir)

    included = analytic.loc[analytic["analytic_include"]]
    print(f"Capa analítica guardada: {output_path}")
    print(f"Auditoría guardada: {audit_dir}")
    print(
        "Cohorte confirmada: "
        f"{included['patient_id'].nunique()} pacientes, {len(included)} imágenes únicas."
    )


if __name__ == "__main__":
    main()
