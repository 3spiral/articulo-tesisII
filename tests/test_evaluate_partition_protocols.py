import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.evaluation.evaluate_partition_protocols import (
    BOOTSTRAP_SEED,
    aggregate_by_patient,
    build_delta_summary,
    build_paired_patient_table,
    calculate_metrics,
    compute_metric_tables,
    delta_by_repetition,
    paired_stratified_bootstrap,
    summarize_fold_diagnostics,
    summarize_metrics,
    validate_evaluation_configuration,
    validate_run_metadata,
)
from src.training.train_beh_crossval import (
    ANALYTIC_COHORT_CSV,
    ARCHITECTURES,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_WEIGHT_DECAY,
    FINAL_BATCH_SIZES,
    PARTITION_PARAMETERS_JSON,
    PATIENT_FOLDS_CSV,
    PROTOCOLS,
    SEED_SCHEDULE_CSV,
    WEIGHT_NAMES,
    build_run_frames,
    load_fold_assignments,
    sha256,
)


def synthetic_complete_predictions() -> pd.DataFrame:
    frames = []
    for architecture_index, architecture in enumerate(ARCHITECTURES):
        for protocol in PROTOCOLS:
            assignments = load_fold_assignments(protocol)
            for repetition in range(1, 6):
                frame = assignments.loc[
                    assignments["repetition"].eq(repetition),
                    [
                        "relative_path",
                        "patient_id",
                        "target_binary",
                        "cv_fold",
                        "root_seed",
                        "training_seed",
                    ],
                ].copy()
                path_rank = pd.factorize(frame["relative_path"], sort=True)[0]
                noise = ((path_rank * 37 + repetition * 13) % 101) / 100.0
                protocol_signal = 0.34 if protocol == "image_level" else 0.24
                probability = (
                    0.08
                    + protocol_signal * frame["target_binary"].to_numpy()
                    + 0.54 * noise
                    + architecture_index * 0.01
                )
                frame["probability"] = np.clip(probability, 0.001, 0.999)
                frame = frame.rename(columns={"cv_fold": "fold"})
                frame.insert(0, "repetition", repetition)
                frame.insert(0, "protocol", protocol)
                frame.insert(0, "architecture", architecture)
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class CorrectedEvaluationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictions = synthetic_complete_predictions()
        (
            cls.patient_metrics,
            cls.image_metrics,
            cls.fold_metrics,
            cls.patient_predictions,
        ) = compute_metric_tables(cls.predictions)

    def test_metrics_follow_fixed_threshold(self) -> None:
        frame = pd.DataFrame(
            {
                "target_binary": [0, 0, 1, 1],
                "probability": [0.1, 0.4, 0.6, 0.9],
            }
        )
        metrics = calculate_metrics(frame)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(
            (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]),
            (2, 0, 0, 2),
        )

    def test_patient_aggregation_uses_mean_probability(self) -> None:
        frame = pd.DataFrame(
            {
                "patient_id": ["a", "a", "b"],
                "target_binary": [0, 0, 1],
                "probability": [0.2, 0.4, 0.8],
            }
        )
        aggregated = aggregate_by_patient(frame).set_index("patient_id")
        self.assertAlmostEqual(aggregated.loc["a", "probability"], 0.3)

    def test_paired_table_requires_same_patients(self) -> None:
        image = pd.DataFrame(
            {
                "patient_id": ["a", "b"],
                "target_binary": [0, 1],
                "probability": [0.2, 0.8],
            }
        )
        patient = image.assign(probability=[0.3, 0.7])
        paired = build_paired_patient_table(image, patient)
        self.assertEqual(set(paired["patient_id"]), {"a", "b"})

    def test_complete_tables_have_prespecified_dimensions(self) -> None:
        self.assertEqual(len(self.predictions), 27_420)
        self.assertEqual(len(self.patient_metrics), 20)
        self.assertEqual(len(self.image_metrics), 20)
        self.assertEqual(len(self.fold_metrics), 100)
        self.assertEqual(len(self.patient_predictions), 1_860)
        self.assertNotIn("f1", self.patient_metrics.columns)

    def test_deltas_are_paired_by_architecture_and_repetition(self) -> None:
        deltas = delta_by_repetition(self.patient_metrics, self.image_metrics)
        self.assertEqual(len(deltas), 20)
        self.assertEqual(
            set(deltas["analysis_level"]), {"patient", "image_sensitivity"}
        )
        self.assertEqual(
            deltas.groupby(["architecture", "analysis_level"])["repetition"]
            .nunique()
            .min(),
            5,
        )

    def test_bootstrap_is_reproducible_and_shared(self) -> None:
        first = paired_stratified_bootstrap(
            self.predictions,
            self.patient_predictions,
            iterations=20,
            seed=BOOTSTRAP_SEED,
        )
        second = paired_stratified_bootstrap(
            self.predictions,
            self.patient_predictions,
            iterations=20,
            seed=BOOTSTRAP_SEED,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue((first["ci_lower_2_5"] <= first["ci_upper_97_5"]).all())

    def test_summaries_cover_required_metrics(self) -> None:
        deltas = delta_by_repetition(self.patient_metrics, self.image_metrics)
        bootstrap = paired_stratified_bootstrap(
            self.predictions,
            self.patient_predictions,
            iterations=5,
            seed=BOOTSTRAP_SEED,
        )
        self.assertEqual(len(summarize_metrics(self.patient_metrics)), 16)
        self.assertEqual(len(summarize_fold_diagnostics(self.fold_metrics)), 4)
        self.assertEqual(len(build_delta_summary(deltas, bootstrap)), 4)

    def test_final_evaluation_rejects_nonstandard_bootstrap(self) -> None:
        validate_evaluation_configuration(10_000, BOOTSTRAP_SEED)
        with self.assertRaises(ValueError):
            validate_evaluation_configuration(1_000, BOOTSTRAP_SEED)

    def test_run_metadata_rejects_pilots_and_wrong_seeds(self) -> None:
        _, validation = build_run_frames("patient_level", 1, 1)
        identity = validation[
            ["root_seed", "partition_seed", "training_seed"]
        ].drop_duplicates().iloc[0]
        with TemporaryDirectory(dir=Path.cwd(), prefix=".evaluation_test_") as temporary:
            predictions_path = Path(temporary) / "predictions.csv"
            predictions_path.write_text("placeholder\n", encoding="utf-8")
            metadata = {
                "run_status": "complete",
                "architecture": "efficientnet_b0",
                "protocol": "patient_level",
                "repetition": 1,
                "fold": 1,
                "root_seed": int(identity["root_seed"]),
                "partition_seed": int(identity["partition_seed"]),
                "training_seed": int(identity["training_seed"]),
                "pilot": False,
                "epochs": DEFAULT_EPOCHS,
                "batch_size": FINAL_BATCH_SIZES["efficientnet_b0"],
                "learning_rate": DEFAULT_LEARNING_RATE,
                "weight_decay": DEFAULT_WEIGHT_DECAY,
                "optimizer": "AdamW",
                "loss": "BCEWithLogitsLoss_unweighted",
                "pretrained_weights": WEIGHT_NAMES["efficientnet_b0"],
                "all_layers_trainable": True,
                "precision": "float32",
                "mixed_precision": False,
                "gradient_accumulation_steps": 1,
                "scheduler": None,
                "early_stopping": False,
                "checkpoint_selection": "fixed_final_epoch",
                "num_workers": DEFAULT_NUM_WORKERS,
                "analytic_cohort_sha256": sha256(ANALYTIC_COHORT_CSV),
                "fold_source_sha256": sha256(PATIENT_FOLDS_CSV),
                "seed_schedule_sha256": sha256(SEED_SCHEDULE_CSV),
                "partition_parameters_sha256": sha256(PARTITION_PARAMETERS_JSON),
                "predictions_sha256": sha256(predictions_path),
            }
            arguments = {
                "architecture": "efficientnet_b0",
                "protocol": "patient_level",
                "repetition": 1,
                "fold": 1,
                "predictions_path": predictions_path,
                "root_seed": int(identity["root_seed"]),
                "partition_seed": int(identity["partition_seed"]),
                "training_seed": int(identity["training_seed"]),
            }
            validate_run_metadata(metadata, **arguments)
            with self.assertRaises(ValueError):
                validate_run_metadata({**metadata, "pilot": True}, **arguments)
            with self.assertRaises(ValueError):
                validate_run_metadata(
                    {**metadata, "training_seed": int(identity["training_seed"]) + 1},
                    **arguments,
                )


if __name__ == "__main__":
    unittest.main()
