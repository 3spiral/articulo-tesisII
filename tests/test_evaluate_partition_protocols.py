import unittest
from unittest.mock import patch

import pandas as pd

from src.evaluation.evaluate_partition_protocols import (
    build_paired_patient_table,
    calculate_metrics,
    paired_stratified_bootstrap_delta,
    validate_evaluation_configuration,
    validate_run_metadata,
)
from src.training.train_beh_crossval import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
)


class TestProtocolEvaluation(unittest.TestCase):
    def setUp(self):
        self.image = pd.DataFrame(
            {
                "protocol": ["image"] * 4,
                "patient_id": ["a", "b", "c", "d"],
                "target_binary": [0, 0, 1, 1],
                "image_count": [2, 2, 2, 2],
                "probability": [0.1, 0.2, 0.8, 0.9],
            }
        )
        self.patient = self.image.assign(
            protocol="patient", probability=[0.4, 0.6, 0.4, 0.6]
        )

    def test_metrics_follow_fixed_threshold(self):
        metrics = calculate_metrics(self.image)
        self.assertEqual(metrics["auroc"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (2, 0, 0, 2))

    def test_paired_table_preserves_same_patients(self):
        paired = build_paired_patient_table(self.image, self.patient)
        self.assertEqual(len(paired), 4)
        self.assertEqual(set(paired["patient_id"]), {"a", "b", "c", "d"})

    def test_bootstrap_is_reproducible(self):
        paired = build_paired_patient_table(self.image, self.patient)
        first = paired_stratified_bootstrap_delta(paired, iterations=100, seed=42)
        second = paired_stratified_bootstrap_delta(paired, iterations=100, seed=42)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["delta_auroc_image_minus_patient"], 0.5)

    def test_final_evaluation_rejects_nonstandard_bootstrap(self):
        with self.assertRaises(ValueError):
            validate_evaluation_configuration(iterations=1_000, seed=42)

    @patch(
        "src.evaluation.evaluate_partition_protocols.sha256",
        side_effect=["manifest-hash", "fold-hash"],
    )
    def test_run_metadata_rejects_pilot(self, _mocked_sha256):
        metadata = {
            "run_status": "complete",
            "protocol": "image",
            "fold": 1,
            "pilot": True,
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
            "manifest_sha256": "manifest-hash",
            "fold_source": "metadata/BEH_image_folds.csv",
            "fold_source_sha256": "fold-hash",
        }
        with self.assertRaises(ValueError):
            validate_run_metadata(metadata, "image", 1)


if __name__ == "__main__":
    unittest.main()
