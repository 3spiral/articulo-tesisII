from __future__ import annotations

import unittest

import pandas as pd

from src.data.build_beh_partition_folds import (
    DEFAULT_COHORT,
    EXPECTED_IMAGES,
    EXPECTED_PATIENTS,
    IMAGE_PROTOCOL,
    N_REPETITIONS,
    PATIENT_PROTOCOL,
    assign_folds,
    audit_partitions,
    derive_seed_schedule,
    expand_patient_assignments,
    load_analytic_cohort,
)


class PartitionFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cohort = load_analytic_cohort(DEFAULT_COHORT)
        cls.schedule = derive_seed_schedule()
        cls.image_folds, cls.patient_folds = assign_folds(
            cls.cohort, cls.schedule
        )
        cls.exposure, cls.summary = audit_partitions(
            cls.cohort, cls.image_folds, cls.patient_folds
        )

    def test_seed_schedule_is_deterministic_and_domain_separated(self) -> None:
        pd.testing.assert_frame_equal(self.schedule, derive_seed_schedule())
        self.assertEqual(len(self.schedule), 25)
        self.assertEqual(self.schedule["training_seed"].nunique(), 25)
        self.assertEqual(self.schedule["image_partition_seed"].nunique(), 5)
        self.assertEqual(self.schedule["patient_partition_seed"].nunique(), 5)

    def test_image_fold_coverage_is_complete_in_every_repetition(self) -> None:
        coverage = self.image_folds.groupby("repetition")["relative_path"].agg(
            ["count", "nunique"]
        )
        self.assertTrue(coverage["count"].eq(EXPECTED_IMAGES).all())
        self.assertTrue(coverage["nunique"].eq(EXPECTED_IMAGES).all())

    def test_patient_fold_coverage_and_grouping_are_complete(self) -> None:
        coverage = self.patient_folds.groupby("repetition")["patient_id"].agg(
            ["count", "nunique"]
        )
        self.assertTrue(coverage["count"].eq(EXPECTED_PATIENTS).all())
        self.assertTrue(coverage["nunique"].eq(EXPECTED_PATIENTS).all())
        expanded = expand_patient_assignments(self.cohort, self.patient_folds)
        patient_fold_counts = expanded.groupby(
            ["repetition", "patient_id"]
        )["cv_fold"].nunique()
        self.assertTrue(patient_fold_counts.eq(1).all())

    def test_patient_protocol_has_zero_identity_and_sister_overlap(self) -> None:
        patient_summary = self.summary.loc[
            self.summary["protocol"].eq(PATIENT_PROTOCOL)
        ]
        self.assertTrue(patient_summary["overlapping_patients"].eq(0).all())
        self.assertTrue(patient_summary["exposed_validation_images"].eq(0).all())

    def test_both_protocols_have_zero_exact_hash_overlap(self) -> None:
        self.assertEqual(set(self.summary["protocol"]), {IMAGE_PROTOCOL, PATIENT_PROTOCOL})
        self.assertTrue(self.summary["hash_intersection_count"].eq(0).all())
        self.assertFalse(self.exposure["hash_in_training"].any())

    def test_oof_exposure_covers_every_image_once_per_protocol_and_repetition(self) -> None:
        coverage = self.exposure.groupby(
            ["protocol", "repetition"]
        )["relative_path"].agg(["count", "nunique"])
        self.assertEqual(len(coverage), N_REPETITIONS * 2)
        self.assertTrue(coverage["count"].eq(EXPECTED_IMAGES).all())
        self.assertTrue(coverage["nunique"].eq(EXPECTED_IMAGES).all())


if __name__ == "__main__":
    unittest.main()
