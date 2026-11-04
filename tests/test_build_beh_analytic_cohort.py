from __future__ import annotations

import unittest

import pandas as pd

from src.data.build_beh_analytic_cohort import (
    CROSS_PATIENT_REASON,
    WITHIN_PATIENT_REASON,
    apply_integrity_policy,
    make_patient_distribution,
)


def synthetic_hashed_source() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_manifest_row": 2,
                "relative_path": "BEH_A/Image.jpg",
                "patient_id": "BEH_A",
                "filename": "Image.jpg",
                "fibroscan_s": 3,
                "target_binary": 1,
                "sha256": "cross-hash",
            },
            {
                "source_manifest_row": 3,
                "relative_path": "BEH_A/Unique.jpg",
                "patient_id": "BEH_A",
                "filename": "Unique.jpg",
                "fibroscan_s": 3,
                "target_binary": 1,
                "sha256": "unique-a",
            },
            {
                "source_manifest_row": 4,
                "relative_path": "BEH_B/Image.jpg",
                "patient_id": "BEH_B",
                "filename": "Image.jpg",
                "fibroscan_s": 3,
                "target_binary": 1,
                "sha256": "cross-hash",
            },
            {
                "source_manifest_row": 5,
                "relative_path": "BEH_B/Unique.jpg",
                "patient_id": "BEH_B",
                "filename": "Unique.jpg",
                "fibroscan_s": 3,
                "target_binary": 1,
                "sha256": "unique-b",
            },
            {
                "source_manifest_row": 6,
                "relative_path": "BEH_C/Image10.jpg",
                "patient_id": "BEH_C",
                "filename": "Image10.jpg",
                "fibroscan_s": 0,
                "target_binary": 0,
                "sha256": "within-hash",
            },
            {
                "source_manifest_row": 7,
                "relative_path": "BEH_C/Image9.jpg",
                "patient_id": "BEH_C",
                "filename": "Image9.jpg",
                "fibroscan_s": 0,
                "target_binary": 0,
                "sha256": "within-hash",
            },
            {
                "source_manifest_row": 8,
                "relative_path": "BEH_D/Image.jpg",
                "patient_id": "BEH_D",
                "filename": "Image.jpg",
                "fibroscan_s": 1,
                "target_binary": 0,
                "sha256": "unique-d",
            },
        ]
    )


class AnalyticCohortPolicyTests(unittest.TestCase):
    def test_cross_patient_duplicate_excludes_complete_identities(self) -> None:
        analytic = apply_integrity_policy(synthetic_hashed_source())
        ambiguous = analytic.loc[analytic["patient_id"].isin({"BEH_A", "BEH_B"})]
        self.assertFalse(ambiguous["analytic_include"].any())
        self.assertTrue(
            ambiguous["analytic_exclusion_reason"].eq(CROSS_PATIENT_REASON).all()
        )

    def test_natural_order_retains_image9_before_image10(self) -> None:
        analytic = apply_integrity_policy(synthetic_hashed_source())
        image9 = analytic.loc[analytic["relative_path"].eq("BEH_C/Image9.jpg")].iloc[0]
        image10 = analytic.loc[analytic["relative_path"].eq("BEH_C/Image10.jpg")].iloc[0]
        self.assertTrue(bool(image9["analytic_include"]))
        self.assertFalse(bool(image10["analytic_include"]))
        self.assertEqual(image10["analytic_exclusion_reason"], WITHIN_PATIENT_REASON)
        self.assertEqual(image10["retained_relative_path"], "BEH_C/Image9.jpg")

    def test_policy_is_deterministic_under_input_reordering(self) -> None:
        source = synthetic_hashed_source()
        first = apply_integrity_policy(source)
        second = apply_integrity_policy(source.sample(frac=1, random_state=123))
        pd.testing.assert_frame_equal(first, second)

    def test_patient_distribution_uses_only_included_rows(self) -> None:
        analytic = apply_integrity_policy(synthetic_hashed_source())
        distribution = make_patient_distribution(analytic)
        self.assertEqual(set(distribution["patient_id"]), {"BEH_C", "BEH_D"})
        self.assertEqual(distribution["image_count"].sum(), 2)


if __name__ == "__main__":
    unittest.main()
