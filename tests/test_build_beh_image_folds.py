from __future__ import annotations

import unittest

import pandas as pd

from src.data.build_beh_image_folds import (
    DEFAULT_MANIFEST,
    assign_image_folds,
    load_patient_fold_reference,
    load_eligible_images,
)
from src.settings import BEH_PATIENT_FOLDS_CSV


class ImageLevelFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.eligible = load_eligible_images(DEFAULT_MANIFEST)

    def test_assignment_is_deterministic(self) -> None:
        first = assign_image_folds(self.eligible)
        second = assign_image_folds(self.eligible)
        pd.testing.assert_frame_equal(first, second)

    def test_every_image_receives_one_fold(self) -> None:
        assigned = assign_image_folds(self.eligible)
        self.assertEqual(len(assigned), 1_388)
        self.assertEqual(assigned["relative_path"].nunique(), 1_388)
        self.assertEqual(set(assigned["cv_fold"]), {1, 2, 3, 4, 5})

    def test_patient_reference_matches_manifest(self) -> None:
        patient_reference = load_patient_fold_reference(
            BEH_PATIENT_FOLDS_CSV, self.eligible
        )
        self.assertEqual(len(patient_reference), 1_388)
        self.assertEqual(patient_reference["patient_id"].nunique(), 95)
        self.assertEqual(set(patient_reference["patient_level_fold"]), {1, 2, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()
