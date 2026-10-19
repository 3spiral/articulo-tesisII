import unittest

from src.training.train_beh_crossval import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    build_model,
    build_run_frames,
    validate_run_configuration,
)


class TestTrainingProtocol(unittest.TestCase):
    def test_patient_folds_have_no_patient_overlap(self):
        for fold in range(1, 6):
            training, validation = build_run_frames("patient", fold)
            self.assertEqual(len(training) + len(validation), 1_388)
            self.assertTrue(
                set(training["patient_id"]).isdisjoint(validation["patient_id"])
            )

    def test_image_folds_have_patient_overlap_but_not_image_overlap(self):
        for fold in range(1, 6):
            training, validation = build_run_frames("image", fold)
            self.assertEqual(len(training) + len(validation), 1_388)
            self.assertTrue(
                set(training["patient_id"]).intersection(validation["patient_id"])
            )
            self.assertTrue(
                set(training["relative_path"]).isdisjoint(validation["relative_path"])
            )

    def test_classifier_emits_one_logit(self):
        model = build_model(pretrained=False)
        self.assertEqual(model.classifier[1].out_features, 1)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_final_run_rejects_nonstandard_epoch_count(self):
        with self.assertRaises(ValueError):
            validate_run_configuration(
                epochs=1,
                batch_size=DEFAULT_BATCH_SIZE,
                learning_rate=DEFAULT_LEARNING_RATE,
                weight_decay=DEFAULT_WEIGHT_DECAY,
                seed=DEFAULT_SEED,
                pilot=False,
            )

    def test_pilot_accepts_only_one_epoch_with_fixed_configuration(self):
        validate_run_configuration(
            epochs=1,
            batch_size=DEFAULT_BATCH_SIZE,
            learning_rate=DEFAULT_LEARNING_RATE,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            seed=DEFAULT_SEED,
            pilot=True,
        )


if __name__ == "__main__":
    unittest.main()
