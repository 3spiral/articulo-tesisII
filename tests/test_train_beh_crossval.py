import unittest

from src.training.train_beh_crossval import (
    ARCHITECTURES,
    PROTOCOLS,
    build_model,
    build_run_frames,
    load_analytic_cohort,
    load_fold_assignments,
    validate_run_configuration,
)


class CorrectedTrainingContractTests(unittest.TestCase):
    def test_analytic_cohort_is_the_only_training_population(self) -> None:
        cohort = load_analytic_cohort()
        self.assertEqual(len(cohort), 1_371)
        self.assertEqual(cohort["patient_id"].nunique(), 93)
        self.assertEqual(cohort["sha256"].nunique(), 1_371)

    def test_canonical_assignments_cover_five_repetitions(self) -> None:
        for protocol in PROTOCOLS:
            assignments = load_fold_assignments(protocol)
            self.assertEqual(len(assignments), 1_371 * 5)
            self.assertEqual(set(assignments["repetition"]), set(range(1, 6)))
            self.assertEqual(set(assignments["cv_fold"]), set(range(1, 6)))

    def test_patient_protocol_has_no_identity_or_hash_overlap(self) -> None:
        for repetition in range(1, 6):
            for fold in range(1, 6):
                training, validation = build_run_frames(
                    "patient_level", repetition, fold
                )
                self.assertFalse(
                    set(training["patient_id"]).intersection(validation["patient_id"])
                )
                self.assertFalse(
                    set(training["sha256"]).intersection(validation["sha256"])
                )

    def test_image_protocol_has_identity_but_not_hash_overlap(self) -> None:
        for repetition in range(1, 6):
            for fold in range(1, 6):
                training, validation = build_run_frames(
                    "image_level", repetition, fold
                )
                self.assertTrue(
                    set(training["patient_id"]).intersection(validation["patient_id"])
                )
                self.assertFalse(
                    set(training["sha256"]).intersection(validation["sha256"])
                )

    def test_corresponding_protocols_reuse_training_seed(self) -> None:
        for repetition in range(1, 6):
            for fold in range(1, 6):
                _, image_validation = build_run_frames(
                    "image_level", repetition, fold
                )
                _, patient_validation = build_run_frames(
                    "patient_level", repetition, fold
                )
                self.assertEqual(
                    image_validation["training_seed"].unique().tolist(),
                    patient_validation["training_seed"].unique().tolist(),
                )

    def test_both_models_emit_one_logit_and_are_fully_trainable(self) -> None:
        for architecture in ARCHITECTURES:
            model = build_model(architecture, pretrained=False)
            output_features = (
                model.classifier[1].out_features
                if architecture == "efficientnet_b0"
                else model.fc.out_features
            )
            self.assertEqual(output_features, 1)
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_final_configuration_is_frozen(self) -> None:
        for architecture in ARCHITECTURES:
            validate_run_configuration(
                architecture=architecture,
                epochs=20,
                batch_size=32,
                pilot=False,
            )
            with self.assertRaises(ValueError):
                validate_run_configuration(
                    architecture=architecture,
                    epochs=19,
                    batch_size=32,
                    pilot=False,
                )
            with self.assertRaises(ValueError):
                validate_run_configuration(
                    architecture=architecture,
                    epochs=20,
                    batch_size=16,
                    pilot=False,
                )

    def test_resnet_pilot_allows_only_prespecified_memory_steps(self) -> None:
        for batch_size in (32, 16, 8):
            validate_run_configuration(
                architecture="resnet50",
                epochs=1,
                batch_size=batch_size,
                pilot=True,
            )
        with self.assertRaises(ValueError):
            validate_run_configuration(
                architecture="resnet50", epochs=1, batch_size=4, pilot=True
            )
        with self.assertRaises(ValueError):
            validate_run_configuration(
                architecture="efficientnet_b0", epochs=1, batch_size=16, pilot=True
            )


if __name__ == "__main__":
    unittest.main()
