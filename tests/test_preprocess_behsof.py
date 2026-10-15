from __future__ import annotations

import unittest

import torch
from PIL import Image

from src.data.preprocess_behsof import (
    EXPECTED_SOURCE_SIZE,
    OUTPUT_SIZE,
    build_train_transform,
    build_validation_transform,
    prepare_beh_image,
)


def make_synthetic_image() -> Image.Image:
    image = Image.new("RGB", EXPECTED_SOURCE_SIZE)
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return image


class BehPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image = make_synthetic_image()

    def test_base_pipeline_returns_rgb_224(self) -> None:
        result = prepare_beh_image(self.image)
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(result.size, OUTPUT_SIZE)

    def test_unexpected_source_size_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            prepare_beh_image(Image.new("RGB", (100, 100)))

    def test_validation_pipeline_is_deterministic(self) -> None:
        transform = build_validation_transform()
        first = transform(self.image)
        second = transform(self.image)
        self.assertTrue(torch.equal(first, second))

    def test_validation_tensor_contract(self) -> None:
        result = build_validation_transform()(self.image)
        self.assertEqual(tuple(result.shape), (3, 224, 224))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result).all().item())

    def test_training_pipeline_is_reproducible_after_reseeding(self) -> None:
        transform = build_train_transform()
        torch.manual_seed(42)
        first = transform(self.image)
        torch.manual_seed(42)
        second = transform(self.image)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
