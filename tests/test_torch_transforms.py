import random
from typing import cast
import unittest

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torchvision.transforms import Compose, ColorJitter, Normalize, \
    RandomGrayscale, RandomApply, RandomHorizontalFlip, RandomResizedCrop , Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode

from evodenss.misc.enums import TransformOperation
from evodenss.networks.transformers import LegacyTransformer, \
    GaussianBlur, Solarization


class Test(unittest.TestCase):

    def _fix_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def test_solarize(self) -> None:
        self._fix_seed(0)
        transformer = LegacyTransformer({TransformOperation.SOLARIZE.value: {'probability': 1.0}})
        to_tensor_transform = Compose([ToTensor()])

        original_img = Image.open("tests/resources/images/original_mnist_seed=0_id=0.png")
        expected_img = Image.open("tests/resources/images/solarized_mnist_seed=0_id=0.png")

        final_tensor = transformer(original_img)
        expected_tensor = cast(Tensor, to_tensor_transform(expected_img))
        self.assertEqual(final_tensor.numpy().tolist(), expected_tensor.numpy().tolist())


    def test_gaussian_blur(self) -> None:
        self._fix_seed(0)
        transformer = LegacyTransformer({TransformOperation.GAUSSIAN_BLUR.value: {'probability': 1.0}})
        to_tensor_transform = Compose([ToTensor()])

        original_img = Image.open("tests/resources/images/original_mnist_seed=0_id=0.png")
        expected_img = Image.open("tests/resources/images/gaussian_blur_mnist_seed=0_id=0.png")

        final_tensor = transformer(original_img)
        expected_tensor = cast(Tensor, to_tensor_transform(expected_img))
        self.assertEqual(final_tensor.numpy().tolist(), expected_tensor.numpy().tolist())


    def test_random_grayscale(self) -> None:
        self._fix_seed(0)
        transformer = LegacyTransformer({TransformOperation.RANDOM_GRAYSCALE.value: {'probability': 1.0}})
        to_tensor_transform = Compose([ToTensor()])

        original_img = Image.open("tests/resources/images/original_imagenet_dog.png")
        expected_img = Image.open("tests/resources/images/random_grayscale_original_imagenet_dog.png")

        final_tensor = transformer(original_img)
        expected_tensor = cast(Tensor, to_tensor_transform(expected_img))
        self.assertEqual(final_tensor.numpy().tolist(), expected_tensor.numpy().tolist())


    def test_random_resized_crop(self) -> None:
        self._fix_seed(0)
        transformer = LegacyTransformer({TransformOperation.RANDOM_RESIZED_CROP.value: {'size': 28}})
        to_tensor_transform = Compose([ToTensor()])

        original_img = Image.open("tests/resources/images/original_mnist_seed=0_id=0.png")
        expected_img = Image.open("tests/resources/images/random_resized_crop_mnist_seed=0_id=0.png")

        final_tensor = transformer(original_img)
        expected_tensor: Tensor = cast(Tensor, to_tensor_transform(expected_img))
        self.assertEqual(final_tensor.numpy().tolist(), expected_tensor.numpy().tolist())


    def test_horizontal_flip(self) -> None:
        self._fix_seed(0)
        transformer = LegacyTransformer({TransformOperation.HORIZONTAL_FLIPPING.value: {'probability': 1.0}})
        to_tensor_transform = Compose([ToTensor()])

        original_img = Image.open("tests/resources/images/original_mnist_seed=0_id=0.png")
        expected_img = Image.open("tests/resources/images/horizontal_flip_mnist_seed=0_id=0.png")

        final_tensor = transformer(original_img)
        expected_tensor = cast(Tensor, to_tensor_transform(expected_img))
        self.assertEqual(final_tensor.numpy().tolist(), expected_tensor.numpy().tolist())


    def test_normalize(self) -> None:
        self._fix_seed(0)
        transformer = LegacyTransformer({
            TransformOperation.NORMALIZE.value:{'mean': [0.485], 'std': [0.09]}
        })
        to_tensor_transform = Compose([ToTensor()])
        original_img = Image.open("tests/resources/images/original_mnist_seed=0_id=0.png")
        expected_img = Image.open("tests/resources/images/normalize_mnist_seed=0_id=0.png")
        final_tensor = torch.round((transformer(original_img).clamp(0, 1) * 255)).int()
        expected_tensor = (cast(Tensor, to_tensor_transform(expected_img)) * 255).int()
        self.assertEqual(final_tensor.numpy().tolist(), expected_tensor.numpy().tolist())


    def test_color_jitter(self)-> None:
        self._fix_seed(0)
        transformer = LegacyTransformer(
            {
                TransformOperation.COLOR_JITTER.value:
                    {
                        'brightness': 0.8,
                        'contrast': 0.8,
                        'saturation': 0.2,
                        'hue': 0.4,
                        'probability': 1.0
                    }
            }
        )
        to_tensor_transform = Compose([ToTensor()])
        original_img = Image.open("tests/resources/images/original_mnist_seed=0_id=0.png")
        expected_img = Image.open("tests/resources/images/color_jitter_mnist_seed=0_id=0.png")

        final_tensor = transformer(original_img)
        expected_tensor = cast(Tensor, to_tensor_transform(expected_img))
        self.assertEqual(final_tensor.numpy().tolist(), expected_tensor.numpy().tolist())


    def test_transform_pipeline_creation(self) -> None:
        transformations = {
            'resize': {'size': 32},
            'normalize': {'mean': [0.485], 'std': [0.229]},
            'random_horizontal_flip': {'probability': 0.0},
            'random_resized_crop': {'size': 28, 'scale': '(0.8, 0.9)'},
            'color_jitter': {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.2, 'hue': 0.1, 'probability': 0.1},
            'random_grayscale': {'probability': 0.1},
            'gaussian_blur': {'probability': 0.1},
            'random_solarize': {'probability': 0.1}
        }
        transformer = LegacyTransformer(transformations)
        expected_transforms = [
            Resize(size=32),
            RandomHorizontalFlip(p=0.0),
            RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.1),
            RandomResizedCrop(size=28, scale=(0.8, 0.9), interpolation=InterpolationMode.BICUBIC),
            RandomGrayscale(p=0.1),
            GaussianBlur(p=0.1),
            Solarization(p=0.1),
            ToTensor(),
            Normalize(mean=[0.485], std=[0.229]),
        ]
        real_transformations = list(map(lambda x: str(x.__dict__), transformer.transform.transforms))
        expected_transformations = list(map(lambda x: str(x.__dict__), expected_transforms))
        real_transformations_cls = list(map(lambda x: x.__class__, transformer.transform.transforms))
        expected_transformations_cls = list(map(lambda x: x.__class__, expected_transforms))

        self.assertCountEqual(real_transformations, expected_transformations)
        self.assertCountEqual(real_transformations_cls, expected_transformations_cls)
        # The last two have always to be ToTensor() and Normalize()
        self.assertEqual(real_transformations[-2:], expected_transformations[-2:])
        self.assertEqual(real_transformations_cls[-2:], expected_transformations_cls[-2:])


if __name__ == '__main__':
    unittest.main()
