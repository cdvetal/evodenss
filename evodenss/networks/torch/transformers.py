from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
import logging
import random
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import Compose, CenterCrop, ColorJitter, Normalize, RandomApply, \
    RandomCrop, RandomGrayscale, RandomHorizontalFlip, RandomResizedCrop, Resize, \
    ToTensor

from evodenss.misc.enums import TransformOperation

if TYPE_CHECKING:
    from torch import nn, Tensor


__all__ = ['LegacyTransformer', 'BarlowTwinsTransformer']


logger = logging.getLogger(__name__)


class GaussianBlur(object):

    def __init__(self, p: float) -> None:
        self.p: float = p

    def __call__(self, img: Image) -> Image:
        if random.random() < self.p:
            sigma: float = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p: float):
        self.p: float = p

    def __call__(self, img: Image) -> Image:
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class BaseTransformer(ABC):

    def __init__(self) -> None:
        pass

    def _get_transform_op(self, operation: TransformOperation, params: Dict[str, Any]) -> Any:
        final_params = deepcopy(params)
        if operation == TransformOperation.RANDOM_CROP:
            return RandomCrop(**final_params)
        elif operation == TransformOperation.COLOR_JITTER:
            probability: float = final_params.pop("probability")
            return RandomApply([ColorJitter(**final_params)], p=probability)
        elif operation == TransformOperation.NORMALIZE:
            return Normalize(**final_params)
        elif operation == TransformOperation.HORIZONTAL_FLIPPING:
            final_params['p'] = final_params.pop("probability")
            return RandomHorizontalFlip(**final_params)
        elif operation == TransformOperation.RANDOM_RESIZED_CROP:
            if "scale" in final_params.keys():
                final_params['scale'] = tuple(map(float, final_params['scale'][1:-1].split(',')))
            return RandomResizedCrop(**final_params, interpolation=Image.BICUBIC)
        elif operation == TransformOperation.RANDOM_GRAYSCALE:
            final_params['p'] = final_params.pop("probability")
            return RandomGrayscale(**final_params)
        elif operation == TransformOperation.GAUSSIAN_BLUR:
            final_params['p'] = final_params.pop("probability")
            return GaussianBlur(**final_params)
        elif operation == TransformOperation.SOLARIZE:
            final_params['p'] = final_params.pop("probability")
            return Solarization(**final_params)
        elif operation == TransformOperation.RESIZE:
            return Resize(**final_params)
        elif operation == TransformOperation.CENTER_CROP:
            return CenterCrop(**final_params)
        else:
            raise ValueError(f"Cannot create transformation object from name {operation}")

    def _create_compose_transform(self, transform_details: Dict[str, Any]) -> Compose:
        augmentation_ops: List[nn.Module] = []
        operation: TransformOperation
        for name, params in transform_details.items():
            operation = TransformOperation(name)
            if operation not in [TransformOperation.NORMALIZE, TransformOperation.RESIZE]:
                augmentation_ops.append(self._get_transform_op(operation, params))
        if TransformOperation.RESIZE.value in transform_details.keys():
            augmentation_ops.append(Resize(**transform_details[TransformOperation.RESIZE.value]))
        #augmentation_ops.append(Resize(size=INPUT_DIMENSIONS[-2:]))
        augmentation_ops.append(ToTensor())
        if TransformOperation.NORMALIZE.value in transform_details.keys():
            augmentation_ops.append(Normalize(**transform_details[TransformOperation.NORMALIZE.value]))
        return Compose(augmentation_ops)

    @abstractmethod
    def __call__(self, img: Image) -> Any:
        raise NotImplementedError("__call__ method should not be used by a BaseTransformer object")


class LegacyTransformer(BaseTransformer):

    def __init__(self, augmentation_params: Dict[str, Any]) -> None:
        super().__init__()
        self.transform: Compose = self._create_compose_transform(augmentation_params)

    def __call__(self, img: Image) -> Any:
        if img.mode == 'RGBA':
            return self.transform(img.convert('RGB')) # random grayscale fails when an image has 4 channels
        return self.transform(img)


class BarlowTwinsTransformer(BaseTransformer):

    def __init__(self, augmentation_params: Dict[str, Any]) -> None:
        super().__init__()
        self.transform_a: Compose = self._create_compose_transform(augmentation_params['input_a'])
        self.transform_b: Compose = self._create_compose_transform(augmentation_params['input_b'])

    def __call__(self, img: Image) -> Tuple[Tensor, Tensor]:
        if img.mode == 'RGBA':
            img = img.convert('RGB') # random grayscale fails when an image has 4 channels
        aug_input_a = self.transform_a(img)
        aug_input_b = self.transform_b(img)
        return aug_input_a, aug_input_b
