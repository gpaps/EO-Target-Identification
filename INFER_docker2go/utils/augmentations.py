import cv2
import random
import numpy as np
from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms.transform import Transform
from detectron2.data.transforms.transform import NoOpTransform


class GaussianBlurTransform(Transform):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=self.sigma, sigmaY=self.sigma)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords  # No change to coords


class GaussianBlurAll(Augmentation):
    def __init__(self, sigma_range=(1.0, 2.5), prob=0.5):
        super().__init__()
        self.sigma_range = sigma_range
        self.prob = prob

    def get_transform(self, image: np.ndarray) -> Transform:
        if random.random() > self.prob:
            return NoOpTransform()  # ✅ Works safely
        sigma = random.uniform(*self.sigma_range)
        return GaussianBlurTransform(sigma)


class DownsampleThenUpsample(Transform):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def apply_image(self, img):
        h, w = img.shape[:2]
        down = cv2.resize(img, (w // self.scale_factor, h // self.scale_factor), interpolation=cv2.INTER_AREA)
        up = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
        return up

    def apply_coords(self, coords):
        return coords  # unchanged


class T_DownsampleAug(Augmentation):
    def __init__(self, scale_factor=2, prob=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        self.prob = prob

    def get_transform(self, image):
        if random.random() > self.prob:
            return NoOpTransform()
        return DownsampleThenUpsample(self.scale_factor)
