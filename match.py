from utils import get_patches, k_closest, histogram
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple

import numpy as np
from config import GlobalPaletteConfig

from PIL.Image import Image


def match1(
        image: Image,
        palette: np.ndarray,
        config: GlobalPaletteConfig,
        neigh: KNeighborsClassifier | None = None
) -> Tuple[np.ndarray, int]:
    """For each patch from `image` find the closest patch from palette and arrange distances into histogram."""

    # TODO: this is duplicated in palette
    image_array = np.asarray(image, dtype='B').reshape(image.height, image.width, len(image.getbands()))

    patches = get_patches(image_array, config, 0.1)
    _, neighbors = k_closest(patches, palette, 1, neigh)
    return histogram(neighbors, patches.shape[0]), len(patches)


def match2(
    image: np.ndarray,
        palette: np.ndarray,
        config: GlobalPaletteConfig,  # TODO: change global palette to local palette (once it's implemented)
        k: int,
        neigh: KNeighborsClassifier | None = None
) -> np.ndarray:
    patches = get_patches(image, config, config.coverage)

    return k_closest(patches, palette, k, neigh)
