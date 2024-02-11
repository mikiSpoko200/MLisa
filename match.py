from utils import get_patches, k_closest, histogram
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple

import numpy as np
from config import default_config

from PIL.Image import Image


def match1(
        image: Image,
        palette: np.ndarray,
        neigh: KNeighborsClassifier | None = None
) -> Tuple[np.ndarray, int]:
    """For each patch from `image` find the closest patch from palette and arrange distances into histogram."""

    # TODO: this is duplicated in palette
    image_array = np.asarray(image, dtype='B').reshape(image.height, image.width, len(image.getbands()))

    patches = get_patches(image_array, default_config.global_palette.coverage)

    _, neighbors = k_closest(patches, palette, 1, neigh)
    return histogram(neighbors, palette.shape[0]), len(patches)


def match2(
        image: Image,
        palette: np.ndarray,
        neigh: KNeighborsClassifier | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    # TODO: this is duplicated in palette
    image_array = np.asarray(image, dtype='B').reshape(image.height, image.width, len(image.getbands()))

    patches = get_patches(image_array, default_config.local_palette.coverage)
    return k_closest(patches, palette, default_config.local_palette.k_neigh, neigh)
