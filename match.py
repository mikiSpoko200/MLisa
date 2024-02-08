from utils import *
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple


def match1(image: np.ndarray, palette: np.ndarray, coverage: float, random: bool, neigh: KNeighborsClassifier | None = None) -> Tuple[np.ndarray, int]:
    """For each patch from `image` find the closest patch from palette and arrange distances into histogram."""
    patches = get_patches(image, coverage, random)
    _, neighbors = k_closest(patches, palette, 1, neigh)

    return histogram(neighbors, patches.shape[0]), len(patches)


def match2(image: np.ndarray, palette: np.ndarray, coverage: float, random: bool, k: int, neigh: KNeighborsClassifier | None = None) -> np.ndarray:
    patches = get_patches(image, coverage, random)

    return k_closest(patches, palette, k, neigh)
