from utils import *
from sklearn.neighbors import KNeighborsClassifier


def match1(image: np.ndarray, palette: np.ndarray, coverage: float, random: bool, neigh: KNeighborsClassifier | None = None) -> np.ndarray:
    patches = get_patches(image, coverage, random)
    _, neighbors = k_closest(patches, palette, 1, neigh)

    return histogram(neighbors, patches.shape[0])


def match2(image: np.ndarray, palette: np.ndarray, coverage: float, random: bool, k: int, neigh: KNeighborsClassifier | None = None) -> np.ndarray:
    patches = get_patches(image, coverage, random)

    return k_closest(patches, palette, k, neigh)
