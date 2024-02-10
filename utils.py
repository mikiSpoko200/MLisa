import enum

import bitmath
import matplotlib as plt
import numpy as np

from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.neighbors import KNeighborsClassifier

import config


class ClassificationTarget(enum.Enum):
    """Enumeration representing the target features for classification."""
    ARTIST = enum.auto()
    GENRE = enum.auto()
    STYLE = enum.auto()


# TODO: move this to "*-palette" configurations
PATCH_SIZE: int = 16  # patches PATCH_SIZE x PATCH_SIZE

BASIC_COLORS = np.array(
    [
        [255, 0, 0],
        [255, 128, 0],
        [255, 255, 0],
        [128, 255, 0],
        [0, 255, 0],
        [0, 255, 128],
        [0, 255, 255],
        [0, 128, 255],
        [0, 0, 255],
        [128, 0, 255],
        [255, 0, 255],
        [255, 0, 128],
    ]
)


def read_image(path):
    image = Image.open(path)
    return image


def get_patches(image: np.ndarray, config: config.GlobalPaletteConfig, max_patch_count: int | float):
    height, width = image.shape[0], image.shape[1]
    assert height >= config.patch_size and width >= config.patch_size

    # FIXME: should this use min?
    sample_count = int(
        min(config.coverage * (height - config.patch_size + 1) * (width - config.patch_size + 1), max_patch_count)
    ) if type(max_patch_count) is int else max_patch_count
    if config.random:
        patches = extract_patches_2d(image, (config.patch_size, config.patch_size), max_patches=sample_count).reshape(
            (-1, config.patch_size * config.patch_size * 3))
    else:
        raise NotImplementedError("Efficient non-random sampling is not implemented")
        # TODO: move strides to config
        # calculate strides
        stride_y, stride_x = 1, 1
        curr_coverage = 1.0
        iter = 0
        while curr_coverage > config.coverage:
            curr_coverage = 1.0 / (stride_x * stride_y)
            if iter % 2 == 0:
                stride_y += 1
            else:
                stride_x += 1

        # sliding window
        patches = []
        for upper_left_y in range(0, height - config.patch_size + 1, stride_y):
            for upper_left_x in range(0, width - config.patch_size + 1, stride_x):
                patch = image[
                        upper_left_y: upper_left_y + config.patch_size,
                        upper_left_x: upper_left_x + config.patch_size,
                        :,
                        ].reshape((config.patch_size * config.patch_size * 3))
                patches.append(patch)
    return np.array(patches)


def k_closest(patches: np.ndarray, palette: np.ndarray, k: int, neigh: KNeighborsClassifier | None = None):
    # TODO: run with n_jobs? - to test
    if neigh is not None:
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(palette, np.arange(palette.shape[0]))
    closest = neigh.kneighbors(patches)

    return closest


def histogram(neighbors: np.ndarray, palette_size: int) -> np.ndarray:
    neighbors = neighbors.flatten()
    hist = np.zeros((palette_size,))
    hist[neighbors] += 1
    # _, histogram = np.unique(neighbors, return_counts=True)
    # histogram = histogram.astype("float64")
    # histogram /= patches_num

    # return histogram
    return hist


def plot_image(x, size):
    plt.figure(figsize=(1.5, 1.5))
    plt.imshow(x.reshape(size, size, 3))
    plt.show()
    plt.close()
