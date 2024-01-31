from PIL import Image
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.neighbors import KNeighborsClassifier

PATCH_SIZE = 3  # patches PATCH_SIZE x PATCH_SIZE

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


def get_patches(image: np.ndarray, coverage: float, random: bool):
    height, width = image.shape[0], image.shape[1]
    assert height >= PATCH_SIZE and width >= PATCH_SIZE

    if random:
        patches = extract_patches_2d(image, (PATCH_SIZE, PATCH_SIZE), coverage)
    else:
        # calculate strides
        stride_y, stride_x = 1, 1
        curr_coverage = 1.0
        iter = 0
        while curr_coverage > coverage:
            coverage = 1.0 / (stride_x * stride_y)
            if iter % 2 == 0:
                stride_y += 1
            else:
                stride_x += 1

        # sliding window
        patches = []
        for upper_left_y in range(0, height - PATCH_SIZE + 1, stride_y):
            for upper_left_x in range(0, width - PATCH_SIZE + 1, stride_x):
                patch = image[upper_left_y : upper_left_y + PATCH_SIZE, upper_left_x : upper_left_x + PATCH_SIZE, :]
                patches.append(patch)
    return np.array(patches)


def k_closest(patches: np.ndarray, palette: np.ndarray, k: int):
    palette = palette.reshape((-1, PATCH_SIZE * PATCH_SIZE * 3))
    patches = patches.reshape((-1, PATCH_SIZE * PATCH_SIZE * 3))

    # TODO: test if faster for k=1 (probably not)
    # patches = np.expand_dims(palette, 1)
    # # patches shape: (n_patches,         1, patch_size ** 2 * 3)
    # # palette shape:            (m_patches, patch_size ** 2 * 3)
    # closest = np.sum((palette - patches) ** 2, axis=2).argmin(axis=1).flatten()

    # TODO: run with n_jobs?
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(palette, np.arange(palette.shape[0]))
    closest = neigh.kneighbors(patches) 

    return closest


def histogram(neighbors: np.ndarray, patches_num: int):
    neighbors = neighbors.flatten()
    _, histogram = np.unique(neighbors, return_counts=True)
    histogram = histogram.astype('float64')
    histogram /= patches_num

    return histogram
