import bitmath
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from config import BatchingKMeansConfig, GlobalPaletteConfig
from PIL.Image import Image

import utils


def contrast(image):
    return (image - image.min()) / (image.max() - image.min())


def normalize_patch(patch, eps=10):
    return (patch - patch.mean()) / np.sqrt(patch.var() + eps)


def whiten(x_list):
    x_norm = (x_list - x_list.mean(axis=0)) / x_list.std(axis=0)
    cov = np.cov(x_norm, rowvar=False)
    u, s, v = np.linalg.svd(cov)

    x_zca = u.dot(np.diag(1.0 / np.sqrt(s + 0.1))).dot(u.T).dot(x_norm.T).T
    return x_zca


def generate_palette(images: list[Image], config: GlobalPaletteConfig, verbose: bool = False, whitening: bool = False):
    patches = list()
    print("asdasds")
    image_arrays = [np.asarray(image, dtype='B').reshape(image.height, image.width, len(image.getbands()))
                    for image in images]

    # F1 256 |
    # F2 256 |
    # F3 256 |
    # F4 256 |
    # TOTAL_ITERATION_SIZE = FEATURE_COUNT * PATCH_COUNT
    total_size = 0
    if verbose:
        for i in tqdm(range(len(images)), desc='\rGetting patches'):
            img_patches = utils.get_patches(image_arrays[i], config)
            patches.extend(img_patches)
            total_size += bitmath.Byte(img_patches.size * img_patches.itemsize).to_MiB()
            print(f"Mem usage: {total_size}")
    else:
        for image in image_arrays:
            img_patches = utils.get_patches(image, config)
            patches.extend(img_patches)

    patches_matrix = np.vstack(patches)

    if whitening:
        patches_matrix = whiten(patches_matrix)

    assert config.parent is not None

    kmeans = (
        MiniBatchKMeans(
            n_clusters=config.batching_k_means.number_of_clusters,
            random_state=config.parent.random_seed,
            verbose=verbose,
            n_init=1,
            max_iter=config.batching_k_means.max_iterations,
            batch_size=config.batching_k_means.batch_size)
        .fit(patches_matrix))

    # return kmeans.labels_, kmeans.cluster_centers_
    return kmeans.cluster_centers_


def merge_palettes(palettes: list[np.ndarray], config: BatchingKMeansConfig,
                   verbose: bool = False, whitening: bool = False):
    patches_matrix = np.vstack(palettes)

    if whitening:
        patches_matrix = whiten(patches_matrix)

    assert config.parent is not None and config.parent.parent is not None

    kmeans = (
        MiniBatchKMeans(
            n_clusters=config.number_of_clusters,
            random_state=config.parent.parent.random_seed,
            verbose=verbose,
            n_init=1,
            max_iter=config.max_iterations,
            batch_size=config.batch_size
        ).fit(patches_matrix))

    # return kmeans.labels_, kmeans.cluster_centers_
    return kmeans.cluster_centers_
