import numpy as np

import config
from config import default_config
if config.PROFILE:  # Emulate conditional compilation
    def tqdm(*args, **_): return args[0]
else:
    from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
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


MAX_PATCHES_TOTAL_SIZE: int = 1 * 1024 * 1024 * 1024


def generate_palette(images: list[Image], verbose: bool = False, whitening: bool = False):
    # Preallocate memory for all patches

    def image_byte_size(image: Image) -> int:
        return image.height * image.width * len(image.getbands())

    patch_memory_size = min(MAX_PATCHES_TOTAL_SIZE, sum(map(image_byte_size, images)))
    patch_count = patch_memory_size // (default_config.patch_size * default_config.patch_size * 3)

    patches = np.zeros((patch_count, default_config.patch_size * default_config.patch_size * 3))
                      # TODO: does this copy here?
    image_generator = (np.asarray(image, dtype='B').reshape(image.height, image.width, len(image.getbands()))
                       for image in images)
    # TODO: fragmentation?
    offset = 0
    for image in tqdm(image_generator, desc="patches"):
        if patch_count <= 0:
            break
        local_patches = utils.get_patches(image, patch_count)
        local_patch_count = len(local_patches)
        patches[offset: offset + local_patch_count] = local_patches
        del local_patches
        offset += local_patch_count
        patch_count -= local_patch_count

    if whitening:
        patches = whiten(patches)

    assert default_config.parent is not None

    kmeans = (
        MiniBatchKMeans(
            n_clusters=default_config.batching_k_means.number_of_clusters,
            random_state=default_config.parent.random_seed,
            verbose=verbose,
            n_init=1,
            max_iter=default_config.batching_k_means.max_iterations,
            batch_size=default_config.batching_k_means.batch_size)
        .fit(patches))
    # return kmeans.labels_, kmeans.cluster_centers_
    return kmeans.cluster_centers_


def merge_palettes(palettes: list[np.ndarray], verbose: bool = False, whitening: bool = False):
    patches_matrix = np.vstack(palettes)

    if whitening:
        patches_matrix = whiten(patches_matrix)

    assert default_config.parent is not None and default_config.parent.parent is not None

    kmeans = (
        MiniBatchKMeans(
            n_clusters=default_config.number_of_clusters,
            random_state=default_config.parent.parent.random_seed,
            verbose=verbose,
            n_init=1,
            max_iter=default_config.max_iterations,
            batch_size=default_config.batch_size
        ).fit(patches_matrix))

    # return kmeans.labels_, kmeans.cluster_centers_
    return kmeans.cluster_centers_
