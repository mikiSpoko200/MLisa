import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

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


def generate_global_palette(images, number_of_clusters, batch_size=10000, max_iter=200, random=False,
                            coverage=0.1, verbose=False, whitening=False):
    patches = []

    if verbose:
        for i in tqdm(range(len(images)), desc='Getting patches'):
            img_patches = utils.get_patches(images[i], coverage, random)
            patches.extend(img_patches)
    else:
        for image in images:
            img_patches = utils.get_patches(image, coverage, random)
            patches.extend(img_patches)

    patches_matrix = np.vstack(patches)

    if whitening:
        patches_matrix = whiten(patches_matrix)

    kmeans = (
        MiniBatchKMeans(
            n_clusters=number_of_clusters,
            random_state=0,
            verbose=verbose,
            n_init=1,
            max_iter=max_iter,
            batch_size=batch_size)
        .fit(patches_matrix))

    # return kmeans.labels_, kmeans.cluster_centers_
    return kmeans.cluster_centers_
