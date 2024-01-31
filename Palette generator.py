import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

import utils

random = False
coverage = 10
patch_size = 16


def generate_global_palette(images, number_of_clusters, verbose=False):
    patches = []

    if verbose:
        for i in tqdm(range(len(images)), desc='Getting patches'):
            img_patches = utils.get_patches(images[i], coverage, random)
            patches.extend(img_patches)
    else:
        for image in images:
            img_patches = utils.get_patches(image, coverage, random)
            patches.extend(img_patches)

    patches_matrix = np.array(patches)

    kmeans = (
        MiniBatchKMeans(
            n_clusters=number_of_clusters,
            random_state=0,
            verbose=verbose,
            n_init=1,
            max_iter=200,
            batch_size=10000)
        .fit(patches_matrix))

    return kmeans.labels_, kmeans.cluster_centers_
