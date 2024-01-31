from utils import *

def match1(image, palette, coverage, random):
    patches = get_patches(image, coverage, random)
    _, neighbors = k_closest(patches, palette, 1)

    return histogram(neighbors, patches.shape[0])

def match2(image, palette, coverage, random, k):
    patches = get_patches(image, coverage, random)

    return k_closest(patches, palette, k)