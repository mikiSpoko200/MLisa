import argparse
import json
import itertools

from memory_profiler import profile

import PIL
import numpy as np
from PIL.Image import Image
from sklearn.neighbors import KNeighborsClassifier

import loader
import match
import palette
import utils
import config
from typing import Iterator
from config import Config, GlobalPaletteConfig

# Emulate conditional compilation
if config.PROFILE:
    def tqdm(*args, **_): return args[0]
else:
    from tqdm import tqdm


@profile
def feature_batches(features_iterator: loader.BatchLoader) -> Iterator[list[Image]]:
    samples = list()
    samples.clear()
    for features in tqdm(itertools.zip_longest(*features_iterator), desc="class batches"):
        samples.clear()
        samples.extend(itertools.chain.from_iterable(
            (image for image in image_batch) for image_batch in features if image_batch is not None))
        yield samples


Class = str


def predict(
        image: Image,
        global_palette: np.ndarray,
        class_histograms: dict[Class, np.ndarray],
        config: GlobalPaletteConfig,
        neighbours: KNeighborsClassifier,
) -> Class:
    (histogram, _) = match.match1(image, global_palette, config, neighbours)
    difference = dict()
    for cls, cls_histogram in class_histograms.items():
        difference[cls] = abs(cls_histogram - histogram).sum()

    return min(difference, key=difference.get)


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--config", type=str, default="./config.json", help="Path to the configuration file")
    parser.add_argument("--dataset-path", type=str, help="Override path to the dataset specified in --config")
    parser.add_argument("--dataset-labels-path", type=str,
                        help="Override path to the dataset labels specified in --config")
    parser.add_argument("--batch-size", nargs=2, type=str,
                        help="Override batch size for the loader specified in --config. Format: <value> <unit> (e.g. 256 MiB)")

    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config_json = json.load(config_file)

    config = Config.from_json(config_json)

    loader_params = [utils.ClassificationTarget.ARTIST, config]
    batch_loader = loader.BatchLoader(*loader_params)

    palettes = [
        palette.generate_palette(image_batch, config.global_palette, verbose=True) for image_batch in
        tqdm(feature_batches(batch_loader), desc=" features")
    ]
    global_palette = palette.merge_palettes(palettes, config.global_palette.batching_k_means)

    neighbours = KNeighborsClassifier(n_neighbors=1).fit(global_palette)

    class_histograms = dict()
    for feature_batch_iterator in loader.BatchLoader(*loader_params):
        avg_histogram = np.zeros((config.global_palette.size,))
        total_patch_count = 0

        # Generate averaged class histogram
        for image_batch in tqdm(feature_batch_iterator, desc=" feature"):
            for image in tqdm(image_batch, desc=" batch"):
                # TODO: image -> np.ndarray should have it's own function,
                #  and in general structure of this code duplicates -- fix it.
                (histogram, patches_count) = match.match1(
                    # TODO: image dimensions are hard coded here
                    image,
                    global_palette,
                    config.global_palette,
                    neighbours
                )
                total_patch_count += patches_count
                avg_histogram += histogram
        class_histograms[feature_batch_iterator.cls] = avg_histogram / total_patch_count

    # NOTE: this is temporary
    with PIL.Image.open(f"{config.dataset_path}/Impressionism/claude-monet_water-lilies-6.jpg") as sample:
        print("prediction: ", predict(sample, global_palette, class_histograms, config.global_palette, neighbours))


if __name__ == "__main__":
    try:
        main()
    finally:
        pass
