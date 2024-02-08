import argparse
import json
import itertools

# from bitmath.integrations import BitmathType
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

import loader
import match
import palette
import utils

from config import Config,

DATASET_PATH = "./wikiart"


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

    # loader.sample(loader.BatchLoader(utils.ClassificationTarget.GENRE, config))

    batch_loader = loader.BatchLoader(utils.ClassificationTarget.GENRE, config)

    features_iterators = [feature for feature in batch_loader]

    # MATCH1
    # training
    def feature_batch(features_iterator):
        samples = list()
        samples.clear()
        for features in itertools.zip_longest(*features_iterator):
            samples.clear()
            samples.extend((image for image in image_batch if image_batch is not None) for image_batch in features)
            yield samples

    palettes = [palette.generate_palette(feature_batch, config.global_palette.size) for feature_batch in
                feature_batch(features_iterators)]
    global_palette = palette.merge_palettes(palettes, config.global_palette.size)

    neigh = KNeighborsClassifier(n_neighbors=1).fit(global_palette)

    per_label_histograms = dict()
    for feature_batch_iterator in loader.BatchLoader(utils.ClassificationTarget.GENRE, config):
        avg_histogram = np.zeros((config.global_palette.size,))
        total_patch_count = 0

        # Generate averaged class histogram
        for image_batch in feature_batch_iterator:
            for image in image_batch:
                (histogram, patches_count) = match.match1(
                    np.array(image),
                    global_palette,
                    config.global_palette.coverage,
                    config.global_palette.random,
                    neigh
                )
                total_patch_count += patches_count
                avg_histogram += histogram
        per_label_histograms[feature_batch_iterator.cls] = avg_histogram / total_patch_count

    # MATCH2
    # training
    for feature_batch_iterator in batch_loader:
        palettes_per_label = dict()
        label = feature_batch_iterator.cls
        print(f"Class: {feature_batch_iterator.cls.replace('_', ' ')}")
        palettes = []
        for index, batch in enumerate(feature_batch_iterator):
            print(f"  Batch {index:>3}: {sum(map(len, batch)) / (1024 * 1024):>7.3f} MB")
            palettes.append(palette.generate_palette(batch, 200))
        palettes_per_label[label] = palette.merge_palettes(palettes, config.global_palette.batching_k_means)

Class = str

def predict(image: bytes, global_palette: np.ndarray, class_histograms: dict[Class, np.ndarray], config: GlobalPaletteConfig) -> Class:
    (histogram, patches_count) = match.match1(
        np.array(image),
        global_palette,
        config.global_palette.coverage,
        config.global_palette.random,
        neigh
    )
    

if __name__ == "__main__":
    main()
