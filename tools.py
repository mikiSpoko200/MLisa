import collections
import dataclasses
import itertools
import json
import os
import shutil
import random
from typing import Iterator

import bitmath
import pandas as pd
import PIL
from PIL.Image import Image
import matplotlib.pyplot as plt

import config
import loader
from config import default_config
from utils import ClassificationTarget
from loader import BatchLoader


# Emulate conditional compilation
if config.PROFILE:
    def tqdm(*args, **_):
        return args[0]
else:
    from tqdm import tqdm


def unbiased():
    loader.BatchLoader(ClassificationTarget.ARTIST)
    target_subindex_size = {
        ClassificationTarget.ARTIST: 330,
        ClassificationTarget.STYLE: 280,
        ClassificationTarget.GENRE: 760,
    }

    subindex = dict()
    random.seed(default_config.random_seed)
    for target, cls_paths in loader.BatchLoader._index.items():
        subindex[target.name] = {
            cls: random.sample(paths, target_subindex_size[target]) for cls, paths in cls_paths.items() if
            len(paths) >= target_subindex_size[target]
        }

    def copy_if_not_exists(source, t):
        if not os.path.exists(t):
            shutil.copy(source, t)

    random_path = "./wikiart-subrandom"
    for cls_paths in subindex.values():
        for paths in cls_paths.values():
            for path in paths:
                s = os.path.join(default_config.dataset_path, path)
                t = os.path.join(random_path, path.split("/")[1])
                copy_if_not_exists(s, t)

    with open("subrandom-index.json", "w") as index:
        json.dump(subindex, index)


def compression_ratios(target: ClassificationTarget):
    _ = BatchLoader(target, config)
    compression_data = list()

    for cls, feature_paths in list(BatchLoader._index[target].items())[5:10]:
        for file in tqdm(feature_paths, desc=cls):
            image_path = os.path.join(default_config.dataset_path, file)

            with PIL.Image.open(image_path) as img:
                compression_data.append([
                    image_path,
                    cls,
                    img.size,
                    bitmath.Byte(os.path.getsize(image_path)).to_MiB(),
                    bitmath.Byte(img.size[0] * img.size[1] * len(img.getbands())).to_MiB()
                ])

    df = pd.DataFrame(compression_data, columns=["path", "class", "dims", "compressed-size", "decompressed-size"])
    print((df["decompressed-size"] / df["compressed-size"]).mean())


# All serialize to json


@dataclasses.dataclass
class ImageInfo:
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def bytes_size(self) -> bitmath.Byte:
        raise NotImplementedError()


@dataclasses.dataclass
class ClassInfo:
    paths: list[ImageInfo]
    total_size: int
    paths: str
    resolutions: list[(int, int)]

    def resolutions_hist(self):
        """Return histogram of areas (% 10?) of images for this class. (data for bar plot)"""
        raise NotImplementedError()


@dataclasses.dataclass
class TargetInfo:
    target: ClassificationTarget

    def compression_ratio(self) -> float:
        raise NotImplementedError()

    def class_data(self) -> dict[str, ClassInfo]:
        raise NotImplementedError()

    def class_total_img_size(self) -> bitmath.Byte:
        raise NotImplementedError()


def visualize_dataset(include_test=False):
    index = dict()
    for target in ClassificationTarget:
        cls_encodings = BatchLoader._cls_encoding(target)
        train = pd.read_csv(
            os.path.join(default_config.dataset_labels_path, f"{target.name.lower()}_train.csv"),
            names=["path", "encoded_cls"]
        )
        if include_test:
            test = pd.read_csv(
                os.path.join(default_config.dataset_labels_path, f"{target.name.lower()}_val.csv"),
                names=["path", "encoded_cls"]
            )
            stacked_data = pd.concat([train, test], ignore_index=True)
        else:
            stacked_data = train
        stacked_data["encoded_cls"] = stacked_data["encoded_cls"].apply(lambda x: cls_encodings[x])
        # Convert to dictionary of class to path lists (for that class)
        index[target] = stacked_data.groupby("encoded_cls")["path"].apply(list).to_dict()

        # Plotting
        num_classes = len(index[target])
        fig_height = max(6, num_classes * 0.6)  # Adjust minimum height
        plt.figure(figsize=(10, fig_height))
        plt.bar(index[target].keys(), [len(val) for val in index[target].values()])
        plt.title(f"Data Entries for {target.name}")
        plt.xlabel("Encoded Class")
        plt.ylabel("Number of Data Entries")
        plt.xticks(rotation=45, ha='right')  # Rotate and align ticks to the right
        plt.gca().margins(x=0.01)  # Add small margin to avoid cutoff
        plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin
        # Save plot to file with the specified naming convention

        # Add average line
        average_value = sum(len(val) for val in index[target].values()) / num_classes
        plt.axhline(y=average_value, color='r', linestyle='--', label='Average')
        plt.legend()

        file_name = f"linear-{target.name.lower()}.jpg"
        plt.savefig(file_name)
        plt.close()

        num_classes = len(index[target])
        fig_height = max(6, num_classes * 0.6)  # Adjust minimum height
        plt.figure(figsize=(10, fig_height))
        plt.bar(index[target].keys(), [len(val) for val in index[target].values()])
        plt.title(f"Data Entries for {target.name}")
        plt.xlabel("Encoded Class")
        plt.ylabel("Number of Data Entries")
        plt.xticks(rotation=45, ha='right')  # Rotate and align ticks to the right
        plt.gca().margins(x=0.01)  # Add small margin to avoid cutoff
        plt.subplots_adjust(bottom=0.15)  # Adjust bottom margin
        plt.yscale('log')  # Set y-axis scale to logarithmic

        # Add average line
        average_value = sum(len(val) for val in index[target].values()) / num_classes
        plt.axhline(y=average_value, color='r', linestyle='--', label='Average')
        plt.legend()

        # Save plot to file with the specified naming convention
        file_name = f"log-{target.name.lower()}.jpg"
        plt.savefig(file_name)
        plt.close()  # Close the plot to avoid displaying it


class LoadingStrategy:
    @staticmethod
    def interleaved_accesses(features_iterator: BatchLoader) -> Iterator[list[Image]]:
        samples = list()
        samples.clear()
        for features in tqdm(itertools.zip_longest(*features_iterator), desc="class batches"):
            samples.clear()
            samples.extend(itertools.chain.from_iterable(
                (image for image in image_batch) for image_batch in features if image_batch is not None)
            )
            yield samples

    @staticmethod
    def serial_accessed(features_iterator: BatchLoader) -> Iterator[list[Image]]:
        for class_batches in features_iterator:
            yield from class_batches


def consume(iterator, n=None):
    """Advance the iterator n-steps ahead. If n is None, consume entirely."""
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(itertools.islice(iterator, n, n), None)


if __name__ == '__main__':
    unbiased()
    # visualize_dataset()

    # loader1 = BatchLoader(ClassificationTarget.ARTIST, config)
    # loader2 = BatchLoader(ClassificationTarget.ARTIST, config)
    #
    # consume(LoadingStrategy.serial_accessed(loader1))
    # consume(LoadingStrategy.interleaved_accesses(loader2))
