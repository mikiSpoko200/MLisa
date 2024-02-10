import os
from collections.abc import Iterable
from typing import Iterator
from config import Config

import PIL
from PIL.Image import Image
import pandas as pd

from utils import ClassificationTarget


class ImageIterator(Iterable[list[Image]]):
    def __init__(self, cls: str, feature_file_paths: list[str], config: Config):
        self.cls = cls
        self._feature_file_paths = feature_file_paths
        self._dataset_path = config.dataset_path
        self._batch_size: int = config.batch_size.to_Byte()
    
    def __iter__(self) -> Iterator[list[Image]]:
        accumulated_images: list[Image] = list()
        total_pixel_data_size = 0
        for file in self._feature_file_paths:
            image_path = os.path.join(self._dataset_path, file)

            try:
                with PIL.Image.open(image_path) as img:
                    pixel_data_size = img.size[0] * img.size[1] * len(img.getbands())

                    if total_pixel_data_size + pixel_data_size <= self._batch_size:
                        total_pixel_data_size += pixel_data_size
                    else:
                        yield accumulated_images
                        total_pixel_data_size = pixel_data_size
                        accumulated_images = list()
                    img.load()
                    accumulated_images.append(img)
            except FileNotFoundError:
                pass
        yield accumulated_images


class BatchLoader(Iterable[ImageIterator]):
    """Iterable that produces iterators which themselves produce per class image batches of specified total size."""
    
    _index: dict[ClassificationTarget, dict[str, list[str]]] = dict()

    def __init__(self, target: ClassificationTarget, config: Config):
        if len(BatchLoader._index) == 0:
            BatchLoader._index = BatchLoader._create_index(config)

        self.target = target
        self.config = config

    def __iter__(self) -> Iterator[ImageIterator]:
        """Returns iterator of iterators that produce per class image batches."""
        for cls, feature_paths in BatchLoader._index[self.target].items():
            yield ImageIterator(cls, feature_paths, self.config)
    
    @staticmethod
    def _cls_encoding(target: ClassificationTarget, config: Config) -> dict[int, str]:
        """Convert class indices from `*_class.txt` files to their string counterparts."""
        with open(os.path.join(config.dataset_labels_path, f"{target.name.lower()}_class.txt"), "r") as f:
            return {int(num): cls.strip() for (num, cls) in map(lambda line: line.split(" "), f.readlines())}

    @staticmethod
    def _create_index(config: Config) -> dict[ClassificationTarget, dict[str, list[str]]]:
        """Create an index of image disc locations for each classification target."""
        index = dict()
        for target in ClassificationTarget:
            cls_encoding = BatchLoader._cls_encoding(target, config)
            entries = pd.read_csv(os.path.join(config.dataset_labels_path, f"{target.name.lower()}_train.csv"), names=["path", "encoded_cls"])
            # Convert class indices to their string counterparts
            entries["encoded_cls"] = entries["encoded_cls"].apply(lambda x: cls_encoding[x])
            # Convert to dictionary of class to path lists (for that class)
            index[target] = entries.groupby("encoded_cls")["path"].apply(list).to_dict()
        return index


def sample(loader: BatchLoader):
    for feature_batch_iterator in loader:
        print(f'Class: {feature_batch_iterator.cls.replace("_", " ")}')
        for index, batch in enumerate(feature_batch_iterator):
            print(f"  Batch {index:>3}: {sum(map(lambda x: len(x.tobytes()), batch)) / (1024 * 1024):>7.3f} MB")
