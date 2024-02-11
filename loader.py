import os
from collections.abc import Iterable
from typing import Iterator

from config import default_config

import PIL
from PIL.Image import Image
import pandas as pd

from utils import ClassificationTarget


class ImageIterator(Iterable[list[Image]]):
    def __init__(self, cls: str, feature_file_paths: list[str]):
        self.cls = cls
        self._feature_file_paths = feature_file_paths
        self._dataset_path = default_config.dataset_path
        self._batch_size: int = default_config.batch_size.to_Byte()

    @staticmethod
    def _insert_text_before_extension(file, text_to_insert):
        # Split the file name and extension
        file_name, file_extension = os.path.splitext(file)

        return f"{file_name}_{text_to_insert}{file_extension}"

    def __iter__(self) -> Iterator[list[Image]]:
        accumulated_images: list[Image] = list()
        total_pixel_data_size = 0
        for file in self._feature_file_paths:
            image_path = os.path.join(self._dataset_path, file)
            for subimage in (map(lambda i: f"{os.path.basename(image_path)}_{i}.jpg",
                                 range(1, 5) if default_config.subrandom else [image_path])):
                try:
                    with PIL.Image.open(subimage) as img:
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

    def __init__(self, target: ClassificationTarget, index=None):
        if len(BatchLoader._index) == 0:
            if index is not None:
                BatchLoader._index = index
            else:
                BatchLoader._index = BatchLoader._create_index()
        self.target = target

    def __iter__(self) -> Iterator[ImageIterator]:
        """Returns iterator of iterators that produce per class image batches."""
        for cls, feature_paths in BatchLoader._index[self.target].items():
            yield ImageIterator(cls, feature_paths)

    @staticmethod
    def _cls_encoding(target: ClassificationTarget) -> dict[int, str]:
        """Convert class indices from `*_class.txt` files to their string counterparts."""
        with open(os.path.join(default_config.dataset_labels_path, f"{target.name.lower()}_class.txt"), "r") as f:
            return {int(num): cls.replace("_", " ").strip() for (num, cls) in
                    map(lambda line: line.split(" "), f.readlines())}

    @staticmethod
    def _create_index() -> dict[ClassificationTarget, dict[str, list[str]]]:
        """Create an index of image disc locations for each classification target."""
        index = dict()
        for target in ClassificationTarget:
            cls_encoding = BatchLoader._cls_encoding(target)
            entries = pd.read_csv(os.path.join(default_config.dataset_labels_path, f"{target.name.lower()}_train.csv"),
                                  names=["path", "encoded_cls"])
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
