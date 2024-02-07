import os
from collections.abc import Iterable
from typing import Iterator

from PIL import Image
import pandas as pd

from utils import ClassificationTarget

DATASET_PATH = "../wikiart"
DATASET_LABELS_PATH = "./"


class ImageIterator(Iterable[list[bytes]]):
    def __init__(self, cls: str, feature_file_paths: list[str], dataset_path: str, batch_size: int):
        self.cls = cls
        self._feature_file_paths = feature_file_paths
        self._dataset_path = dataset_path
        self._batch_size = batch_size
    
    def __iter__(self) -> Iterator[list[bytes]]:
        accumulated_images = list()
        total_pixel_data_size = 0
        for file in self._feature_file_paths:
            image_path = os.path.join(self._dataset_path, file)

            with Image.open(image_path) as img:
                pixel_data_size = img.size[0] * img.size[1] * len(img.getbands())

                if total_pixel_data_size + pixel_data_size <= self._batch_size:
                    total_pixel_data_size += pixel_data_size
                else:
                    yield accumulated_images
                    total_pixel_data_size = pixel_data_size
                    accumulated_images = list()

                accumulated_images.append(img.tobytes())
        yield accumulated_images


class BatchLoader(Iterable[ImageIterator]):
    """Iterable that produces iterators which themselves produce per class image batches of specified total size."""
    
    _index: dict[ClassificationTarget, dict[str, list[str]]] = dict()

    def __init__(self, target: ClassificationTarget, dataset_path: str = "./wikiart", batch_size = 256 * 1024 * 1024):
        if len(BatchLoader._index) == 0 :
            BatchLoader._index = BatchLoader._create_index()

        self.target = target
        self._dataset_path = dataset_path
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[ImageIterator]:
        """Returns iterator of iterators that produce per class image batches."""
        for cls, feature_paths in BatchLoader._index[self.target].items():
            yield ImageIterator(cls, feature_paths, self._dataset_path, self._batch_size)
    
    @staticmethod
    def _cls_encoding(target: ClassificationTarget, csv_path: str = DATASET_LABELS_PATH) -> dict[int, str]:
        """Convert class indices from `*_class.txt` files to their string counterparts."""
        with open(os.path.join(csv_path, f"{target.name.lower()}_class.txt"), "r") as f:
            return { int(num): cls.strip() for (num, cls) in map(lambda line: line.split(" "), f.readlines()) }

    @staticmethod
    def _create_index(csv_path: str = DATASET_LABELS_PATH) -> dict[ClassificationTarget, dict[str, list[str]]]:
        """Create an index of image disc locations for each classification target."""
        index = dict()
        for target in ClassificationTarget:
            cls_encoding = BatchLoader._cls_encoding(target, csv_path)
            entires = pd.read_csv(os.path.join(csv_path, f"{target.name.lower()}_train.csv"), names=["path", "encoded_cls"])
            # Convert class indices to their string counterparts
            entires["encoded_cls"] = entires["encoded_cls"].apply(lambda x: cls_encoding[x])
            # Convert to dictionary of class to path lists (for that class)
            index[target] = entires.groupby("encoded_cls")["path"].apply(list).to_dict()
        return index


if __name__ == "__main__":
    for feature_batch_iterator in BatchLoader(ClassificationTarget.ARTIST, "../wikiart"):
        print(f"Class: {feature_batch_iterator.cls.replace("_", " ")}")
        for index, batch in enumerate(feature_batch_iterator):
            print(f"  Batch {index:>3}: {sum(map(len, batch)) / (1024 * 1024):>7.3f} MB")
