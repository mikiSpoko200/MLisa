import argparse
import json
import os

import PIL
import bitmath
from PIL.Image import Image

import config

from utils import ClassificationTarget
from loader import BatchLoader
from config import Config
import pandas as pd


# Emulate conditional compilation
if config.PROFILE:
    def tqdm(*args, **_): return args[0]
else:
    from tqdm import tqdm


def compression_ratios(target: ClassificationTarget, config: Config):
    _ = BatchLoader(target, config)
    compression_data = list()

    for cls, feature_paths in list(BatchLoader._index[target].items())[5:10]:
        for file in tqdm(feature_paths, desc=cls):
            image_path = os.path.join(config.dataset_path, file)

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


if __name__ == '__main__':
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
    compression_ratios(ClassificationTarget.ARTIST, config)