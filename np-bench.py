import argparse
import json

# from bitmath.integrations import BitmathType
import numpy as np

import loader
import utils
from config import Config
import timeit


def bench(config: Config):

    counter = 0
    for feature_batch_iterator in loader.BatchLoader(utils.ClassificationTarget.GENRE, config):
        for batch in feature_batch_iterator:
            for image in batch:
                _array = np.frombuffer(image, dtype='B')
        counter += 1
        if counter > 3:
            break


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

    _ = loader.BatchLoader(utils.ClassificationTarget.GENRE, config)
    timeit.timeit(lambda: bench(config), number=5)

