import argparse
import json

from bitmath.integrations import BitmathType

import loader
import utils
import palette
import match 
from config import Config

DATASET_PATH = "./wikiart"

def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--config", type=str, default="./config.json", help="Path to the configuration file")
    parser.add_argument("--dataset-path", type=str, help="Override path to the dataset specified in --config")
    parser.add_argument("--dataset-labels-path", type=str, help="Override path to the dataset labels specified in --config")
    parser.add_argument("--batch-size", nargs=2, type=str, help="Override batch size for the loader specified in --config. Format: <value> <unit> (e.g. 256 MiB)")

    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config_json = json.load(config_file)

    config = Config.from_json(config_json)

    loader.sample(loader.BatchLoader(utils.ClassificationTarget.GENRE, config))


if __name__ == "__main__":
    main()
