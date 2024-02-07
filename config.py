import dataclasses
import bitmath
import json

import typing

@dataclasses.dataclass
class Config:
    dataset_path: str
    dataset_labels_path: str
    batch_size: bitmath.Bitmath

    @classmethod
    def from_json(cls, json: typing.Dict[str, typing.Any]) -> typing.Self:
        return cls(
            dataset_path = json["dataset-path"],
            dataset_labels_path = json["dataset-labels-path"],
            batch_size = bitmath.parse_string(json["loader"]["batch-size"])
        )