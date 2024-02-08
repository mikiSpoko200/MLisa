import dataclasses
import bitmath
import typing


@dataclasses.dataclass
class BatchingKMeansConfig:
    batch_size: int
    max_iterations: int
    number_of_clusters: int
    parent: "GlobalPaletteConfig | None" = None

    @classmethod
    def from_json(cls, json: typing.Dict[str, typing.Any]) -> typing.Self:
        return cls(
            batch_size=int(json["batch-size"]),
            max_iterations=int(json["max-iterations"]),
            number_of_clusters=int(json["number-of-clusters"]),
        )


@dataclasses.dataclass
class GlobalPaletteConfig:
    size: int
    coverage: float
    random: bool
    batching_k_means: BatchingKMeansConfig
    parent: "Config | None" = None

    @classmethod
    def from_json(cls, json: typing.Dict[str, typing.Any]) -> typing.Self:
        config = cls(
            size=int(json["size"]),
            coverage=float(json["coverage"]),
            random=bool(json["random"]),
            batching_k_means=BatchingKMeansConfig.from_json(json["batching-k-means"])
        )
        config.batching_k_means.parent = config
        return config


@dataclasses.dataclass
class Config:
    dataset_path: str
    dataset_labels_path: str
    batch_size: bitmath.Bitmath
    global_palette: GlobalPaletteConfig
    random_seed: int

    @classmethod
    def from_json(cls, json: typing.Dict[str, typing.Any]) -> typing.Self:
        config = cls(
            dataset_path=json["dataset-path"],
            dataset_labels_path=json["dataset-labels-path"],
            batch_size=bitmath.parse_string(json["loader"]["batch-size"]),
            global_palette=GlobalPaletteConfig.from_json(json["global-palette"]),
            random_seed=int(json["random-seed"])
        )
        config.global_palette.parent = config
        return config
