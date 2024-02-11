import dataclasses
import json as json_module

import bitmath
import typing


PROFILE = False


@dataclasses.dataclass
class BatchingKMeansConfig:
    batch_size: int
    max_iterations: int
    number_of_clusters: int
    parent: "GlobalPaletteConfig | LocalPaletteConfig | None" = None

    @classmethod
    def from_json(cls, json: typing.Dict[str, typing.Any]) -> typing.Self:
        return cls(
            batch_size=int(json["batch-size"]),
            max_iterations=int(json["max-iterations"]),
            number_of_clusters=int(json["number-of-clusters"]),
        )


@dataclasses.dataclass
class HDF5StorageConfig:
    base_directory: str
    dataset: str

    @classmethod
    def from_json(cls, json: typing.Dict[str, typing.Any]) -> typing.Self:
        return cls(
            base_directory=json["base-directory"],
            dataset=json["dataset"]
        )


@dataclasses.dataclass
class GlobalPaletteConfig:
    size: int
    coverage: float
    random: bool
    batching_k_means: BatchingKMeansConfig
    patch_size: int
    parent: "Config | None" = None

    @classmethod
    def from_json(cls, json: typing.Dict[str, typing.Any]) -> typing.Self:
        config = cls(
            size=int(json["size"]),
            coverage=float(json["coverage"]),
            random=bool(json["random"]),
            batching_k_means=BatchingKMeansConfig.from_json(json["batching-k-means"]),
            patch_size=int(json["patch-size"])
        )
        config.batching_k_means.parent = config
        return config


@dataclasses.dataclass
class LocalPaletteConfig:
    size: int
    coverage: float
    random: bool
    batching_k_means: BatchingKMeansConfig
    patch_size: int
    k_neigh: int
    parent: "Config | None" = None

    @classmethod
    def from_json(cls, json: typing.Dict[str, typing.Any]) -> typing.Self:
        config = cls(
            size=int(json["size"]),
            coverage=float(json["coverage"]),
            random=bool(json["random"]),
            batching_k_means=BatchingKMeansConfig.from_json(json["batching-k-means"]),
            patch_size=int(json["patch-size"]),
            k_neigh=int(json["k-neigh"]),
        )
        config.batching_k_means.parent = config
        return config


@dataclasses.dataclass
class Config:
    dataset_path: str
    dataset_labels_path: str
    batch_size: bitmath.Bitmath
    global_palette: GlobalPaletteConfig
    local_palette: LocalPaletteConfig
    random_seed: int
    hdf5_storage: HDF5StorageConfig | None = None

    @classmethod
    def from_json(cls, json: typing.Dict[str, typing.Any]) -> typing.Self:
        config = cls(
            dataset_path=json["dataset-path"],
            dataset_labels_path=json["dataset-labels-path"],
            batch_size=bitmath.parse_string(json["loader"]["batch-size"]),
            global_palette=GlobalPaletteConfig.from_json(json["global-palette"]),
            local_palette=LocalPaletteConfig.from_json(json["local-palette"]),
            random_seed=int(json["random-seed"])
        )
        if json["data-storage"] == "hdf5":
            config.hdf5_storage = HDF5StorageConfig.from_json(json["hdf5"])
        config.global_palette.parent = config
        config.local_palette.parent = config
        return config

    @classmethod
    def default(cls) -> typing.Self:
        with open("./config.json", "r") as _:
            return cls.from_json(json_module.load(_))


default_config = Config.default()
