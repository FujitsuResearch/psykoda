"""IO functionalities for labeled log"""

from __future__ import annotations

from typing import Tuple, Union

from psykoda.io.labeled import file
from psykoda.io.labeled.loader import Loader
from psykoda.io.labeled.saver import Saver

Config = Union[file.FileStorageConfig]


def factory(config: Config) -> Tuple[Loader, Saver]:
    if isinstance(config, file.FileStorageConfig):
        return (
            file.FileLoader(config.base, config.load),
            file.FileSaver(config.base, config.save),
        )
    raise ValueError(f"config must be {Config}, got {config}")
