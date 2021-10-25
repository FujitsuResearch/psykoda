"""File-based Previous Log Loader and Saver"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas

from psykoda.constants import col
from psykoda.io.internal import load_csv_optional_zip
from psykoda.io.labeled.loader import Loader
from psykoda.io.labeled.saver import Saver

logger = logging.getLogger(__name__)


@dataclass
class FileStorageBaseConfig:
    """Common configuration for FileLoader and FileSaver"""

    dir: str
    labeled_basename_format_datetime: str = "%Y-%m-%d-%H"


@dataclass
class FileStorageConfig:
    """Configuration fed to factory"""

    base: FileStorageBaseConfig
    load: FileLoader.Config
    save: FileSaver.Config


class FileLoader(Loader):
    """File-based Loader"""

    @dataclass
    class Config:
        """Configuration of FileLoader"""

    def __init__(self, base_config: FileStorageBaseConfig, config: Config):
        self.base_config = base_config
        self.config = config

    def load_previous_log(self, entries: pandas.MultiIndex) -> pandas.DataFrame:
        logs = []
        for entry in entries:
            log = self._load_previous_log(*entry)
            if log is not None:
                logs.append(log.loc[entry])
        if logs:
            return pandas.concat(logs)
        return pandas.DataFrame()

    def _load_previous_log(
        self, dt: datetime, src_ip: str
    ) -> Optional[pandas.DataFrame]:
        base_file_name = os.path.join(
            self.base_config.dir,
            dt.strftime(
                f"{self.base_config.labeled_basename_format_datetime}__{src_ip}"
            ),
        )
        try:
            return load_csv_optional_zip(
                base_file_name, parse_dates=[col.DATETIME_ROUNDED, col.DATETIME_FULL]
            ).set_index([col.DATETIME_ROUNDED, col.SRC_IP])
        except FileNotFoundError as ex:
            logger.warning(
                "labeled[%s] does not exist in %s",
                (dt, src_ip),
                self.base_config.dir,
                exc_info=ex,
            )
            return None


class FileSaver(Saver):
    """File-based Saver"""

    @dataclass
    class Config:
        """Configuration of FileSaver"""

        all: bool = False
        compression: bool = False

    def __init__(self, base_config: FileStorageConfig, config: Config):
        self.base_config = base_config
        self.config = config
        if not os.path.isdir(self.base_config.dir):
            os.makedirs(self.base_config.dir)

    def save_previous_log(self, df: pandas.DataFrame, entries: pandas.MultiIndex):
        for (dt, src_ip) in entries:
            self._save_previous_log(df, dt, src_ip)

    def _save_previous_log(
        self,
        df: pandas.DataFrame,
        dt: datetime,
        src_ip: str,
    ) -> str:
        basename = dt.strftime(
            f"{self.base_config.labeled_basename_format_datetime}__{src_ip}"
        )
        if self.config.compression:
            kwargs = {
                "path_or_buf": os.path.join(self.base_config.dir, basename + ".zip"),
                "compression": {
                    "method": "zip",
                    "archive_name": basename,
                },
            }
        else:
            kwargs = {
                "path_or_buf": os.path.join(self.base_config.dir, basename + ".csv"),
            }
        logger.debug("len(df) = %s", len(df))
        df = df[df.index.get_level_values(col.SRC_IP) == src_ip]
        logger.debug("len(df|%s) = %s", src_ip, len(df))
        if not self.config.all:
            df = df[df.index.get_level_values(col.DATETIME_ROUNDED) == dt]
            logger.debug("len(df|(%s,%s)) = %s", dt, src_ip, len(df))
        df.to_csv(**kwargs)
        return kwargs["path_or_buf"]
