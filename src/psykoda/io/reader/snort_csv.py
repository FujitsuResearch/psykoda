"""
Snort CSV files
"""

import re
import warnings
from datetime import datetime, timedelta
from typing import List, TextIO

import pandas

from psykoda.constants import col
from psykoda.io.reader.base import Reader

default_columns = [
    "timestamp",
    "sig_generator",
    "sig_id",
    "sig_rev",
    "msg",
    "proto",
    "src",
    "srcport",
    "dst",
    "dstport",
    "ethsrc",
    "ethdst",
    "ethlen",
    "tcpflags",
    "tcpseq",
    "tcpack",
    "tcplen",
    "tcpwindow",
    "ttl",
    "tos",
    "id",
    "dgmlen",
    "iplen",
    "icmptype",
    "icmpcode",
    "icmpid",
    "icmpseq",
]


class SnortCSV(Reader):
    """Load IDS log from Snort CSV files."""

    def __init__(  # pylint: disable=dangerous-default-value,redefined-outer-name
        # columns is read-only
        # intended use: SnortCSV(columns=columns(conf))
        self,
        *,
        filename: str,
        columns: List[str] = default_columns,
        year_included=False,
    ):
        """Load IDS log from Snort CSV files.

        Parameters
        ----------
        filename
            Name of file to read the log from.
        columns
            read_csv(names): column names and order that CSV file contains.
        year_included
            Whether timestamp column has years included: True for log with `snort -y`.

        Issues
        ------
        * Only works when year_included=True.
        * Log rotation is not supported yet.
        """
        self.filename = filename
        self.columns = columns[:]  # shallow copy
        if year_included:
            datetime_format = "%m/%d/%y-%H:%M:%S.%f "
            self.date_parser = lambda s: datetime.strptime(s, datetime_format)
        else:
            warnings.warn("year will be completed")
            datetime_format = "%m/%d-%H:%M:%S.%f "
            self.date_parser = lambda s: self._complete_year(
                datetime.strptime(s, datetime_format)
            )

    def load_log(self, dt: datetime) -> pandas.DataFrame:
        df = self._load_log_raw(self.filename)
        df = df.assign(datetime_full=df["timestamp"].apply(self.date_parser))
        df = df[
            (dt <= df["datetime_full"]) & (df["datetime_full"] < dt + timedelta(days=1))
        ]
        return df.rename(columns=_columns_renaming)

    def _load_log_raw(self, filename: str):
        return pandas.read_csv(
            filename,
            header=None,
            names=self.columns,
        )

    def _complete_year(self, d: datetime):
        """Magic algorithm to complete year information.

        .. todo::
            internal API which subject to change, especially when log rotation is supported.

        Issues
        ------
        Works only in 2021.
        """
        return d.replace(year=(2021 if d.year == 1900 else d.year))


_columns_renaming = {
    "src": col.SRC_IP,
    "srcport": col.SRC_PORT,
    "dst": col.DEST_IP,
    "dstport": col.DEST_PORT,
    "sig_id": col.SID,
}


class ColumnsNotFound(Exception):
    """snort.conf line does not have columns information."""


def columns_from_conf_line(line: str):
    """Parse snort.conf line into columns information."""
    line = line.lower()
    if re.match(r"^\s#", line):
        raise ColumnsNotFound("line is comment")
    alert_csv = r"^\s*output\s+alert_csv\s*"
    if not re.match(alert_csv, line):
        raise ColumnsNotFound("line is not alert_csv")
    if re.match(alert_csv + r":?\s*$", line):
        return default_columns
    m = re.match(alert_csv + r":\s*(.+)$", line)
    assert m is not None, "a colon is expected after 'alert_csv'"
    options = re.sub(r"\s+", " ", m.group(1)).strip().split()
    assert len(options)
    if len(options) == 1:
        return default_columns
    return options[1].split(",")


def columns(conf: TextIO):
    """Parse snort.conf into columns information."""
    for line in conf:
        try:
            return columns_from_conf_line(line)
        except ColumnsNotFound:
            continue
    raise ColumnsNotFound
