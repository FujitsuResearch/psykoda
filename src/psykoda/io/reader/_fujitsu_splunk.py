"""
Fujitsu-Splunk-exported CSV files
"""

import logging
from datetime import datetime
from os import path
from typing import Optional

import pandas

from psykoda.constants import COMMANDLINE_DATE_FORMAT
from psykoda.io import internal
from psykoda.io.reader.base import Reader

logger = logging.getLogger(__name__)


class FujitsuSplunk(Reader):
    """Load IDS log from Fujitsu-Splunk-exported CSV files."""

    def __init__(
        self,
        *,
        dir_IDS_log: str,
        nrows_read: Optional[int] = None,
    ):
        """Load IDS log from Fujitsu-Splunk-exported CSV files.

        Parameters
        ----------
        dir_IDS_log
            Path to directory where CSV files reside.
        nrows_read
            pandas.read_csv(nrows): number of first rows to read from a file.
            Use for debugging or low-memory environment (and incomplete result is acceptable).
        """
        self.dir_IDS_log = dir_IDS_log
        self.nrows_read = nrows_read

    def load_log(self, dt: datetime) -> pandas.DataFrame:
        df = self._load_log_raw(dt)
        return df.assign(datetime_full=self._datetime_full(df))

    def _load_log_raw(self, dt: datetime):
        base_file_name = path.join(
            self.dir_IDS_log, dt.strftime(f"log_{COMMANDLINE_DATE_FORMAT}")
        )
        return internal.load_csv_optional_zip(base_file_name, nrows=self.nrows_read)

    def _datetime_full(self, df: pandas.DataFrame):
        dt_columns = df[_time_unit_columns.keys()]
        dt_columns = dt_columns.assign(
            date_month=dt_columns["date_month"].map(_month_from_name)
        )
        return pandas.to_datetime(dt_columns.rename(columns=_time_unit_columns))


_time_unit_columns = {
    "date_year": "year",
    "date_month": "month",
    "date_mday": "day",
    "date_hour": "hour",
    "date_minute": "minute",
    "date_second": "second",
}

_month_from_name = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
