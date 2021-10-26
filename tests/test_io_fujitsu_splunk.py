from datetime import datetime
from os import path

import pytest

from psykoda.io.reader._fujitsu_splunk import FujitsuSplunk
from psykoda.preprocess import RoundDatetime, set_index

# pylint: disable=protected-access

rsrc_dir = path.join(
    path.dirname(path.abspath(__file__)),
    "rsrc",
    __name__.replace("tests.", ""),
)


@pytest.fixture
def io():
    return FujitsuSplunk(
        dir_IDS_log=path.join(rsrc_dir, "0"),
    )


@pytest.fixture
def preprocess():
    return lambda df: set_index(RoundDatetime("hour")(df))


def test_io_fujitsu_splunk_load_no_file(io):
    """Neither .csv.zip nor .csv exists"""
    with pytest.raises(FileNotFoundError):
        io._load_log_raw(datetime(2021, 4, 1))


def test_io_fujitsu_splunk_load_raw(io):
    """.csv has fujitsu-splunk columns"""
    assert list(io._load_log_raw(datetime(2021, 4, 2)).columns.array) == [
        "date_year",
        "date_month",
        "date_mday",
        "date_hour",
        "date_minute",
        "date_second",
        "src_ip",
        "src_port",
        "dest_ip",
        "dest_port",
        "sid",
        "host",
        "PRIORITY",
        "event_name",
    ]


def test_io_fujitsu_splunk_load(io, preprocess):
    """.csv.zip converts to canonical form"""
    df = preprocess(io.load_log(datetime(2021, 4, 3)))

    assert list(df.index.get_level_values("datetime_rounded")) == [
        datetime(2021, 4, 3, 3)
    ]
    assert list(df.index.get_level_values("src_ip")) == ["10.1.1.1"]

    for name, values in {
        "datetime_full": [datetime(2021, 4, 3, 3, 34, 5)],
        "src_port": [33434],
        "dest_ip": ["10.1.1.2"],
        "dest_port": [445],
        "sid": [200000],
    }.items():
        assert list(df[name]) == values


def test_io_fujitsu_splunk_load_zip(io):
    """.csv.zip takes precedence over .csv"""
    df = io.load_log(datetime(2021, 4, 4))
    for name, values in {
        "datetime_full": [datetime(2021, 4, 4, 4, 54, 32)],
        "src_port": [4555],
        "dest_ip": ["10.1.1.2"],
        "dest_port": [444],
        "sid": [204444],
    }.items():
        assert list(df[name]) == values


def test_io_fujitsu_splunk_nrows_read(io):
    """nrows_read option"""
    io.nrows_read = 2  # monkey patch
    df = io.load_log(datetime(2021, 4, 5))
    assert len(df) == 2
    io.nrows_read = None  # monkey patch
    df_full = io.load_log(datetime(2021, 4, 5))
    assert len(df_full) == 4
