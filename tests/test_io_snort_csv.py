from datetime import datetime
from os import path

import pytest

from psykoda.io.reader.snort_csv import (
    ColumnsNotFound,
    SnortCSV,
    columns_from_conf_line,
    default_columns,
)
from psykoda.preprocess import RoundDatetime, set_index

rsrc_dir = path.join(
    path.dirname(path.abspath(__file__)),
    "rsrc",
    __name__.replace("tests.", ""),
)


@pytest.fixture
def io():
    with pytest.warns(UserWarning, match="year will be completed"):
        return SnortCSV(
            filename=path.join(rsrc_dir, "alert.csv"),
        )


@pytest.fixture
def preprocess():
    return lambda df: set_index(RoundDatetime("hour")(df))


def test_io_snort_csv(io, preprocess):
    df = preprocess(io.load_log(datetime(2021, 5, 17)))
    assert df.index.names == ["datetime_rounded", "src_ip"]
    for row, name, value in [
        (0, "datetime_full", datetime(2021, 5, 17, 10, 55, 17, 873218)),
        (1, "src_port", 51911),
        (2, "dest_ip", "192.168.12.3"),
        (3, "dest_port", 80),
        (4, "sid", 26834),
    ]:
        assert df[name].iloc[row] == value
    assert (
        df.to_csv(line_terminator="\n")
        == open(path.join(rsrc_dir, "alert.asloaded.csv"), encoding="utf_8").read()
    )


def test_io_snort_config():
    with pytest.raises(ColumnsNotFound):
        columns_from_conf_line("")
    with pytest.raises(ColumnsNotFound):
        columns_from_conf_line("#comment")
    with pytest.raises(ColumnsNotFound):
        columns_from_conf_line(" # comment")
    with pytest.raises(ColumnsNotFound):
        columns_from_conf_line("some random content")
    with pytest.raises(ColumnsNotFound):
        columns_from_conf_line("randomly prefixed output alert_csv")
    assert columns_from_conf_line("output alert_csv") == default_columns
    assert columns_from_conf_line("output alert_csv:") == default_columns
    assert columns_from_conf_line(" output alert_csv ") == default_columns
    assert columns_from_conf_line(" output alert_csv : ") == default_columns
    with pytest.raises(AssertionError):
        columns_from_conf_line(" output alert_csv alert.csv")
    assert columns_from_conf_line("output alert_csv: alert.csv") == default_columns
    assert columns_from_conf_line("output alert_csv: alert.csv msg,src,dst") == [
        "msg",
        "src",
        "dst",
    ]
    assert columns_from_conf_line("output alert_csv: alert.csv msg,src,dst 10M") == [
        "msg",
        "src",
        "dst",
    ]
