import datetime
import json
import logging
import os
import random
from json.decoder import JSONDecodeError

import pandas
import pytest

from psykoda import utils
from psykoda.constants import COMMANDLINE_DATE_FORMAT

logger = logging.getLogger(__name__)

rsrc_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "rsrc", "test_utils"
)


def _gen_tempfilepath(parent_dir: str = None, retry_max: int = 10):
    for _ in range(retry_max):
        tmpname = hex(random.getrandbits(128))
        if parent_dir:
            tmppath = os.path.join(parent_dir, tmpname)
        else:
            tmppath = tmpname

        if os.path.isfile(tmppath):
            continue

        return tmppath

    raise RuntimeError


def test_daterange2list_01():
    start = datetime.datetime.strptime("2012-04-28", COMMANDLINE_DATE_FORMAT)
    end = datetime.datetime.strptime("2012-04-30", COMMANDLINE_DATE_FORMAT)
    expect = [
        datetime.datetime.strptime(ymd, COMMANDLINE_DATE_FORMAT)
        for ymd in ["2012-04-28", "2012-04-29", "2012-04-30"]
    ]

    assert utils.daterange2list(start, end) == expect


def test_daterange2list_02():
    start = datetime.datetime.strptime("2012-04-28", COMMANDLINE_DATE_FORMAT)
    end = datetime.datetime.strptime("2012-04-28", COMMANDLINE_DATE_FORMAT)
    expect = [datetime.datetime.strptime("2012-04-28", COMMANDLINE_DATE_FORMAT)]

    assert utils.daterange2list(start, end) == expect


def test_daterange2list_03():
    start = datetime.datetime.strptime("2012-04-30", COMMANDLINE_DATE_FORMAT)
    end = datetime.datetime.strptime("2012-05-01", COMMANDLINE_DATE_FORMAT)
    expect = [
        datetime.datetime.strptime(ymd, COMMANDLINE_DATE_FORMAT)
        for ymd in ["2012-04-30", "2012-05-01"]
    ]

    assert utils.daterange2list(start, end) == expect


def test_daterange2list_04():
    start = datetime.datetime.strptime("2012-12-31", COMMANDLINE_DATE_FORMAT)
    end = datetime.datetime.strptime("2013-01-01", COMMANDLINE_DATE_FORMAT)
    expect = [
        datetime.datetime.strptime(ymd, COMMANDLINE_DATE_FORMAT)
        for ymd in ["2012-12-31", "2013-01-01"]
    ]

    assert utils.daterange2list(start, end) == expect


def test_daterange2list_05():
    with pytest.raises(TypeError):
        utils.daterange2list("2012-04-28", "2012-04-29")


def test_daterange_contains():
    """in binary operator"""
    start_inclusive = datetime.datetime(2020, 4, 1)
    start_exclusive = datetime.datetime(2020, 3, 31)
    end_inclusive = datetime.datetime(2020, 4, 30)
    end_exclusive = datetime.datetime(2020, 5, 1)
    dr = utils.DateRange(start_inclusive=start_inclusive, end_exclusive=end_exclusive)
    assert start_inclusive in dr
    assert start_exclusive not in dr
    assert end_inclusive in dr
    assert end_exclusive not in dr


def test_daterange_iter():
    """iteration"""
    start_inclusive = datetime.datetime(2020, 4, 1)
    end_inclusive = datetime.datetime(2020, 4, 30)
    end_exclusive = datetime.datetime(2020, 5, 1)
    dr = utils.DateRange(start_inclusive=start_inclusive, end_exclusive=end_exclusive)
    for (actual, expected) in zip(
        dr, utils.daterange2list(start_inclusive, end_inclusive)
    ):
        assert actual == expected


def test_daterange_attr_vals():
    """attribute values"""
    start_inclusive = datetime.datetime(2020, 4, 1)
    start_exclusive = datetime.datetime(2020, 3, 31)
    end_inclusive = datetime.datetime(2020, 4, 30)
    end_exclusive = datetime.datetime(2020, 5, 1)
    length = 30
    dr = utils.DateRange(start_inclusive=start_inclusive, end_exclusive=end_exclusive)
    assert dr._start == start_inclusive
    assert dr._end == end_exclusive
    assert len(dr) == length
    dr = utils.DateRange(start_exclusive=start_exclusive, end_exclusive=end_exclusive)
    assert dr._start == start_inclusive
    assert dr._end == end_exclusive
    assert len(dr) == length
    dr = utils.DateRange(start_inclusive=start_inclusive, end_inclusive=end_inclusive)
    assert dr._start == start_inclusive
    assert dr._end == end_exclusive
    assert len(dr) == length
    dr = utils.DateRange(start_exclusive=start_exclusive, end_inclusive=end_inclusive)
    assert dr._start == start_inclusive
    assert dr._end == end_exclusive
    assert len(dr) == length
    dr = utils.DateRange(start_inclusive=start_inclusive, length=length)
    assert dr._start == start_inclusive
    assert dr._end == end_exclusive
    assert len(dr) == length
    dr = utils.DateRange(start_exclusive=start_exclusive, length=length)
    assert dr._start == start_inclusive
    assert dr._end == end_exclusive
    assert len(dr) == length
    dr = utils.DateRange(end_inclusive=end_inclusive, length=length)
    assert dr._start == start_inclusive
    assert dr._end == end_exclusive
    assert len(dr) == length
    dr = utils.DateRange(end_exclusive=end_exclusive, length=length)
    assert dr._start == start_inclusive
    assert dr._end == end_exclusive
    assert len(dr) == length


def test_load_json_01():
    data = utils.load_json(os.path.join(rsrc_dir, "testdata_01.json"))
    assert data["key"] == "value"


def test_load_json_02():
    data = utils.load_json(os.path.join(rsrc_dir, "testdata_02.json"))
    assert data[1]["key"] == 2


def test_load_json_03():
    with pytest.raises(JSONDecodeError):
        utils.load_json(os.path.join(rsrc_dir, "testdata_03.json"))


def test_save_json():
    tmppath = _gen_tempfilepath(rsrc_dir)
    testdata = {"key": "value"}
    try:
        utils.save_json(testdata, tmppath)
        with open(tmppath, "r", encoding="utf_8") as file:
            data = json.load(file)
        assert data["key"] == "value"
    finally:
        if os.path.isfile(tmppath):
            os.remove(tmppath)


def test_get_series_single():
    single_index = pandas.Index([i for i in range(2) for _ in range(3)], name="column")
    expected = pandas.Series([i for i in range(2) for _ in range(3)], name="column")

    actual = utils.get_series(single_index, "column")
    assert actual.index.equals(single_index)
    assert actual.reset_index(drop=True).equals(expected)


def test_get_series_multi_0():
    multi_index = pandas.MultiIndex.from_tuples(
        [(i, j) for i in range(2) for j in range(3) for _ in range(2)],
        names=["column", "another"],
    )
    expected = pandas.Series([i for i in range(2) for j in range(3) for _ in range(2)])

    actual = utils.get_series(multi_index, "column")
    assert actual.index.equals(multi_index)
    assert actual.reset_index(drop=True).equals(expected)


def test_get_series_multi_1():
    multi_index = pandas.MultiIndex.from_tuples(
        [(i, j) for i in range(2) for j in range(3) for _ in range(2)],
        names=["another", "column"],
    )
    expected = pandas.Series([j for i in range(2) for j in range(3) for _ in range(2)])

    actual = utils.get_series(multi_index, "column")
    assert actual.index.equals(multi_index)
    assert actual.reset_index(drop=True).equals(expected)
