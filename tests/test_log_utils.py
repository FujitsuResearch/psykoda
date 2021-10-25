import pandas as pd
import pytest

from psykoda import preprocess
from psykoda.cli import internal
from psykoda.constants import col

COLUMNNAME_ROWNO = "_row_number"


@pytest.fixture
def log_3():
    return pd.DataFrame(
        {
            "user": [
                "alice",
                "bob",
                "carol",
                "dave",
                "eve",
                "frank",
            ]
            * 2
            + ["88"],
        },
        index=pd.Index(
            [
                "10.0.0.1",
                "10.0.0.3",
                "10.0.0.7",
                "10.0.0.15",
                "172.15.0.0",
                "172.16.0.0",
                "172.17.0.0",
                "172.18.0.0",
                "192.168.0.0",
                "192.168.0.255",
                "192.168.255.255",
                "192.255.255.255",
                "88.88.88.88",
            ],
            name=col.SRC_IP,
        ),
    )


def test_filter_out_no_match(log_3):
    expected = log_3.copy()
    actual = preprocess.filter_out(log_3, "user", ["ellen"])
    assert actual.equals(expected)


def test_filter_out_exact_single_match_value(log_3):
    expected = log_3.iloc[[0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12]]
    actual = preprocess.filter_out(log_3, "user", ["eve"])
    assert actual.equals(expected)


def test_filter_out_cidr_multiple_matches_index(log_3):
    expected = log_3.iloc[[4, 11, 12]]
    actual = preprocess.filter_out(
        log_3, col.SRC_IP, ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
    )
    assert actual.equals(expected)


def test_filter_out_nonexistent_column(log_3):
    with pytest.raises(KeyError, match="No value or index column"):
        preprocess.filter_out(log_3, "nonexistent", [])


def test_validate_patterns_valid():
    valid_index = pd.Index(["zero", "one", "two"])
    expected = valid_index.copy()
    actual = internal.validate_patterns(valid_index)
    assert actual.equals(expected)


def test_validate_patterns_invalid():
    invalid_index = pd.Index(["zero ", " one ", " two"])
    expected = pd.Index(["zero", "one", "two"])
    with pytest.warns(Warning, match="Spaces around exclusion pattern"):
        actual = internal.validate_patterns(invalid_index)
    assert actual.equals(expected)


def test_validate_patterns_nocheck():
    numeric_index = pd.Index(range(10))
    expected = numeric_index.copy()
    actual = internal.validate_patterns(numeric_index)
    assert actual.equals(expected)
