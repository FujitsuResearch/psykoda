import ipaddress
import pandas as pd
import pytest

from psykoda import preprocess
from psykoda.constants import col

COLUMNNAME_ROWNO = "_row_number"


@pytest.fixture
def log_1():
    return pd.DataFrame(
        {
            col.SRC_IP: [
                "11.11.11.11",
                "11.11.11.22",
                "22.22.22.11",
                "22.22.22.22",
                "22.22.22.200",
            ],
            col.DEST_PORT: [21, 22, 22, 23, 24],
            COLUMNNAME_ROWNO: range(5),
        }
    ).set_index(col.SRC_IP)


@pytest.fixture
def log_2():
    return pd.DataFrame(
        {"column_1": [0, 1, 2, 3, 4, 5]},
        index=pd.Index(
            [
                "11.11.11.1",
                "11.11.11.2",
                "11.11.11.3",
                "11.11.11.1",
                "11.11.11.2",
                "11.11.11.1",
            ],
            name=col.SRC_IP,
        ),
    )


def screening_numlog(log: pd.DataFrame, min_n: int, max_n: int = 10 ** 8):
    return preprocess.screening_numlog(log, preprocess.ScreeningConfig(min_n, max_n))


def test_extract_log_1():
    empty_df = pd.DataFrame(columns=[1, 2, 3])
    result = preprocess.extract_log(empty_df, None, None, None)

    assert result.equals(empty_df)


def test_extract_log_2(log_1):
    result = preprocess.extract_log(log_1, None, None, [22])

    expect = pd.DataFrame(
        {
            col.SRC_IP: ["11.11.11.11", "22.22.22.22", "22.22.22.200"],
            col.DEST_PORT: [21, 23, 24],
            COLUMNNAME_ROWNO: [0, 3, 4],
        },
    ).set_index(col.SRC_IP)

    assert result.equals(expect)


def test_extract_log_3(log_1):
    result = preprocess.extract_log(log_1, None, None, None)

    assert result.equals(log_1)


def test_extract_log_4(log_1):
    subnets = ["11.11.11.0/24"]
    result = preprocess.extract_log(log_1, subnets, None, None)

    expect = pd.DataFrame(
        {
            col.SRC_IP: ["11.11.11.11", "11.11.11.22"],
            col.DEST_PORT: [21, 22],
            COLUMNNAME_ROWNO: [0, 1],
        },
    ).set_index(col.SRC_IP)

    assert result.equals(expect)


def test_extract_log_5(log_1):
    subnets = ["22.22.22.0/25"]
    result = preprocess.extract_log(log_1, subnets, None, None)

    expect = pd.DataFrame(
        {
            col.SRC_IP: ["22.22.22.11", "22.22.22.22"],
            col.DEST_PORT: [22, 23],
            COLUMNNAME_ROWNO: [2, 3],
        },
    ).set_index(col.SRC_IP)

    assert result.equals(expect)


def test_extract_log_6(log_1):
    subnets = ["22.22.22.128/25"]
    result = preprocess.extract_log(log_1, subnets, None, None)

    expect = pd.DataFrame(
        {
            col.SRC_IP: ["22.22.22.200"],
            col.DEST_PORT: [24],
            COLUMNNAME_ROWNO: [4],
        },
    ).set_index(col.SRC_IP)

    assert result.equals(expect)


def test_extract_log_7(log_1):
    subnets = ["22.22.22.200"]
    result = preprocess.extract_log(log_1, subnets, None, None)

    expect = pd.DataFrame(
        {
            col.SRC_IP: ["22.22.22.200"],
            col.DEST_PORT: [24],
            COLUMNNAME_ROWNO: [4],
        },
    ).set_index(col.SRC_IP)

    assert result.equals(expect)


def test_extract_log_8(log_1):
    subnets = ["11.11.11.0/24", "22.22.22.0/25"]
    result = preprocess.extract_log(log_1, subnets, None, None)

    expect = pd.DataFrame(
        {
            col.SRC_IP: [
                "11.11.11.11",
                "11.11.11.22",
                "22.22.22.11",
                "22.22.22.22",
            ],
            col.DEST_PORT: [21, 22, 22, 23],
            COLUMNNAME_ROWNO: [0, 1, 2, 3],
        },
    ).set_index(col.SRC_IP)

    assert result.equals(expect)


def test_extract_log_9(log_1):
    subnets = ["11.11.11.0/24", "22.22.22.0/25"]
    result = preprocess.extract_log(log_1, subnets, None, [22])

    expect = pd.DataFrame(
        {
            col.SRC_IP: ["11.11.11.11", "22.22.22.22"],
            col.DEST_PORT: [21, 23],
            COLUMNNAME_ROWNO: [0, 3],
        },
    ).set_index(col.SRC_IP)

    assert result.equals(expect)


def test_extract_log_10(log_1):
    result = preprocess.extract_log(log_1, None, [22, 23], None)

    expect = pd.DataFrame(
        {
            col.SRC_IP: ["11.11.11.22", "22.22.22.11", "22.22.22.22"],
            col.DEST_PORT: [22, 22, 23],
            COLUMNNAME_ROWNO: [1, 2, 3],
        },
    ).set_index(col.SRC_IP)

    assert result.equals(expect)


def test_extract_log_11(log_1):
    result = preprocess.extract_log(log_1, None, [22, 23], [22])

    expect = pd.DataFrame(
        {
            col.SRC_IP: ["22.22.22.22"],
            col.DEST_PORT: [23],
            COLUMNNAME_ROWNO: [3],
        },
    ).set_index(col.SRC_IP)

    assert result.equals(expect)


def test_extract_log_12(log_1):
    result = preprocess.extract_log(log_1, ["11.11.11.0/24"], [21, 22, 23], [21, 23])

    expect = pd.DataFrame(
        {
            col.SRC_IP: ["11.11.11.22"],
            col.DEST_PORT: [22],
            COLUMNNAME_ROWNO: [1],
        },
    ).set_index(col.SRC_IP)

    assert result.equals(expect)


def test_extract_log_13(log_1):
    with pytest.warns(Warning, match="all IP addresses are included"):
        result = preprocess.extract_log(log_1, [], None, None)

    assert result.equals(log_1)


def test_extract_log_14():
    original = pd.DataFrame(
        {
            col.SRC_IP: [
                "9.0.0.0",
                "10.0.0.0",
                "11.0.0.0",
                "172.15.0.0",
                "172.16.0.0",
                "172.31.0.0",
                "172.32.0.0",
                "192.167.0.0",
                "192.168.0.0",
                "192.169.0.0",
            ],
            col.DEST_PORT: [0 for _ in range(10)],
        }
    ).set_index(col.SRC_IP)
    for (subnet, rows) in {
        "private-A": [1],
        "private-B": [4, 5],
        "private-C": [8],
    }.items():
        assert preprocess.extract_log(original, [subnet]).equals(original.iloc[rows])


def test_screening_numlog_1(log_2):
    expect = log_2.copy()
    result = screening_numlog(log_2, 0)

    assert result.equals(expect)


def test_screening_numlog_2(log_2):
    result = screening_numlog(log_2, 2, 3)
    expect = pd.DataFrame(
        {"column_1": [0, 1, 3, 4, 5]},
        index=[
            "11.11.11.1",
            "11.11.11.2",
            "11.11.11.1",
            "11.11.11.2",
            "11.11.11.1",
        ],
    )

    assert result.equals(expect)


def test_screening_numlog_3(log_2):
    result = screening_numlog(log_2, 1, 2)
    expect = pd.DataFrame(
        {"column_1": [1, 2, 4]},
        index=[
            "11.11.11.2",
            "11.11.11.3",
            "11.11.11.2",
        ],
    )

    assert result.equals(expect)


def test_screening_numlog_4(log_2):
    result = screening_numlog(log_2, 2, 2)
    expect = pd.DataFrame(
        {"column_1": [1, 4]},
        index=[
            "11.11.11.2",
            "11.11.11.2",
        ],
    )

    assert result.equals(expect)


def test_screening_numlog_5(log_2):
    expect = log_2.copy()
    result = screening_numlog(log_2, 1, 3)

    assert result.equals(expect)


def test_addr_in_subnets():
    sub_networks = [
        ipaddress.ip_network("192.168.1.0/24"),
        ipaddress.ip_network("10.10.10.0/23"),
        ipaddress.ip_network("20.20.20.0/25"),
    ]

    expected_table = {
        "192.168.0.254": False,
        "192.168.1.1": True,
        "192.168.1.254": True,
        "192.168.2.1": False,
        "10.10.9.254": False,
        "10.10.10.1": True,
        "10.10.11.254": True,
        "10.10.12.1": False,
        "20.20.19.254": False,
        "20.20.20.1": True,
        "20.20.20.126": True,
        "20.20.20.128": False,
    }

    helper_func = preprocess.addr_in_subnets(sub_networks)
    for ipaddr, expected_result in expected_table.items():
        assert helper_func(ipaddr) == expected_result
