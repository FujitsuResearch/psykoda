import datetime
import ipaddress
import itertools
from os import path

import dacite
import numpy
import pandas
import pytest
import scipy.sparse

from psykoda.constants import col, ip
from psykoda.feature_extraction import (
    FeatureExtractionConfig,
    FeatureLabel,
    calculate_idf,
    feature_extraction_all,
    location_matcher,
)

rsrc_dir = path.join(
    path.dirname(path.abspath(__file__)),
    "rsrc",
    __name__.replace("tests.", ""),
)


def test_idf_single_index():
    df = pandas.read_csv(path.join(rsrc_dir, "idf.in.csv"))[["i0", "c0"]].set_index(
        "i0"
    )
    assert len(df) == 1024
    assert df.index.nunique() == 144
    idf = calculate_idf(df, "c0", num_idf_feature=16, min_count=8)[0]

    # Equality actually holds for test data.
    assert len(idf) == 16

    # Since min_count is threshold for value_counts
    # but IDF is calculated on unique index values,
    # "c31" value appearing in eight rows with seven
    # unique indexes has IDF value of 144/7, which
    # is log1p-transformed.
    assert numpy.isclose(idf.max(), numpy.log(144 / 7 + 1))


def test_idf_multi_index():
    df = pandas.read_csv(path.join(rsrc_dir, "idf.in.csv")).set_index(["i0", "i1"])
    assert df.index.nunique() == 806

    idf = calculate_idf(df, "c1", num_idf_feature=8, min_count=8)[0]

    # Does not match num_idf_feature, because of ties.
    assert len(idf) == 10

    # "c28" and "c37" both appear in eight rows with
    # distinct index values.
    assert numpy.isclose(idf.max(), numpy.log(806 / 8 + 1))


def test_location_matcher_nooverlap():
    matcher = location_matcher(
        *zip(
            ("10.0.0.0/23", "Japan"),
            ("10.0.2.0/24", "US"),
            ("10.0.3.0/25", "EU"),
            ("10.0.3.128/25", "UK"),
        )
    )
    for (arg, expected) in [
        ("10.0.0.0", "Japan"),
        ("10.0.1.0", "Japan"),
        ("10.0.2.0", "US"),
        ("10.0.3.127", "EU"),
        ("10.0.3.128", "UK"),
        ("10.0.4.0", ip.UNKNOWN_PRIVATE),
        ("11.0.0.0", ip.UNKNOWN_GLOBAL),
    ]:
        assert matcher(ipaddress.ip_address(arg)) == expected


@pytest.fixture
def featurelabel_class():
    feature = scipy.sparse.csr_matrix(
        [
            [0, 0, 2, 0],
            [10, 0, 12, 0],
            [0, 0, 0, 0],
            [30, 0, 32, 0],
            [0, 0, 42, 0],
            [0, 0, 0, 0],
        ]
    )
    index = [
        (pandas.Timestamp("2021-06-01 01:00:00"), "1.1.1.1"),
        (pandas.Timestamp("2021-06-02 02:00:00"), "2.2.2.2"),
        (pandas.Timestamp("2021-06-03 03:00:00"), "3.3.3.3"),
        (pandas.Timestamp("2021-06-04 04:00:00"), "4.4.4.4"),
        (pandas.Timestamp("2021-06-05 05:00:00"), "5.5.5.5"),
        (pandas.Timestamp("2021-06-06 06:00:00"), "6.6.6.6"),
    ]
    columns = ["columns_%d" % i for i in range(4)]
    label = numpy.array(range(feature.shape[0]))
    idf_sid = None
    idf_dport = None

    return FeatureLabel(
        feature=feature,
        index=index,
        columns=columns,
        label=label,
        idf_sid=idf_sid,
        idf_dport=idf_dport,
    )


def assert_equals_FeatureLabel_fields(
    fl,
    expected_feature,
    expected_index,
    expected_columns,
    expected_label,
):
    assert (fl.feature - expected_feature).nnz == 0
    assert len(fl.index) == len(expected_index)
    for (ts1, addr1), (ts2, addr2) in zip(expected_index, fl.index):
        assert ts1 == ts2
        assert addr1 == addr2
    for c1, c2 in zip(expected_columns, fl.columns):
        assert c1 == c2
    assert all(fl.label == expected_label)


def test_FL_extract_nonzeros_rows_01(featurelabel_class):
    expected_feature = scipy.sparse.csr_matrix(
        [[0, 0, 2, 0], [10, 0, 12, 0], [30, 0, 32, 0], [0, 0, 42, 0]]
    )
    expected_index = [
        featurelabel_class.index[0],
        featurelabel_class.index[1],
        featurelabel_class.index[3],
        featurelabel_class.index[4],
    ]
    expected_columns = featurelabel_class.columns[:]
    expected_label = numpy.array([0, 1, 3, 4])

    featurelabel_class.extract_nonzeros_rows()

    assert_equals_FeatureLabel_fields(
        featurelabel_class,
        expected_feature,
        expected_index,
        expected_columns,
        expected_label,
    )


def test_FL_extract_nonzeros_cols_01(featurelabel_class):
    expected_feature = scipy.sparse.csr_matrix(
        [[0, 2], [10, 12], [0, 0], [30, 32], [0, 42], [0, 0]]
    )
    expected_index = featurelabel_class.index[:]
    expected_columns = [featurelabel_class.columns[0], featurelabel_class.columns[2]]
    expected_label = featurelabel_class.label[:]

    featurelabel_class.extract_nonzeros_cols()

    assert_equals_FeatureLabel_fields(
        featurelabel_class,
        expected_feature,
        expected_index,
        expected_columns,
        expected_label,
    )


def test_FL_extract_nonzeros_01(featurelabel_class):
    expected_feature = scipy.sparse.csr_matrix(
        [
            [0, 2],
            [10, 12],
            [30, 32],
            [0, 42],
        ]
    )
    expected_index = [
        featurelabel_class.index[0],
        featurelabel_class.index[1],
        featurelabel_class.index[3],
        featurelabel_class.index[4],
    ]
    expected_columns = [featurelabel_class.columns[0], featurelabel_class.columns[2]]
    expected_label = numpy.array([0, 1, 3, 4])

    featurelabel_class.extract_nonzeros()

    assert_equals_FeatureLabel_fields(
        featurelabel_class,
        expected_feature,
        expected_index,
        expected_columns,
        expected_label,
    )


def test_FL_extract_loc_01(featurelabel_class):
    expected_series = pandas.Series([42], index=["columns_2"])
    sample = (pandas.Timestamp("2021-06-05 05:00:00"), "5.5.5.5")

    actual_series = featurelabel_class.loc(sample)

    assert sum(actual_series != expected_series) == 0


def test_FL_put_labels_01(featurelabel_class):
    labeled_samples = pandas.Series(
        [1, 1], index=[featurelabel_class.index[2], featurelabel_class.index[4]]
    )
    expected_label = numpy.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    featurelabel_class.put_labels(labeled_samples)

    assert all(expected_label == featurelabel_class.label)


def test_FL_put_labels_02(featurelabel_class):
    expected_label = numpy.array([0.0] * len(featurelabel_class.index))

    featurelabel_class.put_labels(None)

    assert all(expected_label == featurelabel_class.label)


def test_FL_split_train_test_01(featurelabel_class):
    date_to_training = datetime.datetime.strptime("2021-06-04", "%Y-%m-%d")
    expected_train_mask = numpy.array([True, True, True, True, False, False])
    expected_train_feature = featurelabel_class.feature[expected_train_mask]
    expected_train_label = featurelabel_class.label[expected_train_mask]
    expected_test_feature = featurelabel_class.feature[~expected_train_mask]
    expected_test_index = pandas.Index(featurelabel_class.index)[~expected_train_mask]

    (
        actual_train_feature,
        actual_train_label,
        actual_test_feature,
        actual_test_index,
    ) = featurelabel_class.split_train_test(date_to_training)

    assert (actual_train_feature - expected_train_feature).nnz == 0
    assert all(actual_train_label == expected_train_label)
    assert (actual_test_feature - expected_test_feature).nnz == 0
    assert all(actual_test_index == expected_test_index)


def test_FL_split_train_test_02(featurelabel_class):
    featurelabel_class.label = None

    date_to_training = datetime.datetime.strptime("2021-06-04", "%Y-%m-%d")
    expected_train_mask = numpy.array([True, True, True, True, False, False])
    expected_train_feature = featurelabel_class.feature[expected_train_mask]
    expected_train_label = pandas.Series(0.0, index=featurelabel_class.index)[
        expected_train_mask
    ]
    expected_test_feature = featurelabel_class.feature[~expected_train_mask]
    expected_test_index = pandas.Index(featurelabel_class.index)[~expected_train_mask]

    (
        actual_train_feature,
        actual_train_label,
        actual_test_feature,
        actual_test_index,
    ) = featurelabel_class.split_train_test(date_to_training)

    assert (actual_train_feature - expected_train_feature).nnz == 0
    assert all(actual_train_label == expected_train_label)
    assert (actual_test_feature - expected_test_feature).nnz == 0
    assert all(actual_test_index == expected_test_index)


@pytest.fixture
def log_for_fea():
    log = pandas.DataFrame(
        [
            [
                pandas.Timestamp("2020-03-29 01:00:00"),
                "10.1.1.1",
                "10.1.1.10",
                1,
                "101",
            ],
            [
                pandas.Timestamp("2020-03-29 01:00:00"),
                "10.1.1.1",
                "10.1.1.10",
                1,
                "101",
            ],
            [
                pandas.Timestamp("2020-03-29 01:00:00"),
                "10.1.1.1",
                "10.1.1.10",
                1,
                "101",
            ],
            [
                pandas.Timestamp("2020-03-30 01:00:00"),
                "10.10.1.1",
                "10.10.1.10",
                2,
                "102",
            ],
            [
                pandas.Timestamp("2020-03-30 02:00:00"),
                "10.10.1.1",
                "10.10.1.10",
                2,
                "102",
            ],
            [
                pandas.Timestamp("2020-03-30 03:00:00"),
                "10.10.1.1",
                "10.10.1.10",
                2,
                "102",
            ],
            [
                pandas.Timestamp("2020-03-30 04:00:00"),
                "10.10.1.1",
                "10.10.1.10",
                2,
                "102",
            ],
            [
                pandas.Timestamp("2020-03-30 05:00:00"),
                "10.10.1.1",
                "10.10.1.10",
                2,
                "102",
            ],
            [
                pandas.Timestamp("2020-03-30 06:00:00"),
                "10.10.1.1",
                "10.10.1.10",
                2,
                "102",
            ],
            [
                pandas.Timestamp("2020-03-30 07:00:00"),
                "10.10.1.1",
                "10.10.1.10",
                2,
                "102",
            ],
            [
                pandas.Timestamp("2020-03-30 08:00:00"),
                "10.10.1.1",
                "10.10.1.10",
                2,
                "102",
            ],
            [
                pandas.Timestamp("2020-03-30 09:00:00"),
                "10.10.1.1",
                "10.10.1.10",
                2,
                "102",
            ],
            [
                pandas.Timestamp("2020-03-30 10:00:00"),
                "10.10.1.1",
                "10.10.1.10",
                2,
                "102",
            ],
            [
                pandas.Timestamp("2020-03-31 01:00:00"),
                "10.100.1.1",
                "10.1.1.10",
                3,
                "103",
            ],
            [
                pandas.Timestamp("2020-03-31 02:00:00"),
                "10.100.1.1",
                "10.10.1.10",
                3,
                "103",
            ],
            [
                pandas.Timestamp("2020-03-31 03:00:00"),
                "10.100.1.1",
                "10.100.1.10",
                4,
                "103",
            ],
            [
                pandas.Timestamp("2020-03-31 03:00:00"),
                "10.100.1.1",
                "10.100.1.10",
                5,
                "104",
            ],
        ],
        columns=[col.DATETIME_ROUNDED, col.SRC_IP, col.DEST_IP, col.DEST_PORT, col.SID],
    )
    log = log.set_index([col.DATETIME_ROUNDED, col.SRC_IP])
    return log


@pytest.fixture
def config_for_fea():
    config_dict = {
        "idf": {
            "sid": {"min_count": 2, "num_feature": 30},
            "dest_port": {"min_count": 2, "num_feature": 30},
        },
        "address_to_location": "DUMMY",
    }
    fe_config: FeatureExtractionConfig = dacite.from_dict(
        FeatureExtractionConfig, config_dict
    )
    return fe_config


@pytest.fixture
def iptable_for_fea():
    iptable = pandas.DataFrame(
        [
            ["10.1.0.0/16", "Japan"],
            ["10.10.0.0/16", "Europe"],
            ["10.100.0.0/16", "America"],
        ],
        columns=[col.IP_TABLE_SUBNET, col.IP_TABLE_LOCATION],
    )
    return iptable


def test_calculate_idf(log_for_fea, config_for_fea):
    grouped = log_for_fea.groupby("sid")

    expected_idf = pandas.Series(
        [2.708050, 1.734601, 0.875469],
        index=pandas.Index(["101", "103", "102"], name="sid"),
        name="sid" + col.IDF_SUFFIX,
    )
    expected_groups = [
        ("101", grouped.get_group("101")),
        ("103", grouped.get_group("103")),
        ("102", grouped.get_group("102")),
    ]
    actual_idf, actual_groups = calculate_idf(
        log_for_fea,
        col.SID,
        config_for_fea.idf["sid"].num_feature,
        config_for_fea.idf["sid"].min_count,
    )
    actual_idf = actual_idf.map(lambda x: round(x, 6))

    assert actual_idf.equals(expected_idf)
    for (actual_idx, actual_group), (expected_idx, expected_group) in zip(
        actual_groups, expected_groups
    ):
        assert actual_idx == expected_idx
        assert actual_group.equals(expected_group)


def test_feature_extraction_all(log_for_fea, config_for_fea, iptable_for_fea):
    expected_idf_sid, _ = calculate_idf(
        log_for_fea,
        col.SID,
        config_for_fea.idf["sid"].num_feature,
        config_for_fea.idf["sid"].min_count,
    )
    expected_idf_dport, _ = calculate_idf(
        log_for_fea,
        col.DEST_PORT,
        config_for_fea.idf["dest_port"].num_feature,
        config_for_fea.idf["dest_port"].min_count,
    )
    expected_index = [
        (pandas.Timestamp("2020-03-29 01:00:00"), "10.1.1.1"),
        (pandas.Timestamp("2020-03-30 01:00:00"), "10.10.1.1"),
        (pandas.Timestamp("2020-03-30 02:00:00"), "10.10.1.1"),
        (pandas.Timestamp("2020-03-30 03:00:00"), "10.10.1.1"),
        (pandas.Timestamp("2020-03-30 04:00:00"), "10.10.1.1"),
        (pandas.Timestamp("2020-03-30 05:00:00"), "10.10.1.1"),
        (pandas.Timestamp("2020-03-30 06:00:00"), "10.10.1.1"),
        (pandas.Timestamp("2020-03-30 07:00:00"), "10.10.1.1"),
        (pandas.Timestamp("2020-03-30 08:00:00"), "10.10.1.1"),
        (pandas.Timestamp("2020-03-30 09:00:00"), "10.10.1.1"),
        (pandas.Timestamp("2020-03-30 10:00:00"), "10.10.1.1"),
        (pandas.Timestamp("2020-03-31 01:00:00"), "10.100.1.1"),
        (pandas.Timestamp("2020-03-31 02:00:00"), "10.100.1.1"),
        (pandas.Timestamp("2020-03-31 03:00:00"), "10.100.1.1"),
    ]
    expected_columns = list(
        itertools.product(
            ["America", "Europe", "Japan"],
            ["America", "Europe", "Japan"],
            [
                "sid_101",
                "sid_103",
                "sid_102",
                "dest_port_1",
                "dest_port_3",
                "dest_port_2",
            ],
        )
    )
    expected_feature = scipy.sparse.lil_matrix((14, 54))
    expected_feature[0, 48] = 8.12415060330663
    expected_feature[0, 51] = 8.12415060330663
    for i in range(1, 11):
        for j in [26, 29]:
            expected_feature[i, j] = 0.8754687373538999
    expected_feature[11, 13] = 1.7346010553881064
    expected_feature[11, 16] = 2.0794415416798357
    expected_feature[12, 7] = 1.7346010553881064
    expected_feature[12, 10] = 2.0794415416798357
    expected_feature[13, 1] = 1.7346010553881064
    expected_feature = expected_feature.tocsr()

    fl = feature_extraction_all(
        log=log_for_fea, idf_config=config_for_fea.idf, iptable=iptable_for_fea
    )

    assert fl.idf_sid.equals(expected_idf_sid)
    assert fl.idf_dport.equals(expected_idf_dport)
    assert fl.index == expected_index
    assert fl.columns == expected_columns
    assert (fl.feature - expected_feature).nnz == 0
