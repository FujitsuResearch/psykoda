import copy
import datetime
import os
import random
import shutil
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import scipy

import psykoda.detection
import psykoda.utils
from psykoda.cli import internal
from psykoda.feature_extraction import FeatureExtractionConfig, FeatureLabel, IDFConfig
from psykoda.io.labeled.file import (
    FileLoader,
    FileSaver,
    FileStorageBaseConfig,
    FileStorageConfig,
)

rsrc_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "rsrc", "test_internal"
)


def gen_nonexistent_path(parent_dir: str = ""):
    retry_max = 100
    for _ in range(retry_max):
        random_text = hex(random.getrandbits(128))
        path = os.path.join(parent_dir, random_text)
        if not os.path.exists(path):
            return path
    raise RuntimeError


@pytest.fixture
def fixture_report_transfer():
    # -- Directory Tree --
    # rsrc_dir [DIR]
    #   report_transfer [DIR]  <- REPORTFILE_DIR
    #     reportfile.txt [FILE]  <- REPORTFILE_PATH, REPORTFILE_NAME(basename)
    #     subdir [DIR]  <- REPORTFILE_SUBDIR
    REPORTFILE_NAME = "reportfile.txt"
    REPORTFILE_DIR = os.path.join(rsrc_dir, "report_transfer")
    REPORTFILE_PATH = os.path.join(REPORTFILE_DIR, REPORTFILE_NAME)
    REPORTFILE_SUBDIR = os.path.join(REPORTFILE_DIR, "subdir")

    def cleanup():
        # Cleen up the directory
        if os.path.isdir(REPORTFILE_DIR):
            shutil.rmtree(REPORTFILE_DIR)

        # Create the report file and directories
        os.makedirs(REPORTFILE_SUBDIR, exist_ok=True)
        with open(REPORTFILE_PATH, "w", encoding="utf_8") as file:
            file.write("abc")

    cleanup()
    yield {
        "reportfile_name": REPORTFILE_NAME,
        "reportfile_dir": REPORTFILE_DIR,
        "reportfile_path": REPORTFILE_PATH,
        "reportfile_subdir": REPORTFILE_SUBDIR,
    }
    cleanup()


@pytest.fixture
def fixture_report_all():
    PARENT_DIR = os.path.join(rsrc_dir, "report_all")
    ANOMALY_DIR = os.path.join(PARENT_DIR, "anomaly_found")
    ANOMALY_PATH = os.path.join(ANOMALY_DIR, "stats.json")
    NOT_ANOMALY_DIR = os.path.join(PARENT_DIR, "anomaly_not_found")
    NOT_ANOMALY_PATH = os.path.join(NOT_ANOMALY_DIR, "stats.json")
    DSTFILE_DIR = os.path.join(PARENT_DIR, "save")
    DSTFILE_PATH = os.path.join(DSTFILE_DIR, "report.csv")

    def cleanup():
        # Remove the directory and create it.
        if os.path.isdir(DSTFILE_DIR):
            shutil.rmtree(DSTFILE_DIR)
        os.makedirs(DSTFILE_DIR, exist_ok=True)

    cleanup()
    yield {
        "anomaly_dir": ANOMALY_DIR,
        "anomaly_path": ANOMALY_PATH,
        "not_anomaly_dir": NOT_ANOMALY_DIR,
        "not_anomaly_path": NOT_ANOMALY_PATH,
        "dstfile_dir": DSTFILE_DIR,
        "dstfile_path": DSTFILE_PATH,
    }
    cleanup()


def test_report_transfer_01(fixture_report_transfer):
    subdir = fixture_report_transfer["reportfile_subdir"]
    filename = fixture_report_transfer["reportfile_name"]
    filepath = fixture_report_transfer["reportfile_path"]

    internal.report_transfer(filepath, subdir)
    expected_filepath = os.path.join(subdir, filename)
    assert os.path.isfile(expected_filepath)
    with open(expected_filepath, "r", encoding="utf_8") as file:
        data = file.read()
        assert data == "abc"


def test_report_transfer_02(fixture_report_transfer):
    parent_dir = fixture_report_transfer["reportfile_dir"]
    filename = fixture_report_transfer["reportfile_name"]
    filepath = fixture_report_transfer["reportfile_path"]

    nonexistent_dir = gen_nonexistent_path(parent_dir)
    nonexistent_subdir = gen_nonexistent_path(nonexistent_dir)

    internal.report_transfer(filepath, nonexistent_subdir)
    expected_filepath = os.path.join(nonexistent_subdir, filename)
    assert os.path.isfile(expected_filepath)
    with open(expected_filepath, "r", encoding="utf_8") as file:
        data = file.read()
        assert data == "abc"


def test_report_transfer_03(fixture_report_transfer):
    parent_dir = fixture_report_transfer["reportfile_dir"]
    subdir = fixture_report_transfer["reportfile_subdir"]
    filepath = fixture_report_transfer["reportfile_path"]

    with pytest.raises(TypeError):
        internal.report_transfer(filepath, None)

    nonexistent_filepath = gen_nonexistent_path(parent_dir)
    with pytest.raises(TypeError):
        internal.report_transfer(nonexistent_filepath, subdir)


def test_report_all_01(fixture_report_all):
    path_list_stast = [fixture_report_all["anomaly_path"]]
    path_save = fixture_report_all["dstfile_path"]
    src_dir = fixture_report_all["anomaly_dir"]

    internal.report_all(path_list_stast, path_save)

    assert os.path.isfile(path_save)

    df = pd.read_csv(path_save)
    expected_df = pd.read_csv(os.path.join(src_dir, "expected_report.csv"))
    assert df.equals(expected_df)


def test_report_all_02(fixture_report_all):
    path_list_stast = [fixture_report_all["not_anomaly_path"]]
    path_save = fixture_report_all["dstfile_path"]
    src_dir = fixture_report_all["not_anomaly_dir"]

    internal.report_all(path_list_stast, path_save)

    assert os.path.isfile(path_save)

    df = pd.read_csv(path_save)
    expected_df = pd.read_csv(os.path.join(src_dir, "expected_report.csv"))
    assert df.equals(expected_df)


@dataclass
class CommandLineArgsForTests:
    subnet: Optional[str]
    service: Optional[str]
    period_train: Optional[int]
    date_from_training: Optional[datetime.datetime]
    date_to_training: Optional[datetime.datetime]
    no_plot: Optional[bool]
    date_from: Optional[datetime.datetime]
    date_to: Optional[datetime.datetime]
    debug: Optional[bool]


@pytest.fixture
def fixture_main_detection_01():
    ARGS = CommandLineArgsForTests(
        subnet="subnetdummy",
        service="servicedummy",
        period_train=1,
        date_from_training=pd.Timestamp("2021-06-01 01:00:00"),
        date_to_training=pd.Timestamp("2021-06-03 03:00:00"),
        date_from=pd.Timestamp("2021-06-04 03:00:00"),
        date_to=pd.Timestamp("2021-06-06 06:00:00"),
        no_plot=False,
        debug=False,
    )
    PARENT_DIR = os.path.join(rsrc_dir, "main_detection")
    REPORT_DIR = os.path.join(PARENT_DIR, ARGS.subnet, ARGS.service)

    def cleanup():
        # Cleen up the directory
        if os.path.isdir(REPORT_DIR):
            shutil.rmtree(os.path.join(PARENT_DIR, ARGS.subnet))

    cleanup()
    yield {"args": ARGS, "parent_dir": PARENT_DIR, "report_dir": REPORT_DIR}
    cleanup()


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
        (pd.Timestamp("2021-06-01 01:00:00"), "1.1.1.1"),
        (pd.Timestamp("2021-06-02 02:00:00"), "2.2.2.2"),
        (pd.Timestamp("2021-06-03 03:00:00"), "3.3.3.3"),
        (pd.Timestamp("2021-06-04 04:00:00"), "4.4.4.4"),
        (pd.Timestamp("2021-06-05 05:00:00"), "5.5.5.5"),
        (pd.Timestamp("2021-06-06 06:00:00"), "6.6.6.6"),
    ]
    columns = ["columns_%d" % i for i in range(4)]
    label = np.array(range(feature.shape[0]))
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


def test_main_detection_01(fixture_main_detection_01):
    """main_detection does nothing if log and label are empty"""
    config = internal.DetectConfig(
        arguments=None,
        detection_units=None,
        preprocess=None,
        io=internal.IOConfig(
            input=None,
            previous=None,
            output=internal.OutputConfig(
                dir=None, share_dir=None, subdir=fixture_main_detection_01["report_dir"]
            ),
        ),
        feature_extraction=None,
        anomaly_detection=None,
    )
    args = fixture_main_detection_01["args"]
    log = pd.DataFrame()
    label = pd.Series()

    expected_ret = None
    expected_dirpath = os.path.join(config.io.output.subdir, args.subnet, args.service)

    actual_ret = internal.main_detection(args, config, log, label)

    assert actual_ret == expected_ret
    assert os.path.isdir(expected_dirpath)
    assert not os.path.isfile(os.path.join(expected_dirpath, internal.FILENAME_STATS))


def test_main_detection_prepare_data_01(fixture_main_detection_01):
    """main_detection_prepare_data does nothing if log and label are empty"""
    args = fixture_main_detection_01["args"]
    log = pd.DataFrame()

    expected_ret = None

    actual_ret = internal.main_detection_prepare_data(args, None, log, None)

    assert actual_ret == expected_ret


def test_main_detection_prepare_data_02(fixture_main_detection_01, monkeypatch):
    """main_detection_prepare_data does nothing if log and label are empty"""
    args = fixture_main_detection_01["args"]
    config = FeatureExtractionConfig(idf={}, address_to_location=None)
    log = pd.DataFrame(["dummy"])

    expected_ret = None

    monkeypatch.setattr("pandas.read_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "psykoda.feature_extraction.feature_extraction_all",
        lambda *args, **kwargs: None,
    )
    actual_ret = internal.main_detection_prepare_data(args, config, log, None)

    assert actual_ret == expected_ret


def test_main_detection_prepare_data_03(
    fixture_main_detection_01, featurelabel_class, monkeypatch
):
    """main_detection_prepare_data returns feature_label"""
    args = fixture_main_detection_01["args"]
    config = FeatureExtractionConfig(idf={}, address_to_location=None)
    log = pd.DataFrame(["dummy"])
    label = pd.Series(
        [1, 1], index=[featurelabel_class.index[2], featurelabel_class.index[4]]
    )
    tmp_fl = copy.deepcopy(featurelabel_class)
    tmp_fl.extract_nonzeros()

    expected_feature = tmp_fl.feature / tmp_fl.feature.max()
    expected_label = np.array([0.0, 0.0, 0.0, 1.0])

    monkeypatch.setattr("pandas.read_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "psykoda.feature_extraction.feature_extraction_all",
        lambda *args, **kwargs: featurelabel_class,
    )

    actual_ret = internal.main_detection_prepare_data(args, config, log, label)

    assert (actual_ret.feature - expected_feature).nnz == 0
    assert all(actual_ret.label == expected_label)


def test_main_detection_after_prepare_data_01(
    fixture_main_detection_01, featurelabel_class
):
    """main_detection_after_prepare_data splits data and construct x_train_labeled"""
    args = fixture_main_detection_01["args"]
    label = pd.Series(
        index=[
            featurelabel_class.index[2],
            featurelabel_class.index[3],
            featurelabel_class.index[4],
        ]
    )

    tmp_fl = copy.deepcopy(featurelabel_class)
    # train_test_splitted, x_train_labeled
    expected_tts = tmp_fl.split_train_test(args.date_to_training)
    expected_xtl = tmp_fl.feature[[2, 3, 4]]

    actual_tts, actual_xtl = internal.main_detection_after_prepare_data(
        args, label, featurelabel_class
    )

    assert (actual_tts[0] - expected_tts[0]).nnz == 0
    assert all(actual_tts[1] == expected_tts[1])
    assert (actual_tts[2] - expected_tts[2]).nnz == 0
    assert all(actual_tts[3] == expected_tts[3])
    assert (actual_xtl - expected_xtl).nnz == 0


def test_output_result_01(fixture_main_detection_01, monkeypatch):
    """output_result emits files (plot is excluded from test)"""
    args = fixture_main_detection_01["args"]
    log = pd.DataFrame(
        [
            [pd.Timestamp("2021-06-01 01:00:00"), "1.1.1.1", 1, 11],
            [pd.Timestamp("2021-06-01 02:00:00"), "2.2.2.2", 2, 22],
            [pd.Timestamp("2021-06-01 03:00:00"), "3.3.3.3", 3, 33],
        ],
        columns=["datetime_rounded", "src_ip", "column_1", "column_2"],
    ).set_index(["datetime_rounded", "src_ip"])
    label = pd.Series()
    dir_report = fixture_main_detection_01["report_dir"]
    x_train_labeled_embeddings = None
    x_test_embeddings = None
    idx_anomaly = [2, 3, 4]
    shap_value_idx_sorted = pd.DataFrame(
        [
            [pd.Timestamp("2021-06-01 01:00:00"), "1.1.1.1", 1],
            [pd.Timestamp("2021-06-01 02:00:00"), "2.2.2.2", 2],
            [pd.Timestamp("2021-06-01 03:00:00"), "3.3.3.3", 3],
        ],
        columns=["datetime_rounded", "src_ip", "column_1"],
    ).set_index(["datetime_rounded", "src_ip"])
    anomaly_score_sorted = shap_value_idx_sorted["column_1"]
    stats = {}
    previous_config = FileStorageConfig(
        base=FileStorageBaseConfig(
            dir=os.path.join(fixture_main_detection_01["parent_dir"], "write_log")
        ),
        load=FileLoader.Config(),
        save=FileSaver.Config(compression=True, all=False),
    )

    os.makedirs(dir_report, exist_ok=True)

    # skip test for plot_detection
    monkeypatch.setattr(
        "psykoda.io.reporting.plot.plot_detection", lambda *args, **kwargs: None
    )

    expected_name_anomaly = [
        (pd.Timestamp("2021-06-01 01:00:00"), "1.1.1.1"),
        (pd.Timestamp("2021-06-01 02:00:00"), "2.2.2.2"),
        (pd.Timestamp("2021-06-01 03:00:00"), "3.3.3.3"),
    ]

    actual_ret = internal.output_result(
        args=args,
        log=log,
        label=label,
        dir_report=dir_report,
        x_train_labeled_embeddings=x_train_labeled_embeddings,
        x_test_embeddings=x_test_embeddings,
        idx_anomaly=idx_anomaly,
        shap_value_idx_sorted=shap_value_idx_sorted,
        anomaly_score_sorted=anomaly_score_sorted,
        stats=stats,
        previous_config=previous_config,
    )

    try:
        assert os.path.isfile(os.path.join(dir_report, internal.FILENAME_REPORT))
        assert os.path.isdir(previous_config.base.dir)
        for dt, src_ip in shap_value_idx_sorted.index:
            assert os.path.isfile(
                os.path.join(
                    previous_config.base.dir, dt.strftime(f"%Y-%m-%d-%H__{src_ip}.zip")
                )
            )
        assert actual_ret["num_anomaly"] == 3
        assert actual_ret["name_anomaly"] == expected_name_anomaly
    finally:
        if os.path.isdir(previous_config.base.dir):
            shutil.rmtree(previous_config.base.dir)


@pytest.mark.parametrize(
    "train_test_splitted", [(None, [0], None, [0, 0, 0]), (None, [0, 0, 0], None, [0])]
)
def test_main_detection_skip_or_detect_01(
    fixture_main_detection_01, train_test_splitted
):
    """main_detection_skip_or_detect skips when required_srcip is not satisfied"""
    args = fixture_main_detection_01["args"]
    anomaly_detection_config = internal.AnomalyDetectionConfig(
        required_srcip=internal.SkipDetectionConfig(train=2, test=2),
        deepsad=None,
        train=None,
        threshold=None,
    )

    actual_ret = internal.main_detection_skip_or_detect(
        args=args,
        log=None,
        label=pd.Series(),
        dir_report=None,
        feature_label=None,
        train_test_splitted=train_test_splitted,
        x_train_labeled=scipy.sparse.csr_matrix([]),
        anomaly_detection_config=anomaly_detection_config,
        previous_config=None,
    )

    assert "skipped" in actual_ret


@pytest.fixture
def fixture_main_detection_02():
    args = CommandLineArgsForTests(
        date_from=datetime.datetime(2020, 4, 1, 0, 0),
        date_from_training=datetime.datetime(2020, 3, 4, 0, 0),
        date_to=datetime.datetime(2021, 7, 12, 13, 39, 15, 303669),
        date_to_training=datetime.datetime(2020, 3, 31, 0, 0),
        period_train=28,
        debug=False,
        no_plot=True,
        service="ALL_but_SSH",
        subnet="ALL",
    )

    PARENT_DIR = os.path.join(rsrc_dir, "main_detection")

    log = pd.read_csv(
        os.path.join(PARENT_DIR, "sample_log.csv"),
        parse_dates=["datetime_rounded", "datetime_full"],
    ).set_index(["datetime_rounded", "src_ip"])

    label = pd.read_csv(
        os.path.join(PARENT_DIR, "sample_label.csv"), parse_dates=["datetime_rounded"]
    ).set_index(["datetime_rounded", "src_ip"])["0"]

    config = internal.DetectConfig(
        arguments=None,
        detection_units=None,
        io=internal.IOConfig(
            input=None,
            previous=internal.PreviousConfig(
                load=None,
                log=FileStorageConfig(
                    base=FileStorageBaseConfig(
                        dir=os.path.join(PARENT_DIR, "labeled_dir"),
                    ),
                    save=FileSaver.Config(),
                    load=FileLoader.Config(),
                ),
            ),
            output=internal.OutputConfig(
                dir=None, share_dir=None, subdir=os.path.join(PARENT_DIR, "sub_dir")
            ),
        ),
        preprocess=None,
        feature_extraction=FeatureExtractionConfig(
            idf={
                "sid": IDFConfig(min_count=1, num_feature=30),
                "dest_port": IDFConfig(min_count=1, num_feature=30),
            },
            address_to_location=os.path.join(PARENT_DIR, "sample_IPtable.csv"),
        ),
        anomaly_detection=internal.AnomalyDetectionConfig(
            required_srcip=internal.SkipDetectionConfig(train=3, test=5),
            deepsad=psykoda.detection.DeepSAD.Config(
                dim_hidden=[4, 4, 4, 4, 2],
                eta=16,
                lam=1e-06,
                path_pretrained_model=None,
            ),
            train=psykoda.detection.DeepSAD.TrainConfig(
                epochs_pretrain=30, epochs_train=100, learning_rate=0.001, batch_size=64
            ),
            threshold=internal.ThresholdConfig(num_anomaly=5, min_score=10),
        ),
    )

    dir_report = os.path.join(config.io.output.subdir, args.subnet, args.service)

    def cleanup():
        if os.path.isdir(config.io.output.subdir):
            shutil.rmtree(config.io.output.subdir)
        if os.path.isdir(config.io.previous.log.base.dir):
            shutil.rmtree(config.io.previous.log.base.dir)

    cleanup()
    yield {
        "parent_dir": PARENT_DIR,
        "dir_report": dir_report,
        "args": args,
        "log": log,
        "label": label,
        "config": config,
    }
    cleanup()


def test_main_detection_skip_or_detect_02(fixture_main_detection_02):
    """main_detection_skip_or_detect performs detection"""
    parent_dir = fixture_main_detection_02["parent_dir"]
    args = fixture_main_detection_02["args"]
    log = fixture_main_detection_02["log"]
    label = fixture_main_detection_02["label"]
    dir_report = fixture_main_detection_02["dir_report"]
    fl_columns_df = pd.read_csv(
        os.path.join(parent_dir, "sample_feature_label_columns.csv"), header=None
    )
    fl_columns = []
    for items in fl_columns_df.to_numpy().tolist():
        fl_columns.append(tuple(items))
    feature_label = FeatureLabel(
        feature=scipy.sparse.csr_matrix([[0] * 11]),
        index=[0],
        columns=fl_columns,
        idf_sid=None,
        idf_dport=None,
    )
    tts_0 = scipy.sparse.csr_matrix(
        pd.read_csv(
            os.path.join(parent_dir, "sample_train_test_splitted[0].csv"), header=None
        )
    )
    tts_1 = np.loadtxt(
        os.path.join(parent_dir, "sample_train_test_splitted[1].csv"), dtype="float64"
    )
    tts_2 = scipy.sparse.csr_matrix(
        pd.read_csv(
            os.path.join(parent_dir, "sample_train_test_splitted[2].csv"), header=None
        )
    )
    tts_3_df = pd.read_csv(
        os.path.join(parent_dir, "sample_train_test_splitted[3].csv"),
        header=None,
        parse_dates=[0],
    ).set_index([0, 1])
    tts_3 = tts_3_df.index
    train_test_splitted = (tts_0, tts_1, tts_2, tts_3)
    x_train_labeled = scipy.sparse.csr_matrix(
        pd.read_csv(os.path.join(parent_dir, "sample_x_train_labeled.csv"), header=None)
    )
    config = fixture_main_detection_02["config"]
    anomaly_detection_config = config.anomaly_detection
    previous_config = config.io.previous.log

    os.makedirs(dir_report, exist_ok=True)

    actual_ret = internal.main_detection_skip_or_detect(
        args=args,
        log=log,
        label=label,
        dir_report=dir_report,
        feature_label=feature_label,
        train_test_splitted=train_test_splitted,
        x_train_labeled=x_train_labeled,
        anomaly_detection_config=anomaly_detection_config,
        previous_config=previous_config,
    )

    assert os.path.isdir(previous_config.base.dir)
    assert os.path.isfile(os.path.join(dir_report, internal.FILENAME_REPORT))
    assert "skipped" not in actual_ret

    # Check only for the existence of the key.
    # The values corresponding to these keys are not constant.
    assert "num_anomaly" in actual_ret
    assert "name_anomaly" in actual_ret


def test_main_detection_02(fixture_main_detection_02):
    """main_detection performs detection"""
    dir_report = fixture_main_detection_02["dir_report"]
    report_path = os.path.join(dir_report, internal.FILENAME_STATS)

    args = fixture_main_detection_02["args"]
    config: internal.DetectConfig = fixture_main_detection_02["config"]
    log = fixture_main_detection_02["log"]
    label = fixture_main_detection_02["label"]

    internal.main_detection(args, config, log, label)

    assert os.path.isfile(report_path)
    actual_stats = psykoda.utils.load_json(report_path)
    assert "skipped" not in actual_stats
    # Check only for the existence of the key.
    # The values corresponding to these keys are not constant.
    assert "num_anomaly" in actual_stats
    assert "name_anomaly" in actual_stats
