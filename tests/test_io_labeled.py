from datetime import datetime
from os import path, remove

import pandas
import pytest

from psykoda.io.labeled.file import FileLoader, FileSaver, FileStorageBaseConfig

rsrc_dir = path.join(
    path.dirname(path.abspath(__file__)),
    "rsrc",
    __name__.replace("tests.", ""),
)


@pytest.fixture
def loader():
    return FileLoader(
        base_config=FileStorageBaseConfig(
            dir=path.join(rsrc_dir, "1"),
        ),
        config=FileLoader.Config(),
    )


@pytest.fixture
def saver():
    return FileSaver(
        base_config=FileStorageBaseConfig(
            dir=path.join(rsrc_dir, "1"),
        ),
        config=FileSaver.Config(),
    )


def test_io_fujitsu_splunk_load_labeled(loader: FileLoader):
    df = loader._load_previous_log(datetime(2021, 4, 5, 6, 7, 8), "10.20.30.40")
    assert list(df.index.get_level_values("datetime_rounded")) == [
        datetime(2021, 4, 5, 6),
    ]
    assert list(df.index.get_level_values("src_ip")) == ["10.20.30.40"]
    for name, values in {
        "datetime_full": [datetime(2021, 4, 5, 6, 7, 8)],
        "dest_ip": ["10.11.12.13"],
        "dest_port": [67],
        "sid": [234444],
    }.items():
        assert list(df[name]) == values


def test_io_fujitsu_splunk_save_labeled(saver: FileSaver):
    df = pandas.DataFrame(
        {
            "datetime_rounded": [
                datetime(2021, 4, 6, 8),
                datetime(2021, 4, 6, 8),
                datetime(2021, 4, 6, 10),
            ],
            "extra_column": [
                "is",
                "preserved",
                "as is",
            ],
            "src_ip": [
                "10.20.40.88",
                "10.20.40.80",
                "10.20.40.80",
            ],
            "datetime_full": [
                datetime(2021, 4, 6, 8, 10, 12),
                datetime(2021, 4, 6, 8, 11, 14),
                datetime(2021, 4, 6, 10, 14, 18),
            ],
            "dest_ip": [
                "10.12.14.16",
                "10.13.16.19",
                "10.14.18.22",
            ],
            "dest_port": [77] * 3,
            "sid": [222234] * 3,
        }
    ).set_index(["datetime_rounded", "src_ip"])
    expected = {}
    for save_anomaly_log_all, expected in {
        False: """datetime_rounded,src_ip,extra_column,datetime_full,dest_ip,dest_port,sid
2021-04-06 08:00:00,10.20.40.80,preserved,2021-04-06 08:11:14,10.13.16.19,77,222234
""",
        True: """datetime_rounded,src_ip,extra_column,datetime_full,dest_ip,dest_port,sid
2021-04-06 08:00:00,10.20.40.80,preserved,2021-04-06 08:11:14,10.13.16.19,77,222234
2021-04-06 10:00:00,10.20.40.80,as is,2021-04-06 10:14:18,10.14.18.22,77,222234
""",
    }.items():
        saver.config.all = save_anomaly_log_all
        path_written = saver._save_previous_log(
            df,
            datetime(2021, 4, 6, 8),
            "10.20.40.80",
        )
        content = open(path_written, encoding="utf_8").read()
        print(content)
        assert content == expected
        remove(path_written)
