"""IO-related internally used utilities."""

import logging

import pandas

logger = logging.getLogger(__name__)


def load_csv_optional_zip(base_file_name: str, **read_csv_kwargs) -> pandas.DataFrame:
    """Load base_file_name.csv.zip if exists; base_file_name.csv otherwise

    Parameters
    ----------
    base_file_name
        file name without extension.
    read_csv_kwargs
        kwargs passed to pandas.read_csv
    """
    logger.info(base_file_name)
    try:
        return pandas.read_csv(base_file_name + ".csv.zip", **read_csv_kwargs)
    except FileNotFoundError:
        pass
    return pandas.read_csv(base_file_name + ".csv", **read_csv_kwargs)
