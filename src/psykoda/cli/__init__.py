"""Command line interface"""
import argparse
import copy
import dataclasses
import datetime
import logging
from typing import Optional

import dacite

from psykoda.cli import internal
from psykoda.constants import COMMANDLINE_DATE_FORMAT

logger = logging.getLogger(__name__)


Config = internal.DetectConfig


def main():
    """
    Parse command line arguments and call main routine.

    Raises
    ------
    ValueError
        Command line arguments are invalid.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        help="Path to configuration file",
        default="config.json",
    )
    ap.add_argument("--debug", action="store_true", help="Output debug logs")
    ap.add_argument(
        "--date_from", type=strptime, help="Start date of the period to be detected"
    )
    ap.add_argument(
        "--date_to", type=strptime, help="End date of the period to be detected"
    )
    ap.add_argument(
        "--period_train", type=int, default=28, help="Number of days of training period"
    )
    ap.add_argument("--nrows-read", type=int)
    ap.add_argument(
        "--no_plot", action="store_true", help="Do not display result graphs"
    )

    args = ap.parse_args()
    internal.configure_logging(debug=args.debug)
    logger.debug("args %s\n", args)
    config: Config = dacite.from_dict(Config, internal.load_config(args.config))
    logger.debug("configuration %s", config)
    main_detect(args, config)


def main_detect(args, config: Config):
    """
    Main routine for anmaly detection.

    Parameters
    ----------
    args
        Command line arguments.
    config
        Settings for this command.
    """
    # Generate parameters from command line arguments
    params = internal.set_default_date_detect(copy.deepcopy(args), config.arguments)
    logger.debug("params %s\n", params)
    internal.main_preproc_and_detection(params, config)
    return logger.info("finish")


def strptime(date_string: str):
    """
    Convert a string to datetime in COMMANDLINE_DATE_FORMAT format.

    Parameters
    ----------
    date_string
        Date string

    Returns
    -------
    datetime
    """
    return datetime.datetime.strptime(date_string, COMMANDLINE_DATE_FORMAT)
