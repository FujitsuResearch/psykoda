#%load_ext autoreload
#%autoreload 2

import dataclasses
import glob
import logging
import os
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix

from psykoda import detection, feature_extraction, preprocess, utils
from psykoda.constants import COMMANDLINE_DATE_FORMAT, col
from psykoda.io import labeled, reporting

logger = logging.getLogger(__name__)
to_stderr = {"_log_err": True}

FILENAME_WEIGHT = "best_weight.h5"
FILENAME_IDF_SID = "idf_sid.csv"
FILENAME_IDF_DPORT = "idf_dport.csv"
FILENAME_PLOT_DETECTION = "plot_detection.png"
FILENAME_STATS = "stats.json"
FILENAME_REPORT = "report.csv"
FILENAME_FEATURE_MATRIX = "feature_matrix.csv"


def configure_logging(debug: bool):
    """
    Configure execution log settings.

    Parameters
    ----------
    debug
        Whether to log "debug levels".
    """

    PATH_LOG = "./log/log_" + datetime.strftime(datetime.today(), "%Y-%m-%d") + ".log"
    os.makedirs(os.path.dirname(PATH_LOG), exist_ok=True)
    log_level = logging.INFO
    if debug:
        log_level = logging.DEBUG

    # utilities
    stderr_filter = lambda record: getattr(record, "_log_err", False)

    # app config
    stderr_handler = logging.StreamHandler()
    stderr_handler.addFilter(stderr_filter)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(logging.Formatter("%(message)s"))
    handlers = [stderr_handler]

    logfile_handler = logging.FileHandler(PATH_LOG)
    logfile_handler.setLevel(logging.DEBUG)
    logfile_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(module)s # %(funcName)s line %(lineno)d] %(message)s"
        )
    )
    handlers.append(logfile_handler)

    logging.basicConfig(handlers=handlers, level=log_level)


class Incomplete_Args_Exception(Exception):
    pass


load_config = utils.load_json


@dataclass
class OutputConfig:
    dir: str
    share_dir: Optional[str]
    subdir: Optional[str]


@dataclass
class PreprocessConfig:
    exclude_lists: Optional[str]
    screening: preprocess.ScreeningConfig


@dataclass
class InputConfig:
    dir: str


@dataclasses.dataclass
class LoadPreviousConfigItem:
    list: Optional[str]
    ndate: int = 730


@dataclasses.dataclass
class LoadPreviousConfig:
    """
    Log loading settings.

    Parameters
    ----------
    list
        path to CSV file in which labeled IP addresses are listed
    ndate
        time range for labeled IP addresses, in days
    """

    known_normal: Optional[LoadPreviousConfigItem]
    known_anomaly: Optional[LoadPreviousConfigItem]
    unknown: Optional[LoadPreviousConfigItem]


@dataclass
class PreviousConfig:
    load: LoadPreviousConfig
    log: labeled.Config


@dataclass
class IOConfig:
    input: InputConfig
    previous: PreviousConfig
    output: OutputConfig


@dataclass
class Service:
    """Service definition: set of destination port numbers

    Examples
    --------
    >>> all = Service()
    >>> ssh = Service(include=[22])
    >>> all_but_ssh = Service(exclude=[22])
    >>> ssh_or_https = Service(include=[22, 443])
    """

    include: Optional[List[int]]
    exclude: Optional[List[int]]


@dataclass
class Subnet:
    """Subnet configuration: set of CIDR-formatted IP addresses with services to analyze

    Examples
    --------
    >>> private_A = Subnet(["10.0.0.0/8"], get_names_of_services_from_config())
    >>> private = Subnet(["private-A", "private-B", "private-C"], get_names_of_services_from_config())  # these constants are available for convenience and readability
    >>> my_network = Subnet(["10.0.0.0/16", "10.1.1.0/24"], get_names_of_services_from_config())
    """

    cidrs: List[str]
    services: List[str]


@dataclass
class DetectionUnitConfig:
    """Detection unit configuration

    Parameters
    ----------
    services
        map from names of service to service definitions
    subnets
        map from names of subnet to subnet configurations
    """

    services: Dict[str, Service]
    subnets: Dict[str, Subnet]


@dataclass
class TargetPeriod:
    days: int = 30


@dataclass
class ArgumentsConfig:
    """Arguments modification configuration

    Parameters
    ----------
    target_period:
        default target period used to determine date_from and date_to values if missing.
    """

    target_period: TargetPeriod


def set_default_date_detect(args, config: ArgumentsConfig):
    """
    Configure training from/to dates according to args and config.

    Parameters
    ----------
    args
        Command line args.
    config
        Settings for arguments.

    Returns
    -------
    args
        Command line args with training from/to dates added.
    """
    date_time_today = datetime.today()

    if args.date_from is None:
        args.date_from = date_time_today - timedelta(config.target_period.days)

    if args.date_to is None:
        args.date_to = date_time_today - timedelta(1)

    args.date_from_training = args.date_from - timedelta(args.period_train)
    args.date_to_training = args.date_from - timedelta(1)

    return args


@dataclass
class SkipDetectionConfig:
    train: int
    test: int


@dataclass
class ThresholdConfig:
    num_anomaly: int
    min_score: float


@dataclass
class AnomalyDetectionConfig:
    required_srcip: SkipDetectionConfig
    deepsad: detection.DeepSAD.Config
    train: detection.DeepSAD.TrainConfig
    threshold: ThresholdConfig


@dataclasses.dataclass
class DetectConfig:
    arguments: ArgumentsConfig
    detection_units: DetectionUnitConfig
    io: IOConfig
    preprocess: PreprocessConfig
    feature_extraction: feature_extraction.FeatureExtractionConfig
    anomaly_detection: AnomalyDetectionConfig


def main_detection(args, config: DetectConfig, log: pd.DataFrame, label: pd.Series):
    """

    Parameters
    ----------
    args
    config
    log
        :index:
        :columns:
    label
        filled with 1
        :index:
    """

    dir_report = os.path.join(config.io.output.subdir, args.subnet, args.service)
    os.makedirs(dir_report, exist_ok=True)
    feature_label = main_detection_prepare_data(
        args, config.feature_extraction, log, label
    )
    if feature_label is None:
        return
    feature_label.idf_sid.to_csv(os.path.join(dir_report, FILENAME_IDF_SID))
    feature_label.idf_dport.to_csv(os.path.join(dir_report, FILENAME_IDF_DPORT))
    train_test_splitted, x_train_labeled = main_detection_after_prepare_data(
        args, label, feature_label
    )
    stats = main_detection_skip_or_detect(
        args,
        log,
        label,
        dir_report,
        feature_label,
        train_test_splitted,
        x_train_labeled,
        anomaly_detection_config=config.anomaly_detection,
        previous_config=config.io.previous.log,
    )
    utils.save_json(stats, path=os.path.join(dir_report, FILENAME_STATS))


def main_detection_prepare_data(
    args,
    config: feature_extraction.FeatureExtractionConfig,
    log: pd.DataFrame,
    label: pd.Series,
) -> Optional[feature_extraction.FeatureLabel]:
    """Feature extraction"""

    logger.info("start detect on subnet %s and service %s", args.subnet, args.service)

    if len(log) == 0:
        logger.info("skip analysis; no logs exist")
        return None

    logger.info("extracting features")
    feature_label = feature_extraction.feature_extraction_all(
        log=log,
        iptable=pd.read_csv(config.address_to_location),
        idf_config=config.idf,
    )
    if feature_label is None:
        logger.info("skip analysis; feature matrix is None")
        return None

    feature_label.extract_nonzeros()
    label = label.loc[label.index & feature_label.index]
    feature_label.put_labels(labeled_samples=label)

    feature_label.feature = feature_label.feature / feature_label.feature.max()

    return feature_label


def main_detection_after_prepare_data(
    args, label: pd.Series, feature_label: feature_extraction.FeatureLabel
):
    """Split data and construct labeled training feature."""
    train_test_splitted = feature_label.split_train_test(args.date_to_training)
    idx_labeled = [
        feature_label.index.index(sample)
        for sample in label.index
        if sample in feature_label.index
    ]
    x_train_labeled = feature_label.feature[idx_labeled]
    return train_test_splitted, x_train_labeled


def main_detection_skip_or_detect(
    args,
    log: pd.DataFrame,
    label: pd.Series,
    dir_report: str,
    feature_label: feature_extraction.FeatureLabel,
    train_test_splitted,
    x_train_labeled: csr_matrix,
    anomaly_detection_config: AnomalyDetectionConfig,
    previous_config: labeled.Config,
) -> dict:
    """Anomaly detection and output the result."""
    x_train, y_train, x_test, index_test = train_test_splitted

    stats = {
        "subnet": args.subnet,
        "service": args.service,
        "date_from": args.date_from,
        "date_to": args.date_to,
        "num_samples_st_detection": len(index_test),
        "num_samples_training": len(y_train),
        "date_from_training": args.date_from_training,
        "date_to_training": args.date_to_training,
        "num_samples_labeled": x_train_labeled.shape[0],
        "samples_labeled": label.index.tolist(),
    }
    logger.info("stats: %s", stats)

    if len(y_train) < anomaly_detection_config.required_srcip.train:
        skip_message = f"#src_ip[train] = {len(y_train)} < config.anomaly_detection.required_srcip.train = {anomaly_detection_config.required_srcip.train}"
        logger.info(skip_message)
        stats["skipped"] = skip_message
        return stats

    if len(index_test) < anomaly_detection_config.required_srcip.test:
        skip_message = f"#src_ip[test] = {len(index_test)} < config.anomaly_detection.required_srcip.test = {anomaly_detection_config.required_srcip.test}"
        logger.info(skip_message)
        stats["skipped"] = skip_message
        return stats

    logger.info("training detector")
    verbose = 1 if logger.root.level < 20 else 0
    detector = detection.DeepSAD(anomaly_detection_config.deepsad)
    detector.train(
        X=x_train,
        y=y_train,
        path_model=os.path.join(dir_report, FILENAME_WEIGHT),
        config=anomaly_detection_config.train,
        verbose=verbose,
    )

    logger.info("outputting detection reports")
    anomaly_score = detector.compute_anomaly_score(x_test, scale=True)
    num_anomaly = min(
        sum(anomaly_score > anomaly_detection_config.threshold.min_score),
        anomaly_detection_config.threshold.num_anomaly,
    )

    idx_sorted = np.argsort(anomaly_score)[::-1].tolist()
    idx_anomaly = idx_sorted[:num_anomaly]
    anomaly_score_sorted = pd.Series(
        anomaly_score[idx_sorted],
        index=pd.MultiIndex.from_tuples(
            [index_test[i] for i in idx_sorted],
            names=(col.DATETIME_ROUNDED, col.SRC_IP),
        ),
        name="anomaly_score",
    )

    x_test_embeddings = detector.compute_embeddings(x_test)
    x_train_labeled_embeddings = detector.compute_embeddings(x_train_labeled)

    shap_value_idx_sorted = detector.explain_anomaly(
        x_test[idx_anomaly], background_samples=x_train
    )
    shap_value_idx_sorted = pd.DataFrame(
        shap_value_idx_sorted,
        index=pd.MultiIndex.from_tuples(
            [index_test[i] for i in idx_anomaly],
            names=(col.DATETIME_ROUNDED, col.SRC_IP),
        ),
        columns=feature_label.columns,
    )
    stats = output_result(
        args,
        log,
        label,
        dir_report,
        x_train_labeled_embeddings=x_train_labeled_embeddings,
        x_test_embeddings=x_test_embeddings,
        idx_anomaly=idx_anomaly,
        shap_value_idx_sorted=shap_value_idx_sorted,
        anomaly_score_sorted=anomaly_score_sorted,
        stats=stats,
        previous_config=previous_config,
    )
    if args.debug:
        if isinstance(x_test, csr_matrix):
            x_test = x_test.toarray()
        ret = pd.DataFrame(x_test, index=index_test, columns=feature_label.columns)
        ret = ret.iloc[idx_sorted]
        ret.to_csv(os.path.join(dir_report, FILENAME_FEATURE_MATRIX))
    return stats


def output_result(
    args,
    log: pd.DataFrame,
    label: pd.Series,
    dir_report: str,
    *,
    x_train_labeled_embeddings,
    x_test_embeddings,
    idx_anomaly,
    shap_value_idx_sorted,
    anomaly_score_sorted,
    stats: dict,
    previous_config: labeled.Config,
):
    """Plot the detection result and output the report."""

    reporting.plot.plot_detection(
        X=x_test_embeddings,
        idx_anomaly=idx_anomaly,
        name_anomaly=shap_value_idx_sorted.index,
        X_labeled=x_train_labeled_embeddings,
        name_labeled=label.index,
        path_saved=os.path.join(dir_report, FILENAME_PLOT_DETECTION),
        no_plot=args.no_plot,
    )

    detection.detection_report(
        anomaly_score_sorted,
        shap_value_idx_sorted,
        shap_top_k=5,
    ).to_csv(os.path.join(dir_report, FILENAME_REPORT))

    labeled.factory(previous_config)[1].save_previous_log(
        df=log,
        entries=shap_value_idx_sorted.index,
    )

    stats["num_anomaly"] = len(idx_anomaly)
    stats["name_anomaly"] = shap_value_idx_sorted.index.tolist()

    logger.info(
        "successfully finish detection on subnet %s and service %s\n",
        args.subnet,
        args.service,
    )

    return stats


def report_all(path_list_stats: List[str], path_save: str):
    """
    Summarizing all reports and save it.

    Parameters
    ----------
    path_list_stats : list
        List of stats file paths
    path_save : str
        File path where the report will be saved
    """
    os.makedirs(os.path.dirname(path_save), exist_ok=True)

    logger.info("summarizing all reports...")
    results_pd = pd.DataFrame(
        [], columns=["datetime_rounded", "src_ip", "subnet", "service"]
    )
    idx = 0
    for path in path_list_stats:
        # Load stats
        stats = utils.load_json(path)
        subnet, service = stats["subnet"], stats["service"]
        try:
            anomaly_list = stats["name_anomaly"]
        except (KeyError, TypeError):
            continue
        if not anomaly_list:
            continue

        # Load report
        path_report = path.replace(FILENAME_STATS, FILENAME_REPORT)
        report = pd.read_csv(path_report, index_col=[0, 1], parse_dates=[0])
        logger.info(report.index)

        # Store anomalies in the DataFrame
        for (dt, src_ip) in anomaly_list:
            logger.info((dt, src_ip))
            results_pd.loc[idx] = [dt, src_ip, subnet, service]

            if idx == 0:
                results_shaps = pd.DataFrame([], columns=report.columns)
            results_shaps.loc[idx] = report.loc[(dt, src_ip)]

            idx += 1

    anomaly_found = idx > 0
    if anomaly_found:
        # Anomaly found
        results_pd = pd.concat([results_pd, results_shaps], axis=1)
        results_pd = results_pd.sort_values(
            ["anomaly_score", "datetime_rounded"], ascending=False
        )
        keys = results_pd["src_ip"].unique()
        results_pd_group = results_pd.groupby("src_ip")
        ret = pd.DataFrame([])
        for key in keys:
            ret = pd.concat([ret, results_pd_group.get_group(key)])

        ret.round(4).to_csv(path_save, index=False)
    else:
        # Anomaly not found
        pd.DataFrame([["no anomaly found"]]).to_csv(path_save, index=False)

    logger.info("[RESULT]", extra=to_stderr)
    logger.info("Detection summary file: %s", path_save, extra=to_stderr)
    num_anomaly_ipaddr = len(keys) if anomaly_found else 0
    logger.info(
        "Number of unique anomaly IP addresses: %s", num_anomaly_ipaddr, extra=to_stderr
    )
    if anomaly_found:
        for src_ip in keys:
            max_anomaly_score = max(
                results_pd.query("src_ip == @src_ip")["anomaly_score"]
            )
            logger.info(
                "- %s (max anomaly score: %s)",
                src_ip,
                max_anomaly_score,
                extra=to_stderr,
            )


def report_transfer(path: str, dir_to: str):
    """
    Copy Report Files to the Directory

    Parameters
    ----------
    path : str
        File path of the report to copy.
    dir_to : str
        Directory Path of destination directory.
        If you specify a directory that does not exist, a new directory is created.

    Raises
    ------
    TypeError
        Destination directory not specified.
    TypeError
        Report file does not exist.
    """

    # Argument checking
    if dir_to is None:
        raise TypeError("Destination directory not specified.")
    if not os.path.isfile(path):
        raise TypeError("Report file does not exist.")

    # Copy a report to a directory
    try:
        os.makedirs(dir_to, exist_ok=True)
        shutil.copy(path, dir_to)
    except Exception as ex:
        logger.error(
            "failed transfer report %s to %s, the error message: %s", path, dir_to, ex
        )
        raise ex
    else:
        logger.info("successfully transfered report %s to %s", path, dir_to)


def main_preproc_and_detection(args, config: DetectConfig):
    """Data preprocessing and anomaly detection."""
    log_all = load_preprocess_log(args, config)

    # Detecting for each subnet
    for subnet in config.detection_units.subnets.items():
        for service_name in subnet[1].services:
            detect_per_unit(config, service_name, log_all, subnet, args)

    # Reporting
    path_report_all = os.path.join(config.io.output.subdir, FILENAME_REPORT)
    path_list_stats = glob.glob(
        os.path.join(config.io.output.subdir, "**", FILENAME_STATS),
        recursive=True,
    )
    report_all(path_list_stats, path_save=path_report_all)
    if config.io.output.share_dir is not None:
        report_transfer(
            path_report_all,
            dir_to=os.path.join(
                config.io.output.share_dir,
                args.date_from.strftime(COMMANDLINE_DATE_FORMAT)
                + "__"
                + args.date_to.strftime(COMMANDLINE_DATE_FORMAT),
            ),
        )


def load_preprocess_log(args, config: DetectConfig):
    """Load and preprocess log.

    Warnings
    --------
    Sets config.io.output.subdir
    """
    # Loading logs
    logger.info(
        "loading logs during the period from %s to %s",
        args.date_from_training,
        args.date_to,
    )
    config.io.output.subdir = os.path.join(
        config.io.output.dir,
        args.date_from.strftime(COMMANDLINE_DATE_FORMAT)
        + "__"
        + args.date_to.strftime(COMMANDLINE_DATE_FORMAT),
    )
    log_all = load_log(
        dir_IDS_log=config.io.input.dir,
        date_from=args.date_from_training,
        date_to=args.date_to,
        nrows_read=args.nrows_read,
    )
    log_all = apply_exclude_lists(log_all, config.preprocess.exclude_lists)
    logger.info("[TARGET INFO]", extra=to_stderr)
    logger.info("Number of log entries loaded: %d", len(log_all), extra=to_stderr)
    logger.info(
        "Number of unique source IP addresses: %d",
        len(log_all.reset_index()["src_ip"].unique()),
        extra=to_stderr,
    )

    # Preprocessing logs
    logger.info("preprocessing logs")
    log_all = preprocess.screening_numlog(log_all, config.preprocess.screening)
    return log_all


def detect_per_unit(
    config: DetectConfig,
    service_name: str,
    log_all: pd.DataFrame,
    subnet: Tuple[str, Subnet],
    args,
):
    service = config.detection_units.services[service_name]
    log = preprocess.extract_log(
        log_all,
        subnets=subnet[1].cidrs,
        include_ports=service.include,
        exclude_ports=service.exclude,
    )
    # one can also load_previous(known_anomaly, label_value=1)
    # and/or load_previous(unknown, label_value=None).
    # known_anomaly can be concat-ed with known_normal
    # since both have label values.
    # log_unknown can be concat-ed with log without label values.
    known_normal = load_previous(
        config=config.io.previous.load.known_normal,
        date_to=args.date_to_training,
        label_value=1,
    )
    log_labeled = labeled.factory(config.io.previous.log)[0].load_previous_log(
        entries=known_normal.index,
    )
    log_labeled = apply_exclude_lists(log_labeled, config.preprocess.exclude_lists)
    log_labeled = preprocess.extract_log(
        log_labeled,
        subnets=subnet[1].cidrs,
        include_ports=service.include,
        exclude_ports=service.exclude,
    )
    log.drop(
        known_normal.index.tolist(), level=col.SRC_IP, inplace=True, errors="ignore"
    )
    args.subnet = subnet[0]
    args.service = service_name
    main_detection(args, config, log=pd.concat([log, log_labeled]), label=known_normal)


def load_log(
    dir_IDS_log: str,
    date_from: datetime,
    date_to: datetime,
    nrows_read: Optional[int] = None,
) -> pd.DataFrame:
    """
    load IDS logs of the dates in [ date_from, date_to ]

    Parameters
    ----------
    dir_IDS_log
        The path of the directory containing logs to be load.
    date_from
        Start date.
    date_to
        End date.
    nrows_read
        Maximum number of rows to load, by default None

    Returns
    -------
    log
        IDS log.
    """

    from psykoda.io.reader._fujitsu_splunk import FujitsuSplunk
    from psykoda.preprocess import RoundDatetime, drop_null, set_index
    from psykoda.utils import daterange2list

    daterange = daterange2list(date_from, date_to)

    def _load_log_catch(load, r):
        for dt in r:
            try:
                yield load(dt)
            except FileNotFoundError as ex:
                logger.warning("not found: %s", dt, exc_info=ex)

    io = FujitsuSplunk(dir_IDS_log=dir_IDS_log, nrows_read=nrows_read)
    log = set_index(
        RoundDatetime("hour")(pd.concat(_load_log_catch(io.load_log, daterange)))
    )
    log = drop_null(log)
    return log


def load_previous(
    config: LoadPreviousConfigItem, date_to: datetime, label_value: float
) -> pd.Series:
    from psykoda.preprocess import round_datetime
    from psykoda.utils import DateRange

    if config.list is None:
        return pd.Series()

    def date_filter(row):
        assert isinstance(row.name[0], datetime)
        return round_datetime(row.name[0], "day") in DateRange(
            end_inclusive=date_to, length=config.ndate
        )

    try:
        df = pd.read_csv(
            config.list,
            index_col=[col.DATETIME_ROUNDED, col.SRC_IP],
            parse_dates=[col.DATETIME_ROUNDED],
        )
    except FileNotFoundError as ex:
        logger.warning(
            "Specified list of %s was not found.  "
            "Create one with at least %s columns, "
            "or remove the path or entry from config.",
            config.list,
            (col.DATETIME_ROUNDED, col.SRC_IP),
            exc_info=ex,
        )
        return pd.Series()
    df = df[df.apply(date_filter, axis=1)]
    return pd.Series(label_value, index=df.index)


def apply_exclude_lists(
    log: pd.DataFrame, dir_exclude_lists: Optional[str]
) -> pd.DataFrame:
    """
    exclude logs according to exclude lists in dir_exclude_lists

    Parameters
    ----------
    log
        Source log.
    dir_exclude_lists
        The path of directory containing exclude list csv files.

    Returns
    -------
    log
        Log after applying exclude list.
    """
    from os import path

    from psykoda.constants import EXCLUDE_LIST_FILE_SPLITSTR, EXCLUDE_LIST_PREFIX
    from psykoda.preprocess import exclude_log

    assert log is not None

    if dir_exclude_lists is None:
        return log
    path_list = glob.glob(path.join(dir_exclude_lists, EXCLUDE_LIST_PREFIX + "*.csv"))
    if not path_list:
        logger.warning("no exclude_list files exist")
        return log

    exclusion = (
        {
            "column_name": path.splitext(path.basename(filter_path))[0].split(
                EXCLUDE_LIST_FILE_SPLITSTR
            )[1],
            "filter_patterns": validate_patterns(
                pd.read_csv(filter_path, index_col=0).index
            ),
        }
        for filter_path in path_list
    )

    return exclude_log(log, exclusion)


def validate_patterns(patterns: pd.Index):
    """
    Strip whitespaces in Index from left and right sides.

    Parameters
    ----------
    patterns
        Index.

    Returns
    -------
    ret
        Index with whitespace removed.
    """
    import pandas.api.types as types  # pylint: disable=import-outside-toplevel

    # convention

    if not types.is_string_dtype(patterns):
        return patterns
    ret = patterns.str.strip()
    if not patterns.equals(ret):
        warnings.warn("Spaces around exclusion pattern are deprecated")
    return ret
