"""Preprocessing"""

import collections
import dataclasses
import ipaddress
import warnings
from datetime import datetime
from logging import getLogger
from typing import Callable, Iterable, List, Optional

import pandas

from psykoda.constants import col, ip
from psykoda.utils import get_series, replace_match

logger = getLogger(__name__)


class RoundDatetime:
    def __init__(self, time_unit: str):
        self.time_unit = time_unit
        self.table = {
            smaller_time_unit: 0
            for smaller_time_unit in _time_units[: _time_units.index(self.time_unit)]
        }

    def __call__(self, df):
        return df.assign(
            datetime_rounded=df["datetime_full"].apply(
                lambda dt: dt.replace(**self.table)
            )
        )


def round_datetime(dt: datetime, time_unit: str):
    return dt.replace(
        **{
            smaller_time_unit: 0
            for smaller_time_unit in _time_units[: _time_units.index(time_unit)]
        }
    )


def set_index(df: pandas.DataFrame) -> pandas.DataFrame:
    return df.set_index(["datetime_rounded", "src_ip"]).sort_index()


def drop_null(df: pandas.DataFrame) -> pandas.DataFrame:
    return df[~df.isnull().any(axis=1)]


def _in_subnets(
    series: pandas.Series, subnets: Optional[List[str]], *, empty_is_all: bool = False
) -> pandas.Series:
    assert isinstance(series, pandas.Series)
    # corner case: no filtering
    if subnets is None:
        return pandas.Series(data=True, index=series.index)

    # corner case: subnet has no element
    if empty_is_all and len(subnets) == 0:
        warnings.warn(
            "\n".join(
                [
                    "`subnets` is empty: all IP addresses are included",
                    "Explicit None (in Python; null in JSON) is preferred",
                ]
            )
        )
        return pandas.Series(data=True, index=series.index)

    # An array in which the elements corresponding to
    # the IP addresses in the subnet will be True.
    bools_ip = pandas.Series(data=False, index=series.index)

    # Type conversion
    sub_networks = [
        ipaddress.ip_network(replace_match(ip.IPV4_PRIVATE, addr)) for addr in subnets
    ]

    # Generate bools_ip by helper function
    bools_ip[series.map(addr_in_subnets(sub_networks))] = True

    return bools_ip


def addr_in_subnets(sub_networks: list) -> Callable[[str], bool]:
    """Build "in some of these subnets" filter for IP addresses

    Returns
    -------
    in_subnets(addr)
        predicate for IP addresses

    Warnings
    --------
    Optimized for IPv4.  Does not support IPv6.
    """

    netaddr_and_mask = [
        (
            int.from_bytes(subnet.network_address.packed, "big"),
            int.from_bytes(subnet.netmask.packed, "big"),
        )
        for subnet in sub_networks
    ]

    # Helper function that returns True if the IP address is in a subnets.
    def _ret(addr: str):
        addr_int = sum(
            [int(part) << i for i, part in zip([24, 16, 8, 0], addr.split("."))]
        )
        for netaddr, netmask in netaddr_and_mask:
            if addr_int & netmask == netaddr:
                return True
        return False

    return _ret


def extract_log(
    log: pandas.DataFrame,
    subnets: Optional[List[str]],
    include_ports: Optional[List[int]] = None,
    exclude_ports: Optional[List[int]] = None,
) -> pandas.DataFrame:

    """
    extract logs with subnets and service_dport

    Parameters
    ----------
    subnets
        List of subnets to which the IP addresses to be extracted belong. e.g ["10.25.148.0/24", "192.168.0.0/16"] (CIDR format)
        None to extract all IP addresses.
    include_ports
        List of port numbers to extract. e.g [22, 3389]
        None to extract all port numbers.
    exclude_ports
        List of port numbers not to extract, e.g [22, 3389],
        Empty or None to exclude no port numbers.
        Exclusion takes precedence over inclusion.
    """

    if len(log) == 0:
        return log

    bools_service = pandas.Series(False, index=log.index)

    if include_ports is None:
        bools_service[:] = True
    elif len(include_ports) == 0:
        warnings.warn(
            "`include` is empty: all dest ports are included\nExplicit None (in Python; null in JSON) is preferred"
        )
        bools_service[:] = True
    else:
        for dport in include_ports:
            bools_service[log[col.DEST_PORT] == dport] = True

    if exclude_ports is not None:
        for dport in exclude_ports:
            bools_service[log[col.DEST_PORT] == dport] = False

    bools_ip = _in_subnets(
        get_series(log.index, col.SRC_IP), subnets, empty_is_all=True
    )

    logger.info("service: %s, ip: %s", bools_service, bools_ip)
    return log[bools_service & bools_ip]


def filter_out(
    log: pandas.DataFrame, column_name: str, filter_patterns: pandas.Index
) -> pandas.DataFrame:
    """Filter out rows according to patterns of column values.

    Parameters
    ----------
    log
    column_name
        name of data or index column to match patterns against.
    filter_patterns
        patterns to filter out matching rows.
        if column_name is col.SRC_IP or col.DEST_IP, a pattern is a CIDR notation (ipaddress.ip_network() accepts).
        otherwise, a pattern is a string to match the values exactly.
    """

    # find column_values: handle both value columns and index columns properly
    if column_name in log.columns:
        column_values = log[column_name]
    elif column_name in log.index.names:
        column_values = log.index.get_level_values(column_name).to_series(
            index=log.index
        )
    else:
        raise KeyError(
            f"No value or index column name {column_name} in DataFrame:\n{log}"
        )

    mask = pandas.Series(True, index=log.index)

    # handle IP address column differently
    if column_name in (col.SRC_IP, col.DEST_IP):
        mask[_in_subnets(column_values, list(filter_patterns))] = False
    else:
        for d in filter_patterns:
            # exact match
            mask[column_values == str(d).strip()] = False

    return log[mask]


@dataclasses.dataclass
class ScreeningConfig:
    """
    Log screening settings.
    """

    min: int
    max: int = 10 ** 8


def screening_numlog(
    log: pandas.DataFrame, config: ScreeningConfig
) -> pandas.DataFrame:
    """
    exclude ip addresses whose numbers of logs are out of [ config.min, config.max ]

    Parameters
    ----------
    log
        Source log.
    config
        Settings for screening.

    Returns
    -------
    log
        Screened log.
    """
    logger.debug(
        "screening samples whose num_logs are out of [ %d, %d ]", config.min, config.max
    )

    num_logs = collections.Counter(log.index.get_level_values("src_ip"))
    srcip_list_drop = [
        ip
        for ip in num_logs
        if (num_logs[ip] < config.min) or (config.max < num_logs[ip])
    ]
    logger.debug("drop %s", srcip_list_drop)
    logger.debug("from:\n%s", log)
    rows_to_remove = log.index.get_level_values(col.SRC_IP).isin(srcip_list_drop)
    return log[~rows_to_remove]


def exclude_log(log: pandas.DataFrame, exclusion: Iterable[dict]) -> pandas.DataFrame:
    for kwargs in exclusion:
        if log.empty:
            return log
        log = filter_out(log, **kwargs)
    return log


_time_units = [
    "microsecond",
    "second",
    "minute",
    "hour",
    "day",
    "month",
    "year",
]
