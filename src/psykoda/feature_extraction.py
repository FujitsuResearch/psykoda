"""Extract and manage feature values"""

import copy
import ipaddress
import itertools
from dataclasses import dataclass
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from pandas import DataFrame, Index, Series
from scipy.sparse import csr_matrix, lil_matrix

from psykoda import utils
from psykoda.constants import col, ip

logger = getLogger(__name__)


class FeatureLabel:
    r"""Feature matrix with label values

    Parameters
    ----------
    feature
        scipy sparse feature matrix

        :shape: (n_samples, n_features)
    index
        index of feature matrix

        :length: n_samples
    columns
        column of feature matrix

        :length: n_features
    label
        labels

        :shape: (n_samples, )
    idf_sid
        IDF (Inverse Document Frequency) for, and indexed by, sid
    idf_dport
        IDF for, and indexed by, dest_port
    """

    def __init__(
        self,
        *,
        feature: csr_matrix,
        index: List[Tuple[datetime, str]],
        columns: list,
        label: Optional[np.ndarray] = None,
        idf_sid: Series,
        idf_dport: Series,
    ):
        self.feature = feature
        self.index = index
        self.columns = columns
        self.label = label
        self.idf_sid = idf_sid
        self.idf_dport = idf_dport
        self._invariant()

    def extract_nonzeros(self):
        """
        restructure feature matrix by excluding all zero rows/cols
        """
        self.extract_nonzeros_rows()
        self.extract_nonzeros_cols()
        self._invariant()

    def extract_nonzeros_rows(self):
        """
        exclude rows whose elements are all zeros
        """
        ind = np.where(self.feature.sum(axis=1) > 0)[0]
        self.feature = self.feature[ind]
        self.index = [self.index[i] for i in ind]
        if self.label is not None:
            self.label = self.label[ind]

    def extract_nonzeros_cols(self):
        """
        exclude columns whose elements are all zeros
        """
        ind = np.where(np.ravel(self.feature.sum(axis=0) > 0))[0]
        self.feature = self.feature[:, ind]
        self.columns = [self.columns[i] for i in ind]

    def loc(self, sample: Tuple[datetime, str]) -> Series:
        """
        correspond to DataFrame.loc[sample]
        e.g. sample = (pandas.Timestamp("2021-04-01 14:00:00"), "10.1.1.1")

        Parameters
        ----------
        sample

        Returns
        -------
        Series
            A series of features corresponding to sample.
        """
        idx = self.index.index(sample)
        indices = self.feature[idx].indices
        data = self.feature[idx].data
        col = [self.columns[i] for i in indices]
        return Series(data, index=col)

    def put_labels(self, labeled_samples: Series):
        """
        Assign label value 1 to known normal samples

        Parameters
        ----------
        labeled_samples
            all-1 vector whose indexes are known normal

            :index: Index[datetime_rounded: datetime, src_ip: str]
        """
        self.label = Series(0.0, index=self.index)
        if labeled_samples is not None:
            self.label.loc[labeled_samples.index] = labeled_samples
        self.label = np.array(self.label, dtype="float")

    def split_train_test(
        self, date_to_training: datetime
    ) -> Tuple[csr_matrix, Series, csr_matrix, Index]:
        """split feature matrix and return training and test sets

        Parameters
        ----------
        date_to_training
            samples (and their features) earlier than date_to_training are used for training.

        Returns
        -------
        X_train
            feature matrix for training

            :shape: (n_samples_train, n_features)
        y_train
            labels for training

            :length: n_samples_train
        X_test
            feature matrix for anomaly detection

            :shape: (n_samples_test, n_features)
        index_test: Index[datetime_rounded: datetime, src_ip: str]
            row index for anomaly detection

            :length: n_samples_test

        Notes
        --------
        date_to_training is compared against datetime_rounded.replace(hour=0).  Samples with equality are included in training set.
        """
        if self.label is None:
            self.label = Series(0.0, index=self.index)

        train_mask = np.array(
            [
                dtr_srcip[0].replace(hour=0) <= date_to_training
                for dtr_srcip in self.index
            ]
        )

        return (
            self.feature[train_mask],
            self.label[train_mask],
            self.feature[~train_mask],
            Index(self.index)[~train_mask],
        )

    def _invariant(self):
        """This equality must always hold"""
        assert self.feature.shape == (len(self.index), len(self.columns))


@dataclass
class IDFConfig:
    """
    Settings for IDF (Inverse Document Frequency).
    """

    min_count: int
    num_feature: int


@dataclass
class FeatureExtractionConfig:
    """
    Settings for feature_extraction.
    """

    idf: Dict[str, IDFConfig]
    address_to_location: Optional[str]


def feature_extraction_all(
    log: DataFrame,
    idf_config: Dict[str, IDFConfig],
    iptable: DataFrame,
) -> Optional[FeatureLabel]:
    """compute feature matrix from preprocessed log for each sample

    Parameters
    ----------
    log
        data to construct feature matrix from.

        :index:
            (datetime_rounded, src_ip) (exact match)
        :assumed:
            (dest_ip, dest_port, sid, src_port) (included)
    idf_config
        Configuration for IDF

        :key:
            column
        :value:
            configuration

        refer to calculate_idf and its unittests.
    iptable
        IP locations definition table

        .. todo::
            example
    """
    _dst_port = copy.copy(log[col.DEST_PORT])
    _dst_port[_dst_port >= 49152] = -1
    return _feature_extraction_all(
        log.assign(**{col.DEST_PORT: _dst_port}),
        idf_config=idf_config,
        iptable=iptable,
    )


def _feature_extraction_all(
    log: DataFrame,
    idf_config: Dict[str, IDFConfig],
    iptable: DataFrame,
) -> Optional[FeatureLabel]:
    """refer to feature_extraction_all.

    Parameters
    ----------
    log
    idf_config
        Configuration for IDF

        :key:
            column
        :value:
            configuration

        refer to calculate_idf and its unittests.
    iptable
        IP locations definition table
    """
    idfs_groupss = utils.dmap(
        lambda column, config: calculate_idf(
            log,
            column=column,
            num_idf_feature=config.num_feature,
            min_count=config.min_count,
        ),
        idf_config,
    )
    idfs = utils.vmap(utils.first, idfs_groupss)
    groupss = utils.vmap(utils.second, idfs_groupss)

    if not any(groupss.values()):
        return None

    ip2loc = find_ip_location(_ips(groupss), iptable=iptable)
    dict_loc2idx: Dict[str, int] = utils.index_from_unsorted(ip2loc)
    sample_list = _samples(groupss)
    sample2loc2idx = Series(
        [dict_loc2idx[ip2loc.loc[x[1]]] for x in sample_list], index=sample_list
    )
    ip2idx = ip2loc.map(dict_loc2idx)

    num_locations = len(dict_loc2idx)
    num_idfs: int = sum(map(len, idfs.values()))

    sample2idx = Series(utils.index_from_sorted(sample_list))

    _shape_tensor = [num_locations, num_locations, num_idfs]

    columns = list(
        itertools.product(dict_loc2idx.keys(), dict_loc2idx.keys(), _idfs(groupss))
    )

    feature_matrix = _construct_feature_matrix(
        shape=(len(sample_list), _shape_tensor),
        idfs_groupss=idfs_groupss,
        sample_to_location_index=sample2loc2idx,
        sample_index=sample2idx,
        ip_index=ip2idx,
    ).tocsr()

    return FeatureLabel(
        feature=feature_matrix,
        index=sample_list,
        columns=columns,
        idf_sid=idfs[col.SID],
        idf_dport=idfs[col.DEST_PORT],
    )


Group = Tuple[Any, DataFrame]
Groups = List[Group]


def _construct_feature_matrix(
    shape,
    idfs_groupss: Dict[str, Tuple[Series, Groups]],
    sample_to_location_index,
    sample_index,
    ip_index,
) -> lil_matrix:
    feature_matrix = lil_matrix((shape[0], np.product(shape[1])))
    ravel_index = _ravel_index(shape[1])
    i = 0
    for (column, (idf, groups)) in idfs_groupss.items():
        logger.debug(
            "column: %s, idf: %s, groups: %s", type(column), type(idf), type(groups)
        )
        for (value, group) in groups:
            sizes = group.groupby(group.index.names + [col.DEST_IP]).size()
            # accessing more than 100 times is no more significant than 100 times.
            tf = sizes.clip(lower=0, upper=100)
            samples = sizes.index.droplevel(col.DEST_IP).to_flat_index()
            dips = sizes.index.get_level_values(col.DEST_IP)
            ind = ravel_index(
                (
                    sample_to_location_index.loc[samples].values,
                    ip_index.loc[dips].values,
                    i,
                )
            )
            feature_matrix[sample_index.loc[samples], ind] = tf * idf[value]
            i += 1
    return feature_matrix


def _ravel_index(shape: Iterable[int]):
    """Convert multidimensional index into single dimensional index"""

    def f(coord: Iterable[int]):
        ret = 0
        for (s, c) in zip(shape, coord):
            assert (0 <= np.array(c)).all()
            assert (np.array(c) < s).all()
            ret *= s
            ret += c
        return ret

    return f


def calculate_idf(
    log: DataFrame, column: str, num_idf_feature: int, min_count: int
) -> Tuple[Series, Groups]:
    """Calculate Inverse Document Frequency (IDF) of given column.

    Every unique index is considered a document.  A value of given column is considered to appear in the document if and only if there is at least one row with the index and column values.

    Parameters
    ----------
    log
    column
        name of column to calculate IDF values on.
    num_idf_feature
        (soft) maximum number of unique values of log[column] to keep IDF for.
    min_count
        minimum number of appearance of a log[column] value to calculate IDF for.

    Returns
    -------
    idf
        log(1 + raw_idf), indexed by column values
        e.g. Series([5.3, 3.2, 2.8], index=[22, 80, 3389], name=dest_port_idf)
    groups
        list of (column_value, matching_dataframe), indexed the same as idf
    """

    # total number of documents
    df_denom = log.index.nunique()

    grouped = log.groupby(column)

    # number of documents in which each value appeaed
    unique_idx_by_value = grouped.apply(lambda group: group.index.nunique())

    # filter (by >= min_count)
    relevant_values = min_count <= log[column].value_counts()

    # raw IDF
    idf_raw = df_denom / unique_idx_by_value[relevant_values]

    # transformed IDF
    idf = np.log(1 + idf_raw).sort_values(ascending=False)  # pylint: disable=no-member
    # since idf_raw is Series, np.log actually returns Series, which has sort_values
    idf.name = column + col.IDF_SUFFIX

    # keep largest num_idf_feature values, and ties
    if len(idf) > num_idf_feature:
        idf = idf[idf >= idf.iat[num_idf_feature - 1]]

    groups = [(i, grouped.get_group(i)) for i in idf.index]

    return idf, groups


def find_ip_location(ip_list: List[str], iptable: DataFrame) -> Series:
    """Find location information for IP addresses

    Current implementation returns first matches, but this is not part of specification.

    Parameters
    ----------
    ip_list
    iptable
        location information table with at least two columns

        :IP_TABLE_SUBNET: network address in CIDR format
        :IP_TABLE_LOCATION: location name

    Returns
    -------
    location : Series
        location information

        :index: ip_address: str
        :value: location_name: str
    """

    matcher = location_matcher(
        iptable[col.IP_TABLE_SUBNET], iptable[col.IP_TABLE_LOCATION]
    )
    return Series(
        [matcher(ipaddress.ip_address(address)) for address in ip_list], index=ip_list
    )


def location_matcher(subnets: Iterable[str], locations: Iterable[str]):
    """
    Generate a function that returns location when you input ip address.

    Parameters
    ----------
    subnets
        Subnet addresses in CIDR format.
    locations : Iterable[str]
        Location names for each subnet address.

    Returns
    -------
    matcher
        A function that returns location when you input ip address.
    """
    matchers = [
        (ipaddress.ip_network(subnet), location)
        for (subnet, location) in zip(subnets, locations)
    ] + [
        (ipaddress.ip_network(subnet), ip.UNKNOWN_PRIVATE)
        for subnet in ip.IPV4_PRIVATE.values()
    ]

    def matcher(address):
        for (subnet, location) in matchers:
            if address in subnet:
                return location
        return ip.UNKNOWN_GLOBAL

    return matcher


def _ips(groupss: Dict[str, Groups]) -> List[str]:
    ret: Set[str] = set()
    for groups in groupss.values():
        for (_, group) in groups:
            ret = ret.union(
                itertools.chain(
                    group.index.get_level_values(col.SRC_IP).unique().tolist(),
                    group[col.DEST_IP].unique().tolist(),
                )
            )
    return sorted(ret)


def _samples(groupss: Dict[str, Groups]) -> List[Tuple[datetime, str]]:
    ret: Set[Tuple[datetime, str]] = set()
    for groups in groupss.values():
        for (_, group) in groups:
            ret = ret.union(group.index.unique().tolist())
    return sorted(ret)


def _idfs(groupss: Dict[str, Groups]) -> Iterable[str]:
    for (name, groups) in groupss.items():
        for (value, _) in groups:
            yield f"{name}_{value}"
