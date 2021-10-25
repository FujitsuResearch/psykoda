"""Previous Log Saver API"""

import abc

import pandas


class Saver(abc.ABC):
    """Previous Log Saver"""

    @abc.abstractmethod
    def save_previous_log(
        self,
        df: pandas.DataFrame,
        entries: pandas.MultiIndex,
    ):
        """Save log to storage for future use.

        Parameters
        ----------
        df
            log
        entries
            collection of (datetime_rounded, src_ip)s
            to save log corresponding to
        """
