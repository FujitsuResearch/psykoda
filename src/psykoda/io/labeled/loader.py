"""Previous Log Loader API"""

import abc

import pandas


class Loader(abc.ABC):
    """Previous Log Loader"""

    @abc.abstractmethod
    def load_previous_log(self, entries: pandas.MultiIndex) -> pandas.DataFrame:
        """Load previous log from storage.

        Parameters
        ----------
        entries
            collection of (datetime_rounded, src_ip)s
            to load log corresponding to
        """
