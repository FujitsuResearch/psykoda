"""
API
"""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas


class Reader(ABC):
    """IDS log reader API."""

    @abstractmethod
    def load_log(self, dt: datetime) -> pandas.DataFrame:
        """Load IDS log of date."""
