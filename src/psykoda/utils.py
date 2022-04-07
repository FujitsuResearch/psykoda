"""Miscellaneous utilities"""

import json
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import pandas


def _start(inclusive: Optional[datetime], exclusive: Optional[datetime]):
    if exclusive is None:
        return inclusive
    return exclusive + timedelta(1)


def _end(inclusive: Optional[datetime], exclusive: Optional[datetime]):
    if inclusive is None:
        return exclusive
    return inclusive + timedelta(1)


class DateRange:
    def __init__(
        self,
        *,
        start_inclusive: Optional[datetime] = None,
        start_exclusive: Optional[datetime] = None,
        end_inclusive: Optional[datetime] = None,
        end_exclusive: Optional[datetime] = None,
        length: Optional[int] = None,
    ):
        start = _start(start_inclusive, start_exclusive)
        end = _end(end_inclusive, end_exclusive)
        if start is None:
            assert end is not None
            assert length is not None
            start = end - timedelta(length)
        if end is None:
            assert start is not None
            assert length is not None
            end = start + timedelta(length)
        if length is None:
            self._length = (end - start).days
        else:
            assert start + timedelta(length) == end
            self._length = length
        self._start = start
        self._end = end

    def __str__(self):
        return "[ {} .. {} )".format(
            self._start.strftime("%Y-%m-%d"), self._end.strftime("%Y-%m-%d")
        )

    def __len__(self):
        return self._length

    def __iter__(self):
        current = self._start
        while current < self._end:
            yield current
            current = current + timedelta(1)

    def __contains__(self, dt: datetime):
        return self._start <= dt < self._end


def daterange2list(
    start_inclusive: datetime, end_inclusive: datetime
) -> List[datetime]:
    """Construct list from range of dates.

    .. todo::
        Replace with date range object with iterator?
    """
    daterange = [
        start_inclusive + timedelta(n)
        for n in range((end_inclusive - start_inclusive).days + 1)
    ]

    return daterange


def load_json(path: str) -> dict:
    """Load object from .json file.

    json.load(object_hook) is used to construct pandas.Timestamp from
    object like {type: datetime, value: 2021-04-01}.
    """

    def object_hook(serialized):
        if "type" not in serialized:
            return serialized
        if serialized["type"] == "datetime":
            return pandas.Timestamp(serialized["value"])
        raise TypeError(f"type `{serialized['type']}` is not recognized")

    with open(path, "r", encoding="utf_8") as f:
        ret = json.load(f, object_hook=object_hook)

    return ret


def save_json(obj: dict, path: str):
    """Save object to .json file.

    json.dump(default) is used to serialize datetime as
    object like {type: datetime, value: 2021-04-01}.
    """

    def default(unserializable):
        if isinstance(unserializable, datetime):
            return {"type": "datetime", "value": str(unserializable)}
        raise TypeError("not supported")

    with open(path, "w", encoding="utf_8") as f:
        json.dump(obj, f, indent=4, default=default)


def get_series(index: pandas.Index, level: Union[int, str]) -> pandas.Series:
    """get_velel_values as Series, indexed by itself."""
    return index.get_level_values(level).to_series(index)


K = TypeVar("K")
V = TypeVar("V")
R = TypeVar("R")
F = TypeVar("F")
S = TypeVar("S")


def vmap(f: Callable[[V], R], d: Dict[K, V]) -> Dict[K, R]:
    """map over dict values."""
    return {k: f(v) for (k, v) in d.items()}


def dmap(f: Callable[[K, V], R], d: Dict[K, V]) -> Dict[K, R]:
    """map over dict items."""
    return {k: f(k, v) for (k, v) in d.items()}


def flip(t: Tuple[F, S]) -> Tuple[S, F]:
    """Swap first and second items of 2-tuple."""
    return t[1], t[0]


def first(t: Tuple[F, S]) -> F:
    """First item of 2-tuple."""
    return t[0]


def second(t: Tuple[F, S]) -> S:
    """Second item of 2-tuple."""
    return t[1]


def index_from_sorted(ls: List[V]) -> Dict[V, int]:
    """Minimal perfect (non-cryptographic) hash from unique values.

    Works for unique unsorted list too, but named as sorted."""
    return dict(map(flip, enumerate(ls)))


def index_from_unsorted(it: Iterable[V]) -> Dict[V, int]:
    """Minimal perfect (non-cryptographic) hash from values."""
    return index_from_sorted(sorted(set(it)))


def replace_match(d: Dict[V, V], v: V) -> V:
    """Replace value, if match is found.

    Parameters
    ----------
    d
        replacements
    v
        replacee"""
    try:
        return d[v]
    except KeyError:
        return v
