"""Lightweight fast statistics utilities.

Provides:
- OnlineStats: an efficient online accumulator for mean, variance (Welford), min, max, count.
  Supports adding single values, Python iterables, or NumPy arrays (uses fast vectorized combine when NumPy is available).
- fast_stats: a fast one-shot stats function that uses NumPy when available and falls back to pure Python.

Behavior notes:
- Non-finite values (NaN, inf) are ignored by the online accumulator and by fast_stats' count; min/max ignore non-finite values.
- If no finite values have been seen, mean/stddev return float('nan'), min/max return None, count is 0.
"""
from __future__ import annotations

from typing import Iterable, Optional, Dict
import math

try:
    import numpy as np  # Optional, used for fast batch ops
except Exception:
    np = None


class OnlineStats:
    """Online statistics accumulator using Welford's algorithm.

    Exposes: count, mean, variance (population/sample), stddev, min, max.

    Methods:
    - add(value): add a single numeric value (ignores non-finite values)
    - add_batch(iterable): add many values; if a NumPy array is passed and numpy is available,
      a fast vectorized path is used and combined with current state in O(1).
    - merge(other): merge another OnlineStats into this one in O(1).
    - to_dict(): snapshot dictionary of statistics.
    """

    def __init__(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._M2 = 0.0  # sum of squares of differences from the current mean
        self._min = None
        self._max = None

    def add(self, value: float) -> None:
        """Add a single numeric value. Non-finite values are ignored."""
        if value is None:
            return
        if not math.isfinite(value):
            return
        if self.n == 0:
            self._mean = float(value)
            self._M2 = 0.0
            self._min = float(value)
            self._max = float(value)
            self.n = 1
            return
        self.n += 1
        delta = value - self._mean
        self._mean += delta / self.n
        delta2 = value - self._mean
        self._M2 += delta * delta2
        if value < self._min:
            self._min = float(value)
        if value > self._max:
            self._max = float(value)

    def add_batch(self, values: Iterable[float]) -> None:
        """Add many values. If `values` is a NumPy array and NumPy is available, use a fast path.

        Non-finite values are ignored.
        """
        # Fast path for numpy arrays
        if np is not None and isinstance(values, np.ndarray):
            # Select finite values only
            mask = np.isfinite(values)
            if not mask.any():
                return
            arr = values[mask]
            m2 = float(np.sum((arr - float(np.mean(arr))) ** 2))
            n2 = arr.size
            mean2 = float(np.mean(arr))
            # Create a temporary OnlineStats and merge
            tmp = OnlineStats()
            tmp.n = int(n2)
            tmp._mean = mean2
            tmp._M2 = m2
            tmp._min = float(np.min(arr))
            tmp._max = float(np.max(arr))
            self.merge(tmp)
            return

        # Generic path: iterate values
        for v in values:
            try:
                self.add(v)  # add handles filtering
            except TypeError:
                # skip non-numeric items
                continue

    def merge(self, other: "OnlineStats") -> None:
        """Merge another OnlineStats into this one using parallel combine formulas."""
        if not isinstance(other, OnlineStats):
            raise TypeError("other must be an OnlineStats")
        if other.n == 0:
            return
        if self.n == 0:
            # copy other
            self.n = other.n
            self._mean = other._mean
            self._M2 = other._M2
            self._min = other._min
            self._max = other._max
            return
        n1 = self.n
        n2 = other.n
        delta = other._mean - self._mean
        tot = n1 + n2
        # combined mean
        new_mean = (n1 * self._mean + n2 * other._mean) / tot
        # combined M2
        new_M2 = self._M2 + other._M2 + delta * delta * (n1 * n2) / tot
        self.n = tot
        self._mean = new_mean
        self._M2 = new_M2
        # min / max
        if self._min is None:
            self._min = other._min
        elif other._min is not None and other._min < self._min:
            self._min = other._min
        if self._max is None:
            self._max = other._max
        elif other._max is not None and other._max > self._max:
            self._max = other._max

    @property
    def count(self) -> int:
        return int(self.n)

    @property
    def mean(self) -> float:
        if self.n == 0:
            return float("nan")
        return float(self._mean)

    def variance(self, sample: bool = False) -> float:
        """Return variance. By default returns population variance (ddof=0).

        If sample=True, returns sample variance (ddof=1) when n>1, otherwise nan.
        """
        if self.n == 0:
            return float("nan")
        if sample:
            if self.n < 2:
                return float("nan")
            return float(self._M2 / (self.n - 1))
        return float(self._M2 / self.n)

    def stddev(self, sample: bool = False) -> float:
        v = self.variance(sample=sample)
        if math.isnan(v):
            return float("nan")
        return math.sqrt(v)

    @property
    def minimum(self) -> Optional[float]:
        return None if self._min is None else float(self._min)

    @property
    def maximum(self) -> Optional[float]:
        return None if self._max is None else float(self._max)

    def to_dict(self, sample: bool = False) -> Dict[str, Optional[float]]:
        return {
            "count": self.count,
            "mean": self.mean if self.count > 0 else float("nan"),
            "variance": self.variance(sample=sample),
            "stddev": self.stddev(sample=sample),
            "min": self.minimum,
            "max": self.maximum,
        }


def fast_stats(values: Iterable[float], ddof: int = 0) -> Dict[str, Optional[float]]:
    """One-shot fast statistics for an iterable/array.

    Uses NumPy if available for speed and NaN-awareness. Returns a dict with keys:
    - count, mean, variance, stddev, min, max

    ddof controls the delta degrees of freedom passed to variance/std (0 for population, 1 for sample).
    """
    # Try numpy path
    if np is not None:
        try:
            arr = np.asarray(values)
            # compute mask of finite values
            mask = np.isfinite(arr)
            cnt = int(mask.sum())
            if cnt == 0:
                return {"count": 0, "mean": float("nan"), "variance": float("nan"), "stddev": float("nan"), "min": None, "max": None}
            sub = arr[mask]
            mean = float(np.mean(sub))
            # numpy's ddof argument is supported by np.var / np.std
            var = float(np.var(sub, ddof=ddof))
            std = float(np.sqrt(var))
            mn = float(np.min(sub))
            mx = float(np.max(sub))
            return {"count": cnt, "mean": mean, "variance": var, "stddev": std, "min": mn, "max": mx}
        except Exception:
            # fall back to python path
            pass

    # Pure python fallback
    acc = OnlineStats()
    # ddof handled by converting to sample=True when ddof==1
    for v in values:
        try:
            acc.add(v)
        except Exception:
            continue
    if ddof == 1:
        var = acc.variance(sample=True)
    else:
        var = acc.variance(sample=False)
    return {"count": acc.count, "mean": acc.mean, "variance": var, "stddev": math.sqrt(var) if not math.isnan(var) else float("nan"), "min": acc.minimum, "max": acc.maximum}
