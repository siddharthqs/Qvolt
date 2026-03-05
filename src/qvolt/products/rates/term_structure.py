"""
Term Structure Module

Provides classes for managing interest rate term structures with various
day count conventions and interpolation methods.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum
from datetime import datetime, timedelta,date

import numpy as np
from scipy.interpolate import interp1d


class DayCountConvention(Enum):
    """Day count convention enumeration"""
    ACTUAL_360 = "Actual/360"
    ACTUAL_365 = "Actual/365"
    ACTUAL_ACTUAL = "Actual/Actual"
    THIRTY_360 = "30/360"
    THIRTY_360_EUROPEAN = "30E/360"

    @staticmethod
    def _is_leap_year(year: int) -> bool:
        """Check if year is a leap year"""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def actual_actual(self, start_date: datetime, end_date: datetime) -> float:
        days = (end_date - start_date).days
        # Simplified ACT/ACT - for production use ACT/ACT ISDA
        year1 = start_date.year
        year2 = end_date.year

        if year1 == year2:
            days_in_year = 366 if self._is_leap_year(year1) else 365
            return days / days_in_year
        else:
            # Handle multi-year periods
            total_frac = 0.0
            current = start_date
            while current.year < year2:
                year_end = date(current.year, 12, 31)
                days_to_year_end = (year_end - current).days + 1
                days_in_year = 366 if self._is_leap_year(current.year) else 365
                total_frac += days_to_year_end / days_in_year
                current = date(current.year + 1, 1, 1)

            days_in_final_year = (end_date - current).days
            days_in_year = 366 if self._is_leap_year(year2) else 365
            total_frac += days_in_final_year / days_in_year
            return total_frac

    def year_fraction(self, start_date: datetime, end_date: datetime) -> float:
        match self:
            case DayCountConvention.ACTUAL_360:
                return (end_date - start_date).days / 360.0
            case DayCountConvention.ACTUAL_365:
                return (end_date - start_date).days / 365.0
            case DayCountConvention.ACTUAL_ACTUAL:
                return self.actual_actual(start_date, end_date)
            case DayCountConvention.THIRTY_360:
                d1, m1, y1 = start_date.day, start_date.month, start_date.year
                d2, m2, y2 = end_date.day, end_date.month, end_date.year

                # 30/360 US (Bond Basis)
                if d1 == 31:
                    d1 = 30
                if d2 == 31 and d1 >= 30:
                    d2 = 30
                #Todo : Add Feb adjustment for 30/360 US
                days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
                return days / 360.0
            case DayCountConvention.THIRTY_360_EUROPEAN:
                # Common for Eurobonds and international corporate bonds
                d1, m1, y1 = start_date.day, start_date.month, start_date.year
                d2, m2, y2 = end_date.day, end_date.month, end_date.year

                if d1 == 31:
                    d1 = 30
                if d2 == 31:
                    d2 = 30

                days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
                return days / 360.0
            case _:
                raise ValueError(f"Unknown day count convention: {self.value}")


class InterpolationMethod(Enum):
    """Interpolation method enumeration"""
    LINEAR = "linear"
    LOG_LINEAR = "log_linear"
    CUBIC_SPLINE = "cubic_spline"
    FLAT_FORWARD = "flat_forward"

@dataclass
class TermStructure:
    """
    Represents an interest rate term structure with dates, year fractions,
    discount factors, and zero-coupon rates.

    Supports multiple day count conventions and interpolation methods.

    Attributes
    ----------
    dates : List[datetime]
        List of maturity dates
    discount_factors : List[float]
        List of discount factors corresponding to each date
    day_count_convention : DayCountConvention
        Day count convention to use (default: ACTUAL_365)
    interpolation_method : InterpolationMethod
        Interpolation method for rates between grid points (default: LOG_LINEAR)
    reference_date : datetime, optional
        Reference date for year fraction calculations (default: first date)
    year_fractions : List[float]
        Computed year fractions from reference date (auto-calculated)
    zero_rates : List[float]
        Computed zero-coupon rates (auto-calculated)
    """
    dates: List[datetime]
    discount_factors: List[float]
    day_count_convention: DayCountConvention = DayCountConvention.ACTUAL_365
    interpolation_method: InterpolationMethod = InterpolationMethod.LOG_LINEAR
    asof_date: Optional[datetime] = None

    # Computed fields
    year_fractions: List[float] = field(default_factory=list, init=False)
    zero_rates: List[float] = field(default_factory=list, init=False)
    _interpolator: Optional[Callable] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate and initialize the term structure"""
        # Validation
        if len(self.dates) != len(self.discount_factors):
            raise ValueError("dates and discount_factors must have the same length")

        if len(self.dates) < 2:
            raise ValueError("Need at least 2 points to define a term structure")

        if not all(self.dates[i] < self.dates[i + 1] for i in range(len(self.dates) - 1)):
            raise ValueError("dates must be strictly increasing")

        if not all(df > 0 for df in self.discount_factors):
            raise ValueError("Discount factors must be positive")

        if not all(df <= 1.0 for df in self.discount_factors):
            raise ValueError("Discount factors must be <= 1.0")

        # Set asof date if not provided
        if self.asof_date is None:
            self.reference_date = self.dates[0]

        # Compute year fractions
        self._compute_year_fractions()

        # Compute zero rates
        self._compute_zero_rates()

        # Setup interpolator
        self._setup_interpolator()

    def _compute_year_fractions(self):
        """Compute year fractions from reference date to each date"""
        self.year_fractions = [
            self.day_count_convention.year_fraction(self.asof_date, date)
            for date in self.dates
        ]

    def _compute_zero_rates(self):
        """Compute zero-coupon rates from discount factors and year fractions"""
        self.zero_rates = []
        for df, t in zip(self.discount_factors, self.year_fractions):
            if t > 0 and df > 0:
                rate = -np.log(df) / t
            else:
                rate = 0.0
            self.zero_rates.append(rate)

    def _setup_interpolator(self):
        """Setup the interpolator based on the selected method"""
        if self.interpolation_method == InterpolationMethod.LINEAR:
            self._interpolator = interp1d(
                self.year_fractions, self.discount_factors,
                kind='linear', fill_value='extrapolate', bounds_error=False
            )

        elif self.interpolation_method == InterpolationMethod.LOG_LINEAR:
            # Linear interpolation in log-space for stability
            log_dfs = np.log(self.discount_factors)
            self._interpolator = interp1d(
                self.year_fractions, log_dfs,
                kind='linear', fill_value='extrapolate', bounds_error=False
            )

        elif self.interpolation_method == InterpolationMethod.CUBIC_SPLINE:
            self._interpolator = interp1d(
                self.year_fractions, self.discount_factors,
                kind='cubic', fill_value='extrapolate', bounds_error=False
            )

        elif self.interpolation_method == InterpolationMethod.FLAT_FORWARD:
            # Flat forward rates
            self._interpolator = self._flat_forward_interpolator

        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")

    def _flat_forward_interpolator(self, t: float) -> float:
        """Flat forward rate interpolator"""
        # Find the two adjacent points
        if t <= self.year_fractions[0]:
            return self.discount_factors[0]

        for i in range(len(self.year_fractions) - 1):
            if self.year_fractions[i] <= t <= self.year_fractions[i + 1]:
                # Use the forward rate between these points
                t1, t2 = self.year_fractions[i], self.year_fractions[i + 1]
                df1, df2 = self.discount_factors[i], self.discount_factors[i + 1]

                # Implied forward rate
                if t2 > t1:
                    fwd_rate = -np.log(df2 / df1) / (t2 - t1)
                    df = df1 * np.exp(-fwd_rate * (t - t1))
                    return df
                else:
                    return df1

        # Beyond the last point, use the last forward rate
        if len(self.year_fractions) >= 2:
            t1, t2 = self.year_fractions[-2], self.year_fractions[-1]
            df1, df2 = self.discount_factors[-2], self.discount_factors[-1]

            if t2 > t1:
                fwd_rate = -np.log(df2 / df1) / (t2 - t1)
                df = df2 * np.exp(-fwd_rate * (t - t2))
                return df

        return self.discount_factors[-1]

    def discount_factor(self, maturity_date: datetime) -> float:
        """
        Get the discount factor for a given maturity date.

        Parameters
        ----------
        maturity_date : datetime
            The maturity date

        Returns
        -------
        float
            Discount factor at maturity
        """

        t = self.day_count_convention.year_fraction(self.asof_date, maturity_date)
        return self.discount_factor_at_time(t)

    def discount_factor_at_time(self, t: float) -> float:
        """
        Get the discount factor at time t (in years).

        Parameters
        ----------
        t : float
            Time in years

        Returns
        -------
        float
            Discount factor at time t
        """
        if t <= 0:
            return 1.0

        if self.interpolation_method == InterpolationMethod.LOG_LINEAR:
            log_df = self._interpolator(t)
            return float(np.exp(log_df))
        else:
            return float(self._interpolator(t))

    def zero_rate(self, maturity_date: datetime) -> float:
        """
        Get the zero-coupon rate for a given maturity date.

        Parameters
        ----------
        maturity_date : datetime
            The maturity date

        Returns
        -------
        float
            Zero-coupon rate (continuously compounded)
        """

        t = self.day_count_convention.year_fraction(self.asof_date, maturity_date)
        return self.zero_rate_at_time(t)

    def zero_rate_at_time(self, t: float) -> float:
        """
        Get the zero-coupon rate at time t (in years).

        Parameters
        ----------
        t : float
            Time in years

        Returns
        -------
        float
            Zero-coupon rate (continuously compounded)
        """
        if t <= 0:
            return 0.0

        df = self.discount_factor_at_time(t)
        if df > 0:
            return -np.log(df) / t
        else:
            return 0.0

    def forward_rate(self, start_date: datetime, end_date: datetime) -> float:
        """
        Get the forward rate between two dates.

        Parameters
        ----------
        start_date : datetime
            Start date
        end_date : datetime
            End date

        Returns
        -------
        float
            Forward rate (continuously compounded)
        """

        t1 = self.day_count_convention.year_fraction(self.asof_date, start_date)
        t2 = self.day_count_convention.year_fraction(self.asof_date, end_date)
        return self.forward_rate_at_time(t1, t2)

    def forward_rate_at_time(self, t1: float, t2: float) -> float:
        """
        Get the forward rate between two times.

        Parameters
        ----------
        t1 : float
            Start time in years
        t2 : float
            End time in years

        Returns
        -------
        float
            Forward rate (continuously compounded)
        """
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")

        df1 = self.discount_factor_at_time(t1)
        df2 = self.discount_factor_at_time(t2)

        if df1 > 0 and df2 > 0:
            return -np.log(df2 / df1) / (t2 - t1)
        else:
            return 0.0

    @classmethod
    def flat_curve(cls, rate: float, reference_date: datetime,
                   day_count_convention: DayCountConvention = DayCountConvention.ACTUAL_365,
                   max_tenor_years: float = 30.0) -> 'TermStructure':
        """
        Create a flat term structure.

        Parameters
        ----------
        rate : float
            The flat zero-coupon rate
        reference_date : datetime
            The reference date
        day_count_convention : DayCountConvention
            Day count convention
        max_tenor_years : float
            Maximum tenor in years

        Returns
        -------
        TermStructure
            A flat term structure
        """
        times = np.array([0.001, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0,15.0, max_tenor_years])
        dfs = np.exp(-rate * times)

        dates = [reference_date + timedelta(days=int(t * 365.25)) for t in times]

        return cls(
            dates=dates,
            discount_factors=dfs.tolist(),
            day_count_convention=day_count_convention,
            interpolation_method=InterpolationMethod.LOG_LINEAR,
            asof_date=reference_date
        )

    @classmethod
    def from_zero_rates(cls, dates: List[datetime], zero_rates: List[float],
                        day_count_convention: DayCountConvention = DayCountConvention.ACTUAL_365,
                        reference_date: Optional[datetime] = None) -> 'TermStructure':
        """
        Create a term structure from zero-coupon rates.

        Parameters
        ----------
        dates : List[datetime]
            Maturity dates
        zero_rates : List[float]
            Zero-coupon rates (continuously compounded)
        day_count_convention : DayCountConvention
            Day count convention
        reference_date : datetime, optional
            Reference date (defaults to first date)

        Returns
        -------
        TermStructure
            A term structure constructed from zero rates
        """
        if reference_date is None:
            reference_date = dates[0]

        # Calculate discount factors from zero rates
        term_structure = cls(dates=dates, discount_factors=[0.0] * len(dates),
                  day_count_convention=day_count_convention, asof_date=reference_date)
        dfs = []
        for date, rate in zip(dates, zero_rates):
            t = term_structure.day_count_convention.year_fraction(reference_date, date)
            df = np.exp(-rate * t)
            dfs.append(float(df))
        term_structure.discount_factors = dfs
        return term_structure

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"TermStructure(reference_date={self.reference_date.date()}, "
            f"points={len(self.dates)}, "
            f"day_count={self.day_count_convention.value}, "
            f"interpolation={self.interpolation_method.value})"
        )

    def summary(self) -> str:
        """Return a formatted summary of the term structure"""
        lines = [
            f"Term Structure Summary",
            f"{'=' * 70}",
            f"Reference Date: {self.asof_date.date()}",
            f"Day Count Convention: {self.day_count_convention.value}",
            f"Interpolation Method: {self.interpolation_method.value}",
            f"Number of Points: {len(self.dates)}",
            f"",
            f"{'Date':<12} {'Year Frac':<12} {'Discount Factor':<18} {'Zero Rate':<12}",
            f"{'-' * 70}",
        ]

        for date, t, df, rate in zip(self.dates, self.year_fractions, self.discount_factors, self.zero_rates):
            lines.append(f"{date.date()!s:<12} {t:<12.6f} {df:<18.6f} {rate*100:<12.4f}%")

        return "\n".join(lines)
