
"""
Government Bond Pricing Library
Industry-standard implementation for quantitative finance

Features:
- Clean/Dirty price calculations
- Yield to Maturity (YTM) calculations
- Duration (Macaulay, Modified)
- Convexity
- Accrued interest
- Price from yield and yield from price
- Support for multiple day count conventions
- Support for multiple compounding frequencies
"""

from datetime import datetime, date
from typing import Union, Literal, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.optimize import newton, brentq


class DayCountConvention(Enum):
    """Day count conventions for bond calculations"""
    ACTUAL_ACTUAL = "ACT/ACT"
    ACTUAL_360 = "ACT/360"
    ACTUAL_365 = "ACT/365"
    THIRTY_360 = "30/360"
    THIRTY_360_EUROPEAN = "30E/360"


class Frequency(Enum):
    """Coupon payment frequency"""
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12


@dataclass
class BondSpecs:
    """Bond specification data class"""
    face_value: float
    coupon_rate: float  # Annual coupon rate as decimal (e.g., 0.05 for 5%)
    maturity_date: Union[datetime, date]
    settlement_date: Union[datetime, date]
    frequency: Frequency = Frequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACTUAL_ACTUAL

    def __post_init__(self):
        """Validate bond specifications"""
        if self.face_value <= 0:
            raise ValueError("Face value must be positive")
        if self.coupon_rate < 0:
            raise ValueError("Coupon rate cannot be negative")
        if self.maturity_date <= self.settlement_date:
            raise ValueError("Maturity date must be after settlement date")


class BondPricer:
    """
    Government bond pricing engine

    Implements industry-standard bond pricing methodologies used by
    Bloomberg, Reuters, and institutional trading desks.
    """

    def __init__(self, bond: BondSpecs):
        """
        Initialize bond pricer with bond specifications

        Args:
            bond: BondSpecs object containing bond details
        """
        self.bond = bond
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate bond specifications"""
        if not isinstance(self.bond, BondSpecs):
            raise TypeError("bond must be a BondSpecs instance")

    def _year_fraction(self, start_date: Union[datetime, date],
                       end_date: Union[datetime, date]) -> float:
        """
        Calculate year fraction between two dates using specified day count convention

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Year fraction as float
        """
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        if self.bond.day_count == DayCountConvention.ACTUAL_ACTUAL:
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

        elif self.bond.day_count == DayCountConvention.ACTUAL_360:
            days = (end_date - start_date).days
            return days / 360.0

        elif self.bond.day_count == DayCountConvention.ACTUAL_365:
            days = (end_date - start_date).days
            return days / 365.0

        elif self.bond.day_count == DayCountConvention.THIRTY_360:
            d1, m1, y1 = start_date.day, start_date.month, start_date.year
            d2, m2, y2 = end_date.day, end_date.month, end_date.year

            # 30/360 US (Bond Basis)
            if d1 == 31:
                d1 = 30
            if d2 == 31 and d1 >= 30:
                d2 = 30

            days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
            return days / 360.0

        else:  # THIRTY_360_EUROPEAN
            d1, m1, y1 = start_date.day, start_date.month, start_date.year
            d2, m2, y2 = end_date.day, end_date.month, end_date.year

            if d1 == 31:
                d1 = 30
            if d2 == 31:
                d2 = 30

            days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
            return days / 360.0

    @staticmethod
    def _is_leap_year(year: int) -> bool:
        """Check if year is a leap year"""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

    def _get_coupon_dates(self) -> list[date]:
        """
        Generate all coupon payment dates from settlement to maturity

        Returns:
            List of coupon payment dates
        """
        coupon_dates = []

        if isinstance(self.bond.maturity_date, datetime):
            maturity = self.bond.maturity_date.date()
        else:
            maturity = self.bond.maturity_date

        if isinstance(self.bond.settlement_date, datetime):
            settlement = self.bond.settlement_date.date()
        else:
            settlement = self.bond.settlement_date

        # Work backwards from maturity
        freq = self.bond.frequency.value
        months_between = 12 // freq

        current_date = maturity
        while current_date > settlement:
            coupon_dates.append(current_date)
            # Go back by the payment frequency
            year = current_date.year
            month = current_date.month - months_between

            while month <= 0:
                month += 12
                year -= 1

            try:
                current_date = date(year, month, current_date.day)
            except ValueError:
                # Handle month-end dates (e.g., Jan 31 -> Feb 28/29)
                import calendar
                last_day = calendar.monthrange(year, month)[1]
                current_date = date(year, month, min(current_date.day, last_day))

        return sorted(coupon_dates)

    def accrued_interest(self) -> float:
        """
        Calculate accrued interest from last coupon date to settlement date

        Returns:
            Accrued interest amount
        """
        coupon_dates = self._get_coupon_dates()

        if not coupon_dates:
            return 0.0

        # Find the last coupon date before settlement
        settlement = (self.bond.settlement_date.date()
                      if isinstance(self.bond.settlement_date, datetime)
                      else self.bond.settlement_date)

        last_coupon_date = None
        next_coupon_date = None

        for i, cpn_date in enumerate(coupon_dates):
            if cpn_date > settlement:
                next_coupon_date = cpn_date
                if i > 0:
                    last_coupon_date = coupon_dates[i - 1]
                else:
                    # Settlement is before first coupon - need to look back
                    freq = self.bond.frequency.value
                    months_between = 12 // freq
                    year = cpn_date.year
                    month = cpn_date.month - months_between

                    while month <= 0:
                        month += 12
                        year -= 1

                    import calendar
                    last_day = calendar.monthrange(year, month)[1]
                    last_coupon_date = date(year, month, min(cpn_date.day, last_day))
                break

        if last_coupon_date is None:
            return 0.0

        # Calculate accrued interest
        coupon_payment = self.bond.face_value * self.bond.coupon_rate / self.bond.frequency.value
        accrued_fraction = self._year_fraction(last_coupon_date, settlement)
        period_fraction = self._year_fraction(last_coupon_date, next_coupon_date)

        return coupon_payment * (accrued_fraction / period_fraction)

    def price_from_yield(self, ytm: float, clean: bool = True) -> float:
        """
        Calculate bond price from yield to maturity

        Args:
            ytm: Yield to maturity as decimal (e.g., 0.05 for 5%)
            clean: If True, return clean price; if False, return dirty price

        Returns:
            Bond price (as percentage of par if face_value=100, or absolute value)
        """
        coupon_dates = self._get_coupon_dates()

        if not coupon_dates:
            # Zero-coupon bond
            settlement = (self.bond.settlement_date.date()
                          if isinstance(self.bond.settlement_date, datetime)
                          else self.bond.settlement_date)
            maturity = (self.bond.maturity_date.date()
                        if isinstance(self.bond.maturity_date, datetime)
                        else self.bond.maturity_date)

            years_to_maturity = self._year_fraction(settlement, maturity)
            dirty_price = self.bond.face_value / ((1 + ytm / self.bond.frequency.value)
                                                  ** (years_to_maturity * self.bond.frequency.value))

            return dirty_price if not clean else dirty_price - self.accrued_interest()

        # Calculate present value of all cash flows
        settlement = (self.bond.settlement_date.date()
                      if isinstance(self.bond.settlement_date, datetime)
                      else self.bond.settlement_date)

        coupon_payment = self.bond.face_value * self.bond.coupon_rate / self.bond.frequency.value
        pv = 0.0

        for i, cpn_date in enumerate(coupon_dates):
            years_to_payment = self._year_fraction(settlement, cpn_date)
            periods_to_payment = years_to_payment * self.bond.frequency.value
            discount_factor = 1.0/((1 + ytm / self.bond.frequency.value) ** periods_to_payment)

            # Add coupon payment
            pv += coupon_payment * discount_factor

            # Add principal repayment at maturity
            if i == len(coupon_dates) - 1:
                pv += self.bond.face_value * discount_factor

        dirty_price = pv

        if clean:
            return dirty_price - self.accrued_interest()
        else:
            return dirty_price

    def yield_from_price(self, price: float, clean: bool = True,
                         initial_guess: float = 0.05) -> float:
        """
        Calculate yield to maturity from bond price (Newton-Raphson method)

        Args:
            price: Bond price (clean or dirty as specified)
            clean: If True, price is clean price; if False, price is dirty price
            initial_guess: Initial guess for YTM

        Returns:
            Yield to maturity as decimal
        """

        def objective(ytm):
            calculated_price = self.price_from_yield(ytm, clean=clean)
            return calculated_price - price

        def derivative(ytm):
            # Numerical derivative
            h = 1e-8
            return (self.price_from_yield(ytm + h, clean=clean) -
                    self.price_from_yield(ytm - h, clean=clean)) / (2 * h)

        try:
            # Try Newton-Raphson first (faster convergence)
            ytm = newton(objective, initial_guess, fprime=derivative,
                         maxiter=100, tol=1e-10)
        except:
            # Fall back to Brent's method (more robust)
            try:
                ytm = brentq(objective, -0.5, 2.0, maxiter=100, xtol=1e-10)
            except:
                raise ValueError("Could not converge to a yield solution")

        return ytm

    def macaulay_duration(self, ytm: float) -> float:
        """
        Calculate Macaulay duration

        Args:
            ytm: Yield to maturity as decimal

        Returns:
            Macaulay duration in years
        """
        coupon_dates = self._get_coupon_dates()
        settlement = (self.bond.settlement_date.date()
                      if isinstance(self.bond.settlement_date, datetime)
                      else self.bond.settlement_date)

        coupon_payment = self.bond.face_value * self.bond.coupon_rate / self.bond.frequency.value

        weighted_pv = 0.0
        total_pv = 0.0

        for i, cpn_date in enumerate(coupon_dates):
            years_to_payment = self._year_fraction(settlement, cpn_date)
            periods_to_payment = years_to_payment * self.bond.frequency.value
            discount_factor = (1 + ytm / self.bond.frequency.value) ** periods_to_payment

            cash_flow = coupon_payment
            if i == len(coupon_dates) - 1:
                cash_flow += self.bond.face_value

            pv = cash_flow / discount_factor
            weighted_pv += years_to_payment * pv
            total_pv += pv

        return weighted_pv / total_pv

    def modified_duration(self, ytm: float) -> float:
        """
        Calculate Modified duration (price sensitivity to yield changes)

        Args:
            ytm: Yield to maturity as decimal

        Returns:
            Modified duration in years
        """
        mac_dur = self.macaulay_duration(ytm)
        return mac_dur / (1 + ytm / self.bond.frequency.value)

    def convexity(self, ytm: float) -> float:
        """
        Calculate convexity (second-order price sensitivity)

        Args:
            ytm: Yield to maturity as decimal

        Returns:
            Convexity
        """
        coupon_dates = self._get_coupon_dates()
        settlement = (self.bond.settlement_date.date()
                      if isinstance(self.bond.settlement_date, datetime)
                      else self.bond.settlement_date)

        coupon_payment = self.bond.face_value * self.bond.coupon_rate / self.bond.frequency.value
        freq = self.bond.frequency.value

        weighted_pv = 0.0
        total_pv = 0.0

        for i, cpn_date in enumerate(coupon_dates):
            years_to_payment = self._year_fraction(settlement, cpn_date)
            periods_to_payment = years_to_payment * freq
            discount_factor = (1 + ytm / freq) ** periods_to_payment

            cash_flow = coupon_payment
            if i == len(coupon_dates) - 1:
                cash_flow += self.bond.face_value

            pv = cash_flow / discount_factor
            weighted_pv += pv * periods_to_payment * (periods_to_payment + 1)
            total_pv += pv

        return weighted_pv / (total_pv * (freq ** 2) * ((1 + ytm / freq) ** 2))

    def dv01(self, ytm: float) -> float:
        """
        Calculate DV01 (Dollar Value of 1 basis point)
        Price change for 1bp (0.01%) yield change

        Args:
            ytm: Yield to maturity as decimal

        Returns:
            DV01 value
        """
        mod_dur = self.modified_duration(ytm)
        price = self.price_from_yield(ytm, clean=True)
        return mod_dur * price * 0.0001  # 1bp = 0.01% = 0.0001

    def get_all_metrics(self, ytm: Optional[float] = None,
                        price: Optional[float] = None) -> dict:
        """
        Calculate all bond metrics at once

        Provide either ytm or price (not both)

        Args:
            ytm: Yield to maturity (optional)
            price: Clean price (optional)

        Returns:
            Dictionary with all metrics
        """
        if ytm is None and price is None:
            raise ValueError("Must provide either ytm or price")
        if ytm is not None and price is not None:
            raise ValueError("Provide either ytm or price, not both")

        if price is not None:
            ytm = self.yield_from_price(price, clean=True)

        clean_price = self.price_from_yield(ytm, clean=True)
        dirty_price = self.price_from_yield(ytm, clean=False)
        accrued = self.accrued_interest()
        mac_dur = self.macaulay_duration(ytm)
        mod_dur = self.modified_duration(ytm)
        cvx = self.convexity(ytm)
        dollar_value_01 = self.dv01(ytm)

        return {
            'ytm': ytm,
            'clean_price': clean_price,
            'dirty_price': dirty_price,
            'accrued_interest': accrued,
            'macaulay_duration': mac_dur,
            'modified_duration': mod_dur,
            'convexity': cvx,
            'dv01': dollar_value_01,
            'settlement_date': self.bond.settlement_date,
            'maturity_date': self.bond.maturity_date,
            'years_to_maturity': self._year_fraction(
                self.bond.settlement_date.date() if isinstance(self.bond.settlement_date,
                                                               datetime) else self.bond.settlement_date,
                self.bond.maturity_date.date() if isinstance(self.bond.maturity_date,
                                                             datetime) else self.bond.maturity_date
            )
        }


def create_us_treasury(face_value: float, coupon_rate: float,
                       maturity_date: Union[datetime, date],
                       settlement_date: Union[datetime, date]) -> BondPricer:
    """
    Convenience function to create a US Treasury bond pricer

    US Treasuries use ACT/ACT day count and semi-annual coupons

    Args:
        face_value: Face value of the bond
        coupon_rate: Annual coupon rate as decimal
        maturity_date: Maturity date
        settlement_date: Settlement date

    Returns:
        BondPricer instance
    """
    bond = BondSpecs(
        face_value=face_value,
        coupon_rate=coupon_rate,
        maturity_date=maturity_date,
        settlement_date=settlement_date,
        frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.ACTUAL_ACTUAL
    )
    return BondPricer(bond)


def create_uk_gilt(face_value: float, coupon_rate: float,
                   maturity_date: Union[datetime, date],
                   settlement_date: Union[datetime, date]) -> BondPricer:
    """
    Convenience function to create a UK Gilt pricer

    UK Gilts use ACT/ACT day count and semi-annual coupons

    Args:
        face_value: Face value of the bond
        coupon_rate: Annual coupon rate as decimal
        maturity_date: Maturity date
        settlement_date: Settlement date

    Returns:
        BondPricer instance
    """
    bond = BondSpecs(
        face_value=face_value,
        coupon_rate=coupon_rate,
        maturity_date=maturity_date,
        settlement_date=settlement_date,
        frequency=Frequency.SEMI_ANNUAL,
        day_count=DayCountConvention.ACTUAL_ACTUAL
    )
    return BondPricer(bond)


def create_german_bund(face_value: float, coupon_rate: float,
                       maturity_date: Union[datetime, date],
                       settlement_date: Union[datetime, date]) -> BondPricer:
    """
    Convenience function to create a German Bund pricer

    German Bunds use ACT/ACT day count and annual coupons

    Args:
        face_value: Face value of the bond
        coupon_rate: Annual coupon rate as decimal
        maturity_date: Maturity date
        settlement_date: Settlement date

    Returns:
        BondPricer instance
    """
    bond = BondSpecs(
        face_value=face_value,
        coupon_rate=coupon_rate,
        maturity_date=maturity_date,
        settlement_date=settlement_date,
        frequency=Frequency.ANNUAL,
        day_count=DayCountConvention.ACTUAL_ACTUAL
    )
    return BondPricer(bond)
