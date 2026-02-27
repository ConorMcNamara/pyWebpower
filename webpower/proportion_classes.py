from math import ceil, sqrt

from scipy.optimize import brentq
from scipy.stats import norm


class WpOneProp:
    def __init__(
        self,
        h: float | None,
        n: int | None,
        alpha: float | None,
        power: float | None,
        alternative: str = "two-sided",
    ) -> None:
        self.h = h
        self.n = n
        self.alpha = alpha
        self.power = power
        self.alternative = alternative.casefold()
        self.method = "Power for one-sample proportion test"
        self.note = "NOTE: Sample size for each group"
        self.url = "http://psychstat.org/prop"

    def _get_power(self) -> float:
        if self.alternative == "two-sided":
            power = norm.sf(norm.isf(self.alpha / 2) - self.h * sqrt(self.n)) + norm.cdf(
                norm.ppf(self.alpha / 2) - self.h * sqrt(self.n)
            )
        elif self.alternative == "greater":
            power = norm.sf(norm.isf(self.alpha) - self.h * sqrt(self.n))
        else:
            power = norm.cdf(norm.ppf(self.alpha) - self.h * sqrt(self.n))
        return float(power)

    def _get_effect_size(self, h: float) -> float:
        if self.alternative == "two-sided":
            h = (
                norm.sf(norm.isf(self.alpha / 2) - h * sqrt(self.n))
                + norm.cdf(norm.ppf(self.alpha / 2) - h * sqrt(self.n))
                - self.power
            )
        elif self.alternative == "greater":
            h = norm.sf(norm.isf(self.alpha) - h * sqrt(self.n)) - self.power
        else:
            h = norm.cdf(norm.ppf(self.alpha) - h * sqrt(self.n)) - self.power
        return float(h)

    def _get_n(self, n: int) -> float:
        if self.alternative == "two-sided":
            n = (
                norm.sf(norm.isf(self.alpha / 2) - self.h * sqrt(n))
                + norm.cdf(norm.ppf(self.alpha / 2) - self.h * sqrt(n))
                - self.power
            )
        elif self.alternative == "greater":
            n = norm.sf(norm.isf(self.alpha) - self.h * sqrt(n)) - self.power
        else:
            n = norm.cdf(norm.ppf(self.alpha) - self.h * sqrt(n)) - self.power
        return float(n)

    def _get_alpha(self, alpha: float) -> float:
        if self.alternative == "two-sided":
            alpha = (
                norm.sf(norm.isf(alpha / 2) - self.h * sqrt(self.n))
                + norm.cdf(norm.ppf(alpha / 2) - self.h * sqrt(self.n))
                - self.power
            )
        elif self.alternative == "greater":
            alpha = norm.sf(norm.isf(alpha) - self.h * sqrt(self.n)) - self.power
        else:
            alpha = norm.cdf(norm.ppf(alpha) - self.h * sqrt(self.n)) - self.power
        return float(alpha)

    def pwr_test(self) -> dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.h is None:
            if self.alternative == "two-sided":
                self.h = brentq(self._get_effect_size, 1e-10, 10)
            elif self.alternative == "greater":
                self.h = brentq(self._get_effect_size, -5, 10)
            else:
                self.h = brentq(self._get_effect_size, -10, 5)
        elif self.n is None:
            self.n = ceil(brentq(self._get_n, 2 + 1e-10, 1e09))
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
        return {
            "effect_size": self.h,
            "n": self.n,
            "alpha": self.alpha,
            "power": self.power,
            "alternative": self.alternative,
            "method": self.method,
            "note": self.note,
            "url": self.url,
        }


class WpTwoPropOneN:
    def __init__(
        self,
        h: float | None,
        n: int | None,
        alpha: float | None,
        power: float | None,
        alternative: str = "two-sided",
    ) -> None:
        self.h = h
        self.n = n
        self.alpha = alpha
        self.power = power
        self.alternative = alternative.casefold()
        self.method = "Power for two-sample proportion (equal n)"
        self.note = "NOTE: Sample sizes for EACH group"
        self.url = "http://psychstat.org/prop2p"

    def _get_power(self) -> float:
        if self.alternative == "two-sided":
            power = norm.sf(norm.isf(self.alpha / 2) - self.h * sqrt(self.n / 2)) + norm.cdf(
                norm.ppf(self.alpha / 2) - self.h * sqrt(self.n / 2)
            )
        elif self.alternative == "greater":
            power = norm.sf(norm.isf(self.alpha) - self.h * sqrt(self.n / 2))
        else:
            power = norm.cdf(norm.ppf(self.alpha) - self.h * sqrt(self.n / 2))
        return float(power)

    def _get_effect_size(self, h: float) -> float:
        if self.alternative == "two-sided":
            h = (
                norm.sf(norm.isf(self.alpha / 2) - h * sqrt(self.n / 2))
                + norm.cdf(norm.ppf(self.alpha / 2) - h * sqrt(self.n / 2))
                - self.power
            )
        elif self.alternative == "greater":
            h = norm.sf(norm.isf(self.alpha) - h * sqrt(self.n / 2)) - self.power
        else:
            h = norm.cdf(norm.ppf(self.alpha) - h * sqrt(self.n / 2)) - self.power
        return float(h)

    def _get_n(self, n: int) -> float:
        if self.alternative == "two-sided":
            n = (
                norm.sf(norm.isf(self.alpha / 2) - self.h * sqrt(n / 2))
                + norm.cdf(norm.ppf(self.alpha / 2) - self.h * sqrt(n / 2))
                - self.power
            )
        elif self.alternative == "greater":
            n = norm.sf(norm.isf(self.alpha) - self.h * sqrt(n / 2)) - self.power
        else:
            n = norm.cdf(norm.ppf(self.alpha) - self.h * sqrt(n / 2)) - self.power
        return float(n)

    def _get_alpha(self, alpha: float) -> float:
        if self.alternative == "two-sided":
            alpha = (
                norm.sf(norm.isf(alpha / 2) - self.h * sqrt(self.n / 2))
                + norm.cdf(norm.ppf(alpha / 2) - self.h * sqrt(self.n / 2))
                - self.power
            )
        elif self.alternative == "greater":
            alpha = norm.sf(norm.isf(alpha) - self.h * sqrt(self.n / 2)) - self.power
        else:
            alpha = norm.cdf(norm.ppf(alpha) - self.h * sqrt(self.n / 2)) - self.power
        return float(alpha)

    def pwr_test(self) -> dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.h is None:
            if self.alternative == "two-sided":
                self.h = brentq(self._get_effect_size, 1e-10, 10)
            elif self.alternative == "greater":
                self.h = brentq(self._get_effect_size, -5, 10)
            else:
                self.h = brentq(self._get_effect_size, -10, 5)
        elif self.n is None:
            self.n = ceil(brentq(self._get_n, 2 + 1e-10, 1e09))
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
        return {
            "effect_size": self.h,
            "n": self.n,
            "alpha": self.alpha,
            "power": self.power,
            "alternative": self.alternative,
            "method": self.method,
            "note": self.note,
            "url": self.url,
        }


class WpTwoPropTwoN:
    def __init__(
        self,
        h: float | None,
        n1: int | None,
        n2: int | None,
        alpha: float | None,
        power: float | None,
        alternative: str = "two-sided",
    ) -> None:
        self.h = h
        self.n1 = n1
        self.n2 = n2
        self.alpha = alpha
        self.power = power
        self.alternative = alternative.casefold()
        self.method = "Power for two-sample proportion (unequal n)"
        self.note = "NOTE: Sample size for each group"
        self.url = "http://psychstat.org/prop2p2n"

    def _get_power(self) -> float:
        if self.alternative == "two-sided":
            power = norm.sf(
                norm.isf(self.alpha / 2) - self.h * sqrt(self.n1 * self.n2 / (self.n1 + self.n2))
            ) + norm.cdf(norm.ppf(self.alpha / 2) - self.h * sqrt(self.n1 * self.n2 / (self.n1 + self.n2)))
        elif self.alternative == "greater":
            power = norm.sf(norm.isf(self.alpha) - self.h * sqrt(self.n1 * self.n2 / (self.n1 + self.n2)))
        else:
            power = norm.cdf(norm.ppf(self.alpha) - self.h * sqrt(self.n1 * self.n2 / (self.n1 + self.n2)))
        return float(power)

    def _get_effect_size(self, h: float) -> float:
        if self.alternative == "two-sided":
            h = (
                norm.sf(norm.isf(self.alpha / 2) - h * sqrt(self.n1 * self.n2 / (self.n1 + self.n2)))
                + norm.cdf(norm.ppf(self.alpha / 2) - h * sqrt(self.n1 * self.n2 / (self.n1 + self.n2)))
                - self.power
            )
        elif self.alternative == "greater":
            h = norm.sf(norm.isf(self.alpha) - h * sqrt(self.n1 * self.n2 / (self.n1 + self.n2))) - self.power
        else:
            h = norm.cdf(norm.ppf(self.alpha) - h * sqrt(self.n1 * self.n2 / (self.n1 + self.n2))) - self.power
        return float(h)

    def _get_n1(self, n1: int) -> float:
        if self.alternative == "two-sided":
            n1 = (
                norm.sf(norm.isf(self.alpha / 2) - self.h * sqrt(n1 * self.n2 / (n1 + self.n2)))
                + norm.cdf(norm.ppf(self.alpha / 2) - self.h * sqrt(n1 * self.n2 / (n1 + self.n2)))
                - self.power
            )
        elif self.alternative == "greater":
            n1 = norm.sf(norm.isf(self.alpha) - self.h * sqrt(n1 * self.n2 / (n1 + self.n2))) - self.power
        else:
            n1 = norm.cdf(norm.ppf(self.alpha) - self.h * sqrt(n1 * self.n2 / (n1 + self.n2))) - self.power
        return float(n1)

    def _get_n2(self, n2: int) -> float:
        if self.alternative == "two-sided":
            n2 = (
                norm.sf(norm.isf(self.alpha / 2) - self.h * sqrt(self.n1 * n2 / (self.n1 + n2)))
                + norm.cdf(norm.ppf(self.alpha / 2) - self.h * sqrt(self.n1 * n2 / (self.n1 + n2)))
                - self.power
            )
        elif self.alternative == "greater":
            n2 = norm.sf(norm.isf(self.alpha) - self.h * sqrt(self.n1 * n2 / (self.n1 + n2))) - self.power
        else:
            n2 = norm.cdf(norm.ppf(self.alpha) - self.h * sqrt(self.n1 * n2 / (self.n1 + n2))) - self.power
        return float(n2)

    def _get_alpha(self, alpha: float) -> float:
        if self.alternative == "two-sided":
            alpha = (
                norm.sf(norm.isf(alpha / 2) - self.h * sqrt(self.n1 * self.n2 / (self.n1 + self.n2)))
                + norm.cdf(norm.ppf(alpha / 2) - self.h * sqrt(self.n1 * self.n2 / (self.n1 + self.n2)))
                - self.power
            )
        elif self.alternative == "greater":
            alpha = norm.sf(norm.isf(alpha) - self.h * sqrt(self.n1 * self.n2 / (self.n1 + self.n2))) - self.power
        else:
            alpha = norm.cdf(norm.ppf(alpha) - self.h * sqrt(self.n1 * self.n2 / (self.n1 + self.n2))) - self.power
        return float(alpha)

    def pwr_test(self) -> dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.h is None:
            if self.alternative == "two-sided":
                self.h = brentq(self._get_effect_size, 1e-10, 10)
            elif self.alternative == "greater":
                self.h = brentq(self._get_effect_size, -5, 10)
            else:
                self.h = brentq(self._get_effect_size, -10, 5)
        elif self.n1 is None:
            self.n1 = ceil(brentq(self._get_n1, 2 + 1e-10, 1e09))
        elif self.n2 is None:
            self.n2 = ceil(brentq(self._get_n2, 2 + 1e-10, 1e09))
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
        return {
            "effect_size": self.h,
            "n1": self.n1,
            "n2": self.n2,
            "alpha": self.alpha,
            "power": self.power,
            "alternative": self.alternative,
            "method": self.method,
            "note": self.note,
            "url": self.url,
        }
