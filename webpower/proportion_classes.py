from math import ceil, sqrt

from scipy.stats import norm

from webpower.utils import brentq


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

    def _get_power(self, h: float, n: int, alpha: float) -> float:
        if self.alternative == "two-sided":
            power = norm.sf(norm.isf(alpha / 2) - h * sqrt(n)) + norm.cdf(norm.ppf(alpha / 2) - h * sqrt(n))
        elif self.alternative == "greater":
            power = norm.sf(norm.isf(alpha) - h * sqrt(n))
        else:
            power = norm.cdf(norm.ppf(alpha) - h * sqrt(n))
        return float(power)

    def _get_effect_size(self, h: float, n: int, alpha: float, power: float) -> float:
        if self.alternative == "two-sided":
            result = norm.sf(norm.isf(alpha / 2) - h * sqrt(n)) + norm.cdf(norm.ppf(alpha / 2) - h * sqrt(n)) - power
        elif self.alternative == "greater":
            result = norm.sf(norm.isf(alpha) - h * sqrt(n)) - power
        else:
            result = norm.cdf(norm.ppf(alpha) - h * sqrt(n)) - power
        return float(result)

    def _get_n(self, n: float, h: float, alpha: float, power: float) -> float:
        if self.alternative == "two-sided":
            result = norm.sf(norm.isf(alpha / 2) - h * sqrt(n)) + norm.cdf(norm.ppf(alpha / 2) - h * sqrt(n)) - power
        elif self.alternative == "greater":
            result = norm.sf(norm.isf(alpha) - h * sqrt(n)) - power
        else:
            result = norm.cdf(norm.ppf(alpha) - h * sqrt(n)) - power
        return float(result)

    def _get_alpha(self, alpha: float, h: float, n: int, power: float) -> float:
        if self.alternative == "two-sided":
            result = norm.sf(norm.isf(alpha / 2) - h * sqrt(n)) + norm.cdf(norm.ppf(alpha / 2) - h * sqrt(n)) - power
        elif self.alternative == "greater":
            result = norm.sf(norm.isf(alpha) - h * sqrt(n)) - power
        else:
            result = norm.cdf(norm.ppf(alpha) - h * sqrt(n)) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.h is None or self.n is None or self.alpha is None:
                raise ValueError("h, n, and alpha must be provided to compute power")
            self.power = self._get_power(self.h, self.n, self.alpha)
        elif self.h is None:
            if self.n is None or self.alpha is None or self.power is None:
                raise ValueError("n, alpha, and power must be provided to solve for h")
            n, alpha, power = self.n, self.alpha, self.power
            if self.alternative == "two-sided":
                self.h = brentq(lambda h: self._get_effect_size(h, n, alpha, power), 1e-10, 10)
            elif self.alternative == "greater":
                self.h = brentq(lambda h: self._get_effect_size(h, n, alpha, power), -5, 10)
            else:
                self.h = brentq(lambda h: self._get_effect_size(h, n, alpha, power), -10, 5)
        elif self.n is None:
            if self.h is None or self.alpha is None or self.power is None:
                raise ValueError("h, alpha, and power must be provided to solve for n")
            h, alpha, power = self.h, self.alpha, self.power
            self.n = ceil(brentq(lambda n: self._get_n(n, h, alpha, power), 2 + 1e-10, 1e09))
        else:
            if self.h is None or self.n is None or self.power is None:
                raise ValueError("h, n, and power must be provided to solve for alpha")
            h, n, power = self.h, self.n, self.power
            self.alpha = brentq(lambda alpha: self._get_alpha(alpha, h, n, power), 1e-10, 1 - 1e-10)
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

    def _get_power(self, h: float, n: int, alpha: float) -> float:
        if self.alternative == "two-sided":
            power = norm.sf(norm.isf(alpha / 2) - h * sqrt(n / 2)) + norm.cdf(norm.ppf(alpha / 2) - h * sqrt(n / 2))
        elif self.alternative == "greater":
            power = norm.sf(norm.isf(alpha) - h * sqrt(n / 2))
        else:
            power = norm.cdf(norm.ppf(alpha) - h * sqrt(n / 2))
        return float(power)

    def _get_effect_size(self, h: float, n: int, alpha: float, power: float) -> float:
        if self.alternative == "two-sided":
            result = (
                norm.sf(norm.isf(alpha / 2) - h * sqrt(n / 2)) + norm.cdf(norm.ppf(alpha / 2) - h * sqrt(n / 2)) - power
            )
        elif self.alternative == "greater":
            result = norm.sf(norm.isf(alpha) - h * sqrt(n / 2)) - power
        else:
            result = norm.cdf(norm.ppf(alpha) - h * sqrt(n / 2)) - power
        return float(result)

    def _get_n(self, n: float, h: float, alpha: float, power: float) -> float:
        if self.alternative == "two-sided":
            result = (
                norm.sf(norm.isf(alpha / 2) - h * sqrt(n / 2)) + norm.cdf(norm.ppf(alpha / 2) - h * sqrt(n / 2)) - power
            )
        elif self.alternative == "greater":
            result = norm.sf(norm.isf(alpha) - h * sqrt(n / 2)) - power
        else:
            result = norm.cdf(norm.ppf(alpha) - h * sqrt(n / 2)) - power
        return float(result)

    def _get_alpha(self, alpha: float, h: float, n: int, power: float) -> float:
        if self.alternative == "two-sided":
            result = (
                norm.sf(norm.isf(alpha / 2) - h * sqrt(n / 2)) + norm.cdf(norm.ppf(alpha / 2) - h * sqrt(n / 2)) - power
            )
        elif self.alternative == "greater":
            result = norm.sf(norm.isf(alpha) - h * sqrt(n / 2)) - power
        else:
            result = norm.cdf(norm.ppf(alpha) - h * sqrt(n / 2)) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.h is None or self.n is None or self.alpha is None:
                raise ValueError("h, n, and alpha must be provided to compute power")
            self.power = self._get_power(self.h, self.n, self.alpha)
        elif self.h is None:
            if self.n is None or self.alpha is None or self.power is None:
                raise ValueError("n, alpha, and power must be provided to solve for h")
            n, alpha, power = self.n, self.alpha, self.power
            if self.alternative == "two-sided":
                self.h = brentq(lambda h: self._get_effect_size(h, n, alpha, power), 1e-10, 10)
            elif self.alternative == "greater":
                self.h = brentq(lambda h: self._get_effect_size(h, n, alpha, power), -5, 10)
            else:
                self.h = brentq(lambda h: self._get_effect_size(h, n, alpha, power), -10, 5)
        elif self.n is None:
            if self.h is None or self.alpha is None or self.power is None:
                raise ValueError("h, alpha, and power must be provided to solve for n")
            h, alpha, power = self.h, self.alpha, self.power
            self.n = ceil(brentq(lambda n: self._get_n(n, h, alpha, power), 2 + 1e-10, 1e09))
        else:
            if self.h is None or self.n is None or self.power is None:
                raise ValueError("h, n, and power must be provided to solve for alpha")
            h, n, power = self.h, self.n, self.power
            self.alpha = brentq(lambda alpha: self._get_alpha(alpha, h, n, power), 1e-10, 1 - 1e-10)
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

    def _get_power(self, h: float, n1: int, n2: int, alpha: float) -> float:
        if self.alternative == "two-sided":
            power = norm.sf(norm.isf(alpha / 2) - h * sqrt(n1 * n2 / (n1 + n2))) + norm.cdf(
                norm.ppf(alpha / 2) - h * sqrt(n1 * n2 / (n1 + n2))
            )
        elif self.alternative == "greater":
            power = norm.sf(norm.isf(alpha) - h * sqrt(n1 * n2 / (n1 + n2)))
        else:
            power = norm.cdf(norm.ppf(alpha) - h * sqrt(n1 * n2 / (n1 + n2)))
        return float(power)

    def _get_effect_size(self, h: float, n1: int, n2: int, alpha: float, power: float) -> float:
        if self.alternative == "two-sided":
            result = (
                norm.sf(norm.isf(alpha / 2) - h * sqrt(n1 * n2 / (n1 + n2)))
                + norm.cdf(norm.ppf(alpha / 2) - h * sqrt(n1 * n2 / (n1 + n2)))
                - power
            )
        elif self.alternative == "greater":
            result = norm.sf(norm.isf(alpha) - h * sqrt(n1 * n2 / (n1 + n2))) - power
        else:
            result = norm.cdf(norm.ppf(alpha) - h * sqrt(n1 * n2 / (n1 + n2))) - power
        return float(result)

    def _get_n1(self, n1: float, h: float, n2: int, alpha: float, power: float) -> float:
        if self.alternative == "two-sided":
            result = (
                norm.sf(norm.isf(alpha / 2) - h * sqrt(n1 * n2 / (n1 + n2)))
                + norm.cdf(norm.ppf(alpha / 2) - h * sqrt(n1 * n2 / (n1 + n2)))
                - power
            )
        elif self.alternative == "greater":
            result = norm.sf(norm.isf(alpha) - h * sqrt(n1 * n2 / (n1 + n2))) - power
        else:
            result = norm.cdf(norm.ppf(alpha) - h * sqrt(n1 * n2 / (n1 + n2))) - power
        return float(result)

    def _get_n2(self, n2: float, h: float, n1: int, alpha: float, power: float) -> float:
        if self.alternative == "two-sided":
            result = (
                norm.sf(norm.isf(alpha / 2) - h * sqrt(n1 * n2 / (n1 + n2)))
                + norm.cdf(norm.ppf(alpha / 2) - h * sqrt(n1 * n2 / (n1 + n2)))
                - power
            )
        elif self.alternative == "greater":
            result = norm.sf(norm.isf(alpha) - h * sqrt(n1 * n2 / (n1 + n2))) - power
        else:
            result = norm.cdf(norm.ppf(alpha) - h * sqrt(n1 * n2 / (n1 + n2))) - power
        return float(result)

    def _get_alpha(self, alpha: float, h: float, n1: int, n2: int, power: float) -> float:
        if self.alternative == "two-sided":
            result = (
                norm.sf(norm.isf(alpha / 2) - h * sqrt(n1 * n2 / (n1 + n2)))
                + norm.cdf(norm.ppf(alpha / 2) - h * sqrt(n1 * n2 / (n1 + n2)))
                - power
            )
        elif self.alternative == "greater":
            result = norm.sf(norm.isf(alpha) - h * sqrt(n1 * n2 / (n1 + n2))) - power
        else:
            result = norm.cdf(norm.ppf(alpha) - h * sqrt(n1 * n2 / (n1 + n2))) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.h is None or self.n1 is None or self.n2 is None or self.alpha is None:
                raise ValueError("h, n1, n2, and alpha must be provided to compute power")
            self.power = self._get_power(self.h, self.n1, self.n2, self.alpha)
        elif self.h is None:
            if self.n1 is None or self.n2 is None or self.alpha is None or self.power is None:
                raise ValueError("n1, n2, alpha, and power must be provided to solve for h")
            n1, n2, alpha, power = self.n1, self.n2, self.alpha, self.power
            if self.alternative == "two-sided":
                self.h = brentq(lambda h: self._get_effect_size(h, n1, n2, alpha, power), 1e-10, 10)
            elif self.alternative == "greater":
                self.h = brentq(lambda h: self._get_effect_size(h, n1, n2, alpha, power), -5, 10)
            else:
                self.h = brentq(lambda h: self._get_effect_size(h, n1, n2, alpha, power), -10, 5)
        elif self.n1 is None:
            if self.h is None or self.n2 is None or self.alpha is None or self.power is None:
                raise ValueError("h, n2, alpha, and power must be provided to solve for n1")
            h, n2, alpha, power = self.h, self.n2, self.alpha, self.power
            self.n1 = ceil(brentq(lambda n1: self._get_n1(n1, h, n2, alpha, power), 2 + 1e-10, 1e09))
        elif self.n2 is None:
            if self.h is None or self.n1 is None or self.alpha is None or self.power is None:
                raise ValueError("h, n1, alpha, and power must be provided to solve for n2")
            h, n1, alpha, power = self.h, self.n1, self.alpha, self.power
            self.n2 = ceil(brentq(lambda n2: self._get_n2(n2, h, n1, alpha, power), 2 + 1e-10, 1e09))
        else:
            if self.h is None or self.n1 is None or self.n2 is None or self.power is None:
                raise ValueError("h, n1, n2, and power must be provided to solve for alpha")
            h, n1, n2, power = self.h, self.n1, self.n2, self.power
            self.alpha = brentq(lambda alpha: self._get_alpha(alpha, h, n1, n2, power), 1e-10, 1 - 1e-10)
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
