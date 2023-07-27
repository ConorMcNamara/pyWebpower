from typing import Optional, Dict
from math import sqrt, pow, ceil, log

from scipy.stats import norm
from scipy.optimize import brentq

from webpower.utils import nuniroot


class WpMediation:
    def __init__(
            self,
            n: Optional[int] = None,
            power: Optional[float] = None,
            a: Optional[float] = None,
            b: Optional[float] = None,
            var_x: float = 1,
            var_y: Optional[float] = None,
            var_m: float = 1,
            alpha: Optional[float] = None,
    ) -> None:
        self.power = power
        self.n = n
        self.a = a
        self.b = b
        self.var_x = var_x
        self.var_y = var_y
        self.var_m = var_m
        self.alpha = alpha
        self.method = "Power for simple mediation"
        self.url = "http://psychstat.org/mediation"

    def _get_power(self) -> float:
        numerator = sqrt(self.n) * self.a * self.b
        denominator = sqrt(
            pow(self.a, 2) * self.var_y / (self.var_m - pow(self.a, 2) * self.var_x)
            + pow(self.b, 2) * (self.var_m - pow(self.a, 2) * self.var_x) / self.var_x
        )
        delta = numerator / denominator
        alpha2 = self.alpha / 2
        za2 = norm.ppf(1 - alpha2)
        power = norm.sf(za2 - delta) + norm.cdf(-za2 - delta)
        return power

    def _get_n(self, n: int) -> float:
        numerator = sqrt(n) * self.a * self.b
        denominator = sqrt(
            pow(self.a, 2) * self.var_y / (self.var_m - pow(self.a, 2) * self.var_x)
            + pow(self.b, 2) * (self.var_m - pow(self.a, 2) * self.var_x) / self.var_x
        )
        delta = numerator / denominator
        alpha2 = self.alpha / 2
        za2 = norm.ppf(1 - alpha2)
        n = norm.sf(za2 - delta) + norm.cdf(-za2 - delta) - self.power
        return n

    def _get_var_y(self, var_y: float) -> float:
        numerator = sqrt(self.n) * self.a * self.b
        denominator = sqrt(
            pow(self.a, 2) * var_y / (self.var_m - pow(self.a, 2) * self.var_x)
            + pow(self.b, 2) * (self.var_m - pow(self.a, 2) * self.var_x) / self.var_x
        )
        delta = numerator / denominator
        alpha2 = self.alpha / 2
        za2 = norm.ppf(1 - alpha2)
        var_y = norm.sf(za2 - delta) + norm.cdf(-za2 - delta) - self.power
        return var_y

    def _get_a(self, a: float) -> float:
        numerator = sqrt(self.n) * a * self.b
        denominator = sqrt(
            pow(a, 2) * self.var_y / (self.var_m - pow(a, 2) * self.var_x)
            + pow(self.b, 2) * (self.var_m - pow(a, 2) * self.var_x) / self.var_x
        )
        delta = numerator / denominator
        alpha2 = self.alpha / 2
        za2 = norm.ppf(1 - alpha2)
        a = norm.sf(za2 - delta) + norm.cdf(-za2 - delta) - self.power
        return a

    def _get_b(self, b: float) -> float:
        numerator = sqrt(self.n) * self.a * b
        denominator = sqrt(
            pow(self.a, 2) * self.var_y / (self.var_m - pow(self.a, 2) * self.var_x)
            + pow(b, 2) * (self.var_m - pow(self.a, 2) * self.var_x) / self.var_x
        )
        delta = numerator / denominator
        alpha2 = self.alpha / 2
        za2 = norm.ppf(1 - alpha2)
        b = norm.sf(za2 - delta) + norm.cdf(-za2 - delta) - self.power
        return b

    def _get_alpha(self, alpha: float) -> float:
        numerator = sqrt(self.n) * self.a * self.b
        denominator = sqrt(
            pow(self.a, 2) * self.var_y / (self.var_m - pow(self.a, 2) * self.var_x)
            + pow(self.b, 2) * (self.var_m - pow(self.a, 2) * self.var_x) / self.var_x
        )
        delta = numerator / denominator
        alpha2 = alpha / 2
        za2 = norm.ppf(1 - alpha2)
        alpha = norm.sf(za2 - delta) + norm.cdf(-za2 - delta) - self.power
        return alpha

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.n is None:
            self.n = ceil(brentq(self._get_n, 2 + 1e-10, 1e09))
        elif self.var_y is None:
            self.var_y = nuniroot(self._get_var_y, 1e-10, 1e07)
        elif self.a is None:
            astart = self.var_m / self.var_x
            alow = -sqrt(astart) + 1e-06
            aup = sqrt(astart) - 1e-06
            self.a = nuniroot(self._get_a, alow, aup)
        elif self.b is None:
            self.b = nuniroot(self._get_b, -10, 10)
        else:
            self.alpha = nuniroot(self._get_alpha, 1e-10, 1 - 1e-10)
        return {
            "n": self.n,
            "a": self.a,
            "b": self.b,
            "var_x": self.var_x,
            "var_y": self.var_y,
            "var_m": self.var_m,
            "alpha": self.alpha,
            "power": self.power,
            "method": self.method,
            "url": self.url,
        }


class WpCorrelation:
    def __init__(
            self,
            n: Optional[int] = None,
            r: Optional[float] = None,
            power: Optional[float] = None,
            p: int = 0,
            rho0: float = 0.0,
            alpha: Optional[float] = None,
            alternative: str = "two-sided",
    ) -> None:
        self.n = n
        self.r = r
        self.power = power
        self.p = p
        self.rho0 = rho0
        self.alpha = alpha
        self.alternative = alternative.casefold()
        self.method = "Power for correlation"
        self.url = "http://psychstat.org/correlation"

    def _get_power(self) -> float:
        delta = sqrt(self.n - 3 - self.p) * (
                log((1 + self.r) / (1 - self.r)) / 2
                + self.r
                / (self.n - 1 - self.p)
                / 2
                * (
                        1
                        + (5 + pow(self.r, 2)) / (self.n - 1 - self.p) / 4
                        + (
                                11
                                + 2 * pow(self.r, 2)
                                + 3 * pow(self.r, 4)) / pow(self.n - 1 - self.p, 2)
                        / 8
                )
                - log((1 + self.rho0) / (1 - self.rho0)) / 2
                - self.rho0 / (self.n - 1 - self.p) / 2
        )
        v = (
                (self.n - 3 - self.p)
                / (self.n - 1 - self.p)
                * (
                        1
                        + (4 - pow(self.r, 2)) / (self.n - 1 - self.p) / 2
                        + (22 - 6 * pow(self.r, 2) - 3 * pow(self.r, 4))
                        / pow(self.n - 1 - self.p, 2)
                        / 6
                )
        )
        if self.alternative == "two-sided":
            z_alpha = norm.ppf(1 - self.alpha / 2)
            power = norm.cdf((delta - z_alpha) / sqrt(v)) + norm.cdf(
                (-delta - z_alpha) / sqrt(v)
            )
        else:
            z_alpha = norm.ppf(1 - self.alpha)
            if self.alternative == "greater":
                power = norm.cdf((delta - z_alpha) / sqrt(v))
            else:
                power = norm.cdf((-delta - z_alpha) / sqrt(v))
        return power

    def _get_n(self, n: int) -> float:
        delta = sqrt(n - 3 - self.p) * (
                log((1 + self.r) / (1 - self.r)) / 2
                + self.r
                / (n - 1 - self.p)
                / 2
                * (
                        1
                        + (5 + pow(self.r, 2)) / (n - 1 - self.p) / 4
                        + (
                                11
                                + 2 * pow(self.r, 2)
                                + 3 * pow(self.r, 4)) / pow(n - 1 - self.p, 2)
                        / 8
                )
                - log((1 + self.rho0) / (1 - self.rho0)) / 2
                - self.rho0 / (n - 1 - self.p) / 2
        )
        v = (
                (n - 3 - self.p)
                / (n - 1 - self.p)
                * (
                        1
                        + (4 - pow(self.r, 2)) / (n - 1 - self.p) / 2
                        + (22 - 6 * pow(self.r, 2) - 3 * pow(self.r, 4))
                        / pow(n - 1 - self.p, 2)
                        / 6
                )
        )
        if self.alternative == "two-sided":
            z_alpha = norm.ppf(1 - self.alpha / 2)
            n = norm.cdf((delta - z_alpha) / sqrt(v)) + norm.cdf(
                (-delta - z_alpha) / sqrt(v)
            ) - self.power
        else:
            z_alpha = norm.ppf(1 - self.alpha)
            if self.alternative == "greater":
                n = norm.cdf((delta - z_alpha) / sqrt(v)) - self.power
            else:
                n = norm.cdf((-delta - z_alpha) / sqrt(v)) - self.power
        return n

    def _get_effect_size(self, effect_size: float) -> float:
        delta = sqrt(self.n - 3 - self.p) * (
                log((1 + effect_size) / (1 - effect_size)) / 2
                + effect_size
                / (self.n - 1 - self.p)
                / 2
                * (
                        1
                        + (5 + pow(effect_size, 2)) / (self.n - 1 - self.p) / 4
                        + (
                                11
                                + 2 * pow(effect_size, 2)
                                + 3 * pow(effect_size, 4)) / pow(self.n - 1 - self.p, 2)
                        / 8
                )
                - log((1 + self.rho0) / (1 - self.rho0)) / 2
                - self.rho0 / (self.n - 1 - self.p) / 2
        )
        v = (
                (self.n - 3 - self.p)
                / (self.n - 1 - self.p)
                * (
                        1
                        + (4 - pow(effect_size, 2)) / (self.n - 1 - self.p) / 2
                        + (22 - 6 * pow(effect_size, 2) - 3 * pow(effect_size, 4))
                        / pow(self.n - 1 - self.p, 2)
                        / 6
                )
        )
        if self.alternative == "two-sided":
            z_alpha = norm.ppf(1 - self.alpha / 2)
            effect_size = norm.cdf((delta - z_alpha) / sqrt(v)) + norm.cdf(
                (-delta - z_alpha) / sqrt(v)
            ) - self.power
        else:
            z_alpha = norm.ppf(1 - self.alpha)
            if self.alternative == "greater":
                effect_size = norm.cdf((delta - z_alpha) / sqrt(v)) - self.power
            else:
                effect_size = norm.cdf((-delta - z_alpha) / sqrt(v)) - self.power
        return effect_size

    def _get_alpha(self, alpha: float) -> float:
        delta = sqrt(self.n - 3 - self.p) * (
                log((1 + self.r) / (1 - self.r)) / 2
                + self.r
                / (self.n - 1 - self.p)
                / 2
                * (
                        1
                        + (5 + pow(self.r, 2)) / (self.n - 1 - self.p) / 4
                        + (
                                11
                                + 2 * pow(self.r, 2)
                                + 3 * pow(self.r, 4)) / pow(self.n - 1 - self.p, 2)
                        / 8
                )
                - log((1 + self.rho0) / (1 - self.rho0)) / 2
                - self.rho0 / (self.n - 1 - self.p) / 2
        )
        v = (
                (self.n - 3 - self.p)
                / (self.n - 1 - self.p)
                * (
                        1
                        + (4 - pow(self.r, 2)) / (self.n - 1 - self.p) / 2
                        + (22 - 6 * pow(self.r, 2) - 3 * pow(self.r, 4))
                        / pow(self.n - 1 - self.p, 2)
                        / 6
                )
        )
        if self.alternative == "two-sided":
            z_alpha = norm.ppf(1 - alpha / 2)
            alpha = norm.cdf((delta - z_alpha) / sqrt(v)) + norm.cdf(
                (-delta - z_alpha) / sqrt(v)) - self.power
        else:
            z_alpha = norm.ppf(1 - alpha)
            if self.alternative == "greater":
                alpha = norm.cdf((delta - z_alpha) / sqrt(v)) - self.power
            else:
                alpha = norm.cdf((-delta - z_alpha) / sqrt(v)) - self.power
        return alpha

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.n is None:
            self.n = ceil(brentq(self._get_n, 4 + self.p + 1e-10, 1e07))
        elif self.r is None:
            if self.alternative == "two-sided":
                self.r = brentq(self._get_effect_size, 1e-10, 1 - 1e-10)
            else:
                self.r = brentq(self._get_effect_size, -1 + 1e-10, 1 - 1e-10)
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
        return {
            "n": self.n,
            "effect_size": self.r,
            "alpha": self.alpha,
            "power": self.power,
            "alternative": self.alternative,
            "method": self.method,
            "url": self.url,
        }
