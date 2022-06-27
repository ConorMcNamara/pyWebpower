from typing import Optional, Dict
from math import sqrt, pow, ceil

from scipy.stats import norm
from scipy.optimize import brentq


class WpMediation:
    def __init__(
            self,
            n: Optional[int] = None,
            power: Optional[float] = None,
            a: Optional[float] = None,
            b: Optional[float] = None,
            var_x: Optional[float] = None,
            var_y: Optional[float] = None,
            var_m: Optional[float] = None,
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
        denominator = sqrt(pow(self.a, 2) * self.var_y / (self.var_m - pow(self.a, 2) * self.var_x) + pow(self.b, 2) *
                           (self.var_m - pow(self.a, 2) * self.var_x) / self.var_x)
        delta = numerator / denominator
        alpha2 = self.alpha / 2
        za2 = norm.ppf(1 - alpha2)
        power = 1 - norm.cdf(za2 - delta) + norm.cdf(-za2 - delta)
        return power

    def _get_n(self, n: int):
        numerator = sqrt(n) * self.a * self.b
        denominator = sqrt(pow(self.a, 2) * self.var_y / (self.var_m - pow(self.a, 2) * self.var_x) + pow(self.b, 2) *
                           (self.var_m - pow(self.a, 2) * self.var_x) / self.var_x)
        delta = numerator / denominator
        alpha2 = self.alpha / 2
        za2 = norm.ppf(1 - alpha2)
        n = 1 - norm.cdf(za2 - delta) + norm.cdf(-za2 - delta) - self.power
        return n

    def _get_var_x(self, var_x: float):
        numerator = sqrt(self.n) * self.a * self.b
        denominator = sqrt(pow(self.a, 2) * self.var_y / (self.var_m - pow(self.a, 2) * var_x) + pow(self.b, 2) *
                           (self.var_m - pow(self.a, 2) * var_x) / var_x)
        delta = numerator / denominator
        alpha2 = self.alpha / 2
        za2 = norm.ppf(1 - alpha2)
        var_x = 1 - norm.cdf(za2 - delta) + norm.cdf(-za2 - delta) - self.power
        return var_x

    def _get_var_y(self, var_y: float):
        numerator = sqrt(self.n) * self.a * self.b
        denominator = sqrt(pow(self.a, 2) * var_y / (self.var_m - pow(self.a, 2) * self.var_x) + pow(self.b, 2) *
                           (self.var_m - pow(self.a, 2) * self.var_x) / self.var_x)
        delta = numerator / denominator
        alpha2 = self.alpha / 2
        za2 = norm.ppf(1 - alpha2)
        var_y = 1 - norm.cdf(za2 - delta) + norm.cdf(-za2 - delta) - self.power
        return var_y

    def _get_var_m(self, var_m: float):
        numerator = sqrt(self.n) * self.a * self.b
        denominator = sqrt(pow(self.a, 2) * self.var_y / (var_m - pow(self.a, 2) * self.var_x) + pow(self.b, 2) *
                           (var_m - pow(self.a, 2) * self.var_x) / self.var_x)
        delta = numerator / denominator
        alpha2 = self.alpha / 2
        za2 = norm.ppf(1 - alpha2)
        var_m = 1 - norm.cdf(za2 - delta) + norm.cdf(-za2 - delta) - self.power
        return var_m

    def _get_a(self, a: float):
        numerator = sqrt(self.n) * a * self.b
        denominator = sqrt(pow(a, 2) * self.var_y / (self.var_m - pow(a, 2) * self.var_x) + pow(self.b, 2) *
                           (self.var_m - pow(a, 2) * self.var_x) / self.var_x)
        delta = numerator / denominator
        alpha2 = self.alpha / 2
        za2 = norm.ppf(1 - alpha2)
        a = 1 - norm.cdf(za2 - delta) + norm.cdf(-za2 - delta) - self.power
        return a

    def _get_b(self, b: float):
        numerator = sqrt(self.n) * self.a * b
        denominator = sqrt(pow(self.a, 2) * self.var_y / (self.var_m - pow(self.a, 2) * self.var_x) + pow(b, 2) *
                           (self.var_m - pow(self.a, 2) * self.var_x) / self.var_x)
        delta = numerator / denominator
        alpha2 = self.alpha / 2
        za2 = norm.ppf(1 - alpha2)
        b = 1 - norm.cdf(za2 - delta) + norm.cdf(-za2 - delta) - self.power
        return b

    def _get_alpha(self, alpha: float) -> float:
        numerator = sqrt(self.n) * self.a * self.b
        denominator = sqrt(pow(self.a, 2) * self.var_y / (self.var_m - pow(self.a, 2) * self.var_x) + pow(self.b, 2) *
                           (self.var_m - pow(self.a, 2) * self.var_x) / self.var_x)
        delta = numerator / denominator
        alpha2 = alpha / 2
        za2 = norm.ppf(1 - alpha2)
        alpha = 1 - norm.cdf(za2 - delta) + norm.cdf(-za2 - delta) - self.power
        return alpha

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.n is None:
            self.n = ceil(brentq(self._get_n, 2 + 1e-10, 1e09))
        elif self.var_x is None:
            self.var_x = brentq(self._get_var_x, 1e-10, 1e07)
        elif self.var_y is None:
            self.var_y = brentq(self._get_var_y, 1e-10, 1e07)
        elif self.var_m is None:
            self.var_m = brentq(self._get_var_m, 1e-10, 1e07)
        elif self.a is None:
            astart = self.var_m / self.var_x
            alow = -sqrt(astart) + 1e-06
            aup = sqrt(astart) - 1e-06
            self.a = brentq(self._get_a, alow, aup)
        elif self.b is None:
            self.b = brentq(self._get_b, -10, 10)
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
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

