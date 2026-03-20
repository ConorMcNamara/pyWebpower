from math import ceil, log, sqrt

from scipy.optimize import brentq
from scipy.stats import norm

from webpower.utils import nuniroot


class WpMediation:
    def __init__(
        self,
        n: int | None = None,
        power: float | None = None,
        a: float | None = None,
        b: float | None = None,
        var_x: float = 1,
        var_y: float | None = None,
        var_m: float = 1,
        alpha: float | None = None,
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

    def _get_power(self, n: int, a: float, b: float, var_y: float, alpha: float) -> float:
        numerator = sqrt(n) * a * b
        denominator = sqrt(
            a**2 * var_y / (self.var_m - a**2 * self.var_x) + b**2 * (self.var_m - a**2 * self.var_x) / self.var_x
        )
        delta = numerator / denominator
        alpha2 = alpha / 2
        za2 = norm.ppf(1 - alpha2)
        power = norm.sf(za2 - delta) + norm.cdf(-za2 - delta)
        return float(power)

    def _get_n(self, n: float, a: float, b: float, var_y: float, alpha: float, power: float) -> float:
        numerator = sqrt(n) * a * b
        denominator = sqrt(
            a**2 * var_y / (self.var_m - a**2 * self.var_x) + b**2 * (self.var_m - a**2 * self.var_x) / self.var_x
        )
        delta = numerator / denominator
        alpha2 = alpha / 2
        za2 = norm.ppf(1 - alpha2)
        result = norm.sf(za2 - delta) + norm.cdf(-za2 - delta) - power
        return float(result)

    def _get_var_y(self, var_y: float, n: int, a: float, b: float, alpha: float, power: float) -> float:
        numerator = sqrt(n) * a * b
        denominator = sqrt(
            a**2 * var_y / (self.var_m - a**2 * self.var_x) + b**2 * (self.var_m - a**2 * self.var_x) / self.var_x
        )
        delta = numerator / denominator
        alpha2 = alpha / 2
        za2 = norm.ppf(1 - alpha2)
        result = norm.sf(za2 - delta) + norm.cdf(-za2 - delta) - power
        return float(result)

    def _get_a(self, a: float, n: int, b: float, var_y: float, alpha: float, power: float) -> float:
        numerator = sqrt(n) * a * b
        denominator = sqrt(
            a**2 * var_y / (self.var_m - a**2 * self.var_x) + b**2 * (self.var_m - a**2 * self.var_x) / self.var_x
        )
        delta = numerator / denominator
        alpha2 = alpha / 2
        za2 = norm.ppf(1 - alpha2)
        result = norm.sf(za2 - delta) + norm.cdf(-za2 - delta) - power
        return float(result)

    def _get_b(self, b: float, n: int, a: float, var_y: float, alpha: float, power: float) -> float:
        numerator = sqrt(n) * a * b
        denominator = sqrt(
            a**2 * var_y / (self.var_m - a**2 * self.var_x) + b**2 * (self.var_m - a**2 * self.var_x) / self.var_x
        )
        delta = numerator / denominator
        alpha2 = alpha / 2
        za2 = norm.ppf(1 - alpha2)
        result = norm.sf(za2 - delta) + norm.cdf(-za2 - delta) - power
        return float(result)

    def _get_alpha(self, alpha: float, n: int, a: float, b: float, var_y: float, power: float) -> float:
        numerator = sqrt(n) * a * b
        denominator = sqrt(
            a**2 * var_y / (self.var_m - a**2 * self.var_x) + b**2 * (self.var_m - a**2 * self.var_x) / self.var_x
        )
        delta = numerator / denominator
        alpha2 = alpha / 2
        za2 = norm.ppf(1 - alpha2)
        result = norm.sf(za2 - delta) + norm.cdf(-za2 - delta) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.n is None or self.a is None or self.b is None or self.var_y is None or self.alpha is None:
                raise ValueError("n, a, b, var_y, and alpha must be provided to compute power")
            self.power = self._get_power(self.n, self.a, self.b, self.var_y, self.alpha)
        elif self.n is None:
            if self.a is None or self.b is None or self.var_y is None or self.alpha is None or self.power is None:
                raise ValueError("a, b, var_y, alpha, and power must be provided to solve for n")
            a, b, var_y, alpha, power = self.a, self.b, self.var_y, self.alpha, self.power
            self.n = ceil(float(brentq(lambda n: self._get_n(n, a, b, var_y, alpha, power), 2 + 1e-10, 1e09)))
        elif self.var_y is None:
            if self.n is None or self.a is None or self.b is None or self.alpha is None or self.power is None:
                raise ValueError("n, a, b, alpha, and power must be provided to solve for var_y")
            n, a, b, alpha, power = self.n, self.a, self.b, self.alpha, self.power
            self.var_y = nuniroot(lambda var_y: self._get_var_y(var_y, n, a, b, alpha, power), 1e-10, 1e07)
        elif self.a is None:
            if self.n is None or self.b is None or self.var_y is None or self.alpha is None or self.power is None:
                raise ValueError("n, b, var_y, alpha, and power must be provided to solve for a")
            n, b, var_y, alpha, power = self.n, self.b, self.var_y, self.alpha, self.power
            astart = self.var_m / self.var_x
            alow = -sqrt(astart) + 1e-06
            aup = sqrt(astart) - 1e-06
            self.a = nuniroot(lambda a: self._get_a(a, n, b, var_y, alpha, power), alow, aup)
        elif self.b is None:
            if self.n is None or self.a is None or self.var_y is None or self.alpha is None or self.power is None:
                raise ValueError("n, a, var_y, alpha, and power must be provided to solve for b")
            n, a, var_y, alpha, power = self.n, self.a, self.var_y, self.alpha, self.power
            self.b = nuniroot(lambda b: self._get_b(b, n, a, var_y, alpha, power), -10, 10)
        else:
            if self.n is None or self.a is None or self.b is None or self.var_y is None or self.power is None:
                raise ValueError("n, a, b, var_y, and power must be provided to solve for alpha")
            n, a, b, var_y, power = self.n, self.a, self.b, self.var_y, self.power
            self.alpha = nuniroot(lambda alpha: self._get_alpha(alpha, n, a, b, var_y, power), 1e-10, 1 - 1e-10)
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
        n: int | None = None,
        r: float | None = None,
        power: float | None = None,
        p: int = 0,
        rho0: float = 0.0,
        alpha: float | None = None,
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

    def _get_power(self, n: int, r: float, alpha: float) -> float:
        delta = sqrt(n - 3 - self.p) * (
            log((1 + r) / (1 - r)) / 2
            + r
            / (n - 1 - self.p)
            / 2
            * (1 + (5 + r**2) / (n - 1 - self.p) / 4 + (11 + 2 * r**2 + 3 * r**4) / (n - 1 - self.p) ** 2 / 8)
            - log((1 + self.rho0) / (1 - self.rho0)) / 2
            - self.rho0 / (n - 1 - self.p) / 2
        )
        v = (
            (n - 3 - self.p)
            / (n - 1 - self.p)
            * (1 + (4 - r**2) / (n - 1 - self.p) / 2 + (22 - 6 * r**2 - 3 * r**4) / (n - 1 - self.p) ** 2 / 6)
        )
        if self.alternative == "two-sided":
            z_alpha = norm.ppf(1 - alpha / 2)
            power = norm.cdf((delta - z_alpha) / sqrt(v)) + norm.cdf((-delta - z_alpha) / sqrt(v))
        else:
            z_alpha = norm.ppf(1 - alpha)
            if self.alternative == "greater":
                power = norm.cdf((delta - z_alpha) / sqrt(v))
            else:
                power = norm.cdf((-delta - z_alpha) / sqrt(v))
        return float(power)

    def _get_n(self, n: float, r: float, alpha: float, power: float) -> float:
        delta = sqrt(n - 3 - self.p) * (
            log((1 + r) / (1 - r)) / 2
            + r
            / (n - 1 - self.p)
            / 2
            * (1 + (5 + r**2) / (n - 1 - self.p) / 4 + (11 + 2 * r**2 + 3 * r**4) / (n - 1 - self.p) ** 2 / 8)
            - log((1 + self.rho0) / (1 - self.rho0)) / 2
            - self.rho0 / (n - 1 - self.p) / 2
        )
        v = (
            (n - 3 - self.p)
            / (n - 1 - self.p)
            * (1 + (4 - r**2) / (n - 1 - self.p) / 2 + (22 - 6 * r**2 - 3 * r**4) / (n - 1 - self.p) ** 2 / 6)
        )
        if self.alternative == "two-sided":
            z_alpha = norm.ppf(1 - alpha / 2)
            return float(norm.cdf((delta - z_alpha) / sqrt(v)) + norm.cdf((-delta - z_alpha) / sqrt(v)) - power)
        z_alpha = norm.ppf(1 - alpha)
        if self.alternative == "greater":
            return float(norm.cdf((delta - z_alpha) / sqrt(v)) - power)
        return float(norm.cdf((-delta - z_alpha) / sqrt(v)) - power)

    def _get_effect_size(self, effect_size: float, n: int, alpha: float, power: float) -> float:
        delta = sqrt(n - 3 - self.p) * (
            log((1 + effect_size) / (1 - effect_size)) / 2
            + effect_size
            / (n - 1 - self.p)
            / 2
            * (
                1
                + (5 + effect_size**2) / (n - 1 - self.p) / 4
                + (11 + 2 * effect_size**2 + 3 * effect_size**4) / (n - 1 - self.p) ** 2 / 8
            )
            - log((1 + self.rho0) / (1 - self.rho0)) / 2
            - self.rho0 / (n - 1 - self.p) / 2
        )
        v = (
            (n - 3 - self.p)
            / (n - 1 - self.p)
            * (
                1
                + (4 - effect_size**2) / (n - 1 - self.p) / 2
                + (22 - 6 * effect_size**2 - 3 * effect_size**4) / (n - 1 - self.p) ** 2 / 6
            )
        )
        if self.alternative == "two-sided":
            z_alpha = norm.ppf(1 - alpha / 2)
            return float(norm.cdf((delta - z_alpha) / sqrt(v)) + norm.cdf((-delta - z_alpha) / sqrt(v)) - power)
        z_alpha = norm.ppf(1 - alpha)
        if self.alternative == "greater":
            return float(norm.cdf((delta - z_alpha) / sqrt(v)) - power)
        return float(norm.cdf((-delta - z_alpha) / sqrt(v)) - power)

    def _get_alpha(self, alpha: float, n: int, r: float, power: float) -> float:
        delta = sqrt(n - 3 - self.p) * (
            log((1 + r) / (1 - r)) / 2
            + r
            / (n - 1 - self.p)
            / 2
            * (1 + (5 + r**2) / (n - 1 - self.p) / 4 + (11 + 2 * r**2 + 3 * r**4) / (n - 1 - self.p) ** 2 / 8)
            - log((1 + self.rho0) / (1 - self.rho0)) / 2
            - self.rho0 / (n - 1 - self.p) / 2
        )
        v = (
            (n - 3 - self.p)
            / (n - 1 - self.p)
            * (1 + (4 - r**2) / (n - 1 - self.p) / 2 + (22 - 6 * r**2 - 3 * r**4) / (n - 1 - self.p) ** 2 / 6)
        )
        if self.alternative == "two-sided":
            z_alpha = norm.ppf(1 - alpha / 2)
            return float(norm.cdf((delta - z_alpha) / sqrt(v)) + norm.cdf((-delta - z_alpha) / sqrt(v)) - power)
        z_alpha = norm.ppf(1 - alpha)
        if self.alternative == "greater":
            return float(norm.cdf((delta - z_alpha) / sqrt(v)) - power)
        return float(norm.cdf((-delta - z_alpha) / sqrt(v)) - power)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.n is None or self.r is None or self.alpha is None:
                raise ValueError("n, r, and alpha must be provided to compute power")
            self.power = self._get_power(self.n, self.r, self.alpha)
        elif self.n is None:
            if self.r is None or self.alpha is None or self.power is None:
                raise ValueError("r, alpha, and power must be provided to solve for n")
            r, alpha, power = self.r, self.alpha, self.power
            self.n = ceil(float(brentq(lambda n: self._get_n(n, r, alpha, power), 4 + self.p + 1e-10, 1e07)))
        elif self.r is None:
            if self.n is None or self.alpha is None or self.power is None:
                raise ValueError("n, alpha, and power must be provided to solve for r")
            n, alpha, power = self.n, self.alpha, self.power
            if self.alternative == "two-sided":
                self.r = float(brentq(lambda r: self._get_effect_size(r, n, alpha, power), 1e-10, 1 - 1e-10))
            else:
                self.r = float(brentq(lambda r: self._get_effect_size(r, n, alpha, power), -1 + 1e-10, 1 - 1e-10))
        else:
            if self.n is None or self.r is None or self.power is None:
                raise ValueError("n, r, and power must be provided to solve for alpha")
            n, r, power = self.n, self.r, self.power
            self.alpha = float(brentq(lambda alpha: self._get_alpha(alpha, n, r, power), 1e-10, 1 - 1e-10))
        return {
            "n": self.n,
            "effect_size": self.r,
            "alpha": self.alpha,
            "power": self.power,
            "alternative": self.alternative,
            "method": self.method,
            "url": self.url,
        }
