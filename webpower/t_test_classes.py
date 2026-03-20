from math import ceil, sqrt

from scipy.optimize import brentq
from scipy.stats import nct
from scipy.stats import t as t_dist


class WpOneT:
    def __init__(
        self,
        n: int | None = None,
        d: float | None = None,
        alpha: float | None = None,
        power: float | None = None,
        test_type: str = "two-sample",
        alternative: str = "two-sided",
    ) -> None:
        self.n = n
        self.d = d
        self.alpha = alpha
        self.power = power
        self.test_type = test_type.casefold()
        self.alternative = alternative.casefold()
        if self.test_type == "one-sample":
            self.method = "One Sample"
            self.note = None
            self.t_sample = 1
        elif self.test_type == "paired":
            self.method = "Paired Sample"
            self.note = "n is number of *pairs*"
            self.t_sample = 1
        else:
            self.method = "Two Sample"
            self.note = "n is the number in *each* group"
            self.t_sample = 2
        self.url = "http://psychstat.org/ttest"

    def _get_power(self, n: int, d: float, alpha: float) -> float:
        nu = (n - 1) * self.t_sample
        if self.alternative == "two-sided":
            qu = t_dist.isf(alpha / 2, nu)
            power = nct.sf(qu, nu, sqrt(n / self.t_sample) * d) + nct.cdf(-qu, nu, sqrt(n / self.t_sample) * d)
        elif self.alternative == "greater":
            power = nct.sf(
                t_dist.isf(alpha, nu),
                nu,
                sqrt(n / self.t_sample) * d,
            )
        else:
            power = nct.cdf(
                t_dist.ppf(alpha, nu),
                nu,
                sqrt(n / self.t_sample) * d,
            )
        return float(power)

    def _get_effect_size(self, effect_size: float, n: int, alpha: float, power: float) -> float:
        nu = (n - 1) * self.t_sample
        if self.alternative == "two-sided":
            qu = t_dist.isf(alpha / 2, nu)
            result = (
                nct.sf(qu, nu, sqrt(n / self.t_sample) * effect_size)
                + nct.cdf(-qu, nu, sqrt(n / self.t_sample) * effect_size)
                - power
            )
        elif self.alternative == "greater":
            result = (
                nct.sf(
                    t_dist.isf(alpha, nu),
                    nu,
                    sqrt(n / self.t_sample) * effect_size,
                )
                - power
            )
        else:
            result = (
                nct.cdf(
                    t_dist.ppf(alpha, nu),
                    nu,
                    sqrt(n / self.t_sample) * effect_size,
                )
                - power
            )
        return float(result)

    def _get_n(self, n: float, d: float, alpha: float, power: float) -> float:
        nu = (n - 1) * self.t_sample
        if self.alternative == "two-sided":
            qu = t_dist.isf(alpha / 2, nu)
            result = nct.sf(qu, nu, sqrt(n / self.t_sample) * d) + nct.cdf(-qu, nu, sqrt(n / self.t_sample) * d) - power
        elif self.alternative == "greater":
            result = nct.sf(t_dist.isf(alpha, nu), nu, sqrt(n / self.t_sample) * d) - power
        else:
            result = nct.cdf(t_dist.ppf(alpha, nu), nu, sqrt(n / self.t_sample) * d) - power
        return float(result)

    def _get_alpha(self, alpha: float, n: int, d: float, power: float) -> float:
        nu = (n - 1) * self.t_sample
        if self.alternative == "two-sided":
            qu = t_dist.isf(alpha / 2, nu)
            result = nct.sf(qu, nu, sqrt(n / self.t_sample) * d) + nct.cdf(-qu, nu, sqrt(n / self.t_sample) * d) - power
        elif self.alternative == "greater":
            result = nct.sf(t_dist.isf(alpha, nu), nu, sqrt(n / self.t_sample) * d) - power
        else:
            result = nct.cdf(t_dist.ppf(alpha, nu), nu, sqrt(n / self.t_sample) * d) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.n is None or self.d is None or self.alpha is None:
                raise ValueError("n, d, and alpha must be provided to compute power")
            self.power = self._get_power(self.n, self.d, self.alpha)
        elif self.d is None:
            if self.n is None or self.alpha is None or self.power is None:
                raise ValueError("n, alpha, and power must be provided to solve for d")
            n, alpha, power = self.n, self.alpha, self.power
            if self.alternative == "two-sided":
                self.d = float(brentq(lambda d: self._get_effect_size(d, n, alpha, power), 1e-07, 10))
            elif self.alternative == "greater":
                self.d = float(brentq(lambda d: self._get_effect_size(d, n, alpha, power), -5, 10))
            else:
                self.d = float(brentq(lambda d: self._get_effect_size(d, n, alpha, power), -10, 5))
        elif self.n is None:
            if self.d is None or self.alpha is None or self.power is None:
                raise ValueError("d, alpha, and power must be provided to solve for n")
            d, alpha, power = self.d, self.alpha, self.power
            self.n = ceil(float(brentq(lambda n: self._get_n(n, d, alpha, power), 2 + 1e-10, 1e09)))
        else:
            if self.n is None or self.d is None or self.power is None:
                raise ValueError("n, d, and power must be provided to solve for alpha")
            n, d, power = self.n, self.d, self.power
            self.alpha = float(brentq(lambda alpha: self._get_alpha(alpha, n, d, power), 1e-10, 1 - 1e-10))
        if self.note is not None:
            return {
                "n": self.n,
                "effect_size": self.d,
                "alpha": self.alpha,
                "power": self.power,
                "alternative": self.alternative,
                "method": f"{self.method} t test power calculation",
                "note": self.note,
                "url": self.url,
            }
        else:
            return {
                "n": self.n,
                "effect_size": self.d,
                "alpha": self.alpha,
                "power": self.power,
                "alternative": self.alternative,
                "method": f"{self.method} t test power calculation",
                "url": self.url,
            }


class WpTwoT:
    def __init__(
        self,
        n1: int | None = None,
        n2: int | None = None,
        d: float | None = None,
        alpha: float | None = None,
        power: float | None = None,
        alternative: str = "two-sided",
    ) -> None:
        self.n1 = n1
        self.n2 = n2
        self.d = d
        self.alpha = alpha
        self.power = power
        self.alternative = alternative.casefold()
        self.note = "NOTE: n1 and n2 are number in *each* group"
        self.method = "Unbalanced two-sample t-test"
        self.url = "http://psychstat.org/ttest2n"

    def _get_power(self, n1: int, n2: int, d: float, alpha: float) -> float:
        nu = n1 + n2 - 2
        if self.alternative == "two-sided":
            qu = t_dist.isf(alpha / 2, nu)
            power = nct.sf(qu, nu, d * (1 / sqrt(1 / n1 + 1 / n2))) + nct.cdf(-qu, nu, d * (1 / sqrt(1 / n1 + 1 / n2)))
        elif self.alternative == "greater":
            power = nct.sf(
                t_dist.isf(alpha, nu),
                nu,
                d * (1 / sqrt(1 / n1 + 1 / n2)),
            )
        else:
            power = nct.cdf(
                t_dist.ppf(alpha, nu),
                nu,
                d * (1 / sqrt(1 / n1 + 1 / n2)),
            )
        return float(power)

    def _get_effect_size(self, effect_size: float, n1: int, n2: int, alpha: float, power: float) -> float:
        nu = n1 + n2 - 2
        if self.alternative == "two-sided":
            qu = t_dist.isf(alpha / 2, nu)
            result = (
                nct.sf(qu, nu, effect_size * (1 / sqrt(1 / n1 + 1 / n2)))
                + nct.cdf(-qu, nu, effect_size * (1 / sqrt(1 / n1 + 1 / n2)))
                - power
            )
        elif self.alternative == "greater":
            result = (
                nct.sf(
                    t_dist.isf(alpha, nu),
                    nu,
                    effect_size * (1 / sqrt(1 / n1 + 1 / n2)),
                )
                - power
            )
        else:
            result = (
                nct.cdf(
                    t_dist.ppf(alpha, nu),
                    nu,
                    effect_size * (1 / sqrt(1 / n1 + 1 / n2)),
                )
                - power
            )
        return float(result)

    def _get_n1(self, n1: float, n2: int, d: float, alpha: float, power: float) -> float:
        nu = n1 + n2 - 2
        if self.alternative == "two-sided":
            qu = t_dist.isf(alpha / 2, nu)
            result = (
                nct.sf(qu, nu, d * (1 / sqrt(1 / n1 + 1 / n2)))
                + nct.cdf(-qu, nu, d * (1 / sqrt(1 / n1 + 1 / n2)))
                - power
            )
        elif self.alternative == "greater":
            result = (
                nct.sf(
                    t_dist.isf(alpha, nu),
                    nu,
                    d * (1 / sqrt(1 / n1 + 1 / n2)),
                )
                - power
            )
        else:
            result = (
                nct.cdf(
                    t_dist.ppf(alpha, nu),
                    nu,
                    d * (1 / sqrt(1 / n1 + 1 / n2)),
                )
                - power
            )
        return float(result)

    def _get_n2(self, n2: float, n1: int, d: float, alpha: float, power: float) -> float:
        nu = n1 + n2 - 2
        if self.alternative == "two-sided":
            qu = t_dist.isf(alpha / 2, nu)
            result = (
                nct.sf(qu, nu, d * (1 / sqrt(1 / n1 + 1 / n2)))
                + nct.cdf(-qu, nu, d * (1 / sqrt(1 / n1 + 1 / n2)))
                - power
            )
        elif self.alternative == "greater":
            result = (
                nct.sf(
                    t_dist.isf(alpha, nu),
                    nu,
                    d * (1 / sqrt(1 / n1 + 1 / n2)),
                )
                - power
            )
        else:
            result = (
                nct.cdf(
                    t_dist.ppf(alpha, nu),
                    nu,
                    d * (1 / sqrt(1 / n1 + 1 / n2)),
                )
                - power
            )
        return float(result)

    def _get_alpha(self, alpha: float, n1: int, n2: int, d: float, power: float) -> float:
        nu = n1 + n2 - 2
        if self.alternative == "two-sided":
            qu = t_dist.isf(alpha / 2, nu)
            result = (
                nct.sf(qu, nu, d * (1 / sqrt(1 / n1 + 1 / n2)))
                + nct.cdf(-qu, nu, d * (1 / sqrt(1 / n1 + 1 / n2)))
                - power
            )
        elif self.alternative == "greater":
            result = (
                nct.sf(
                    t_dist.isf(alpha, nu),
                    nu,
                    d * (1 / sqrt(1 / n1 + 1 / n2)),
                )
                - power
            )
        else:
            result = (
                nct.cdf(
                    t_dist.ppf(alpha, nu),
                    nu,
                    d * (1 / sqrt(1 / n1 + 1 / n2)),
                )
                - power
            )
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.n1 is None or self.n2 is None or self.d is None or self.alpha is None:
                raise ValueError("n1, n2, d, and alpha must be provided to compute power")
            self.power = self._get_power(self.n1, self.n2, self.d, self.alpha)
        elif self.d is None:
            if self.n1 is None or self.n2 is None or self.alpha is None or self.power is None:
                raise ValueError("n1, n2, alpha, and power must be provided to solve for d")
            n1, n2, alpha, power = self.n1, self.n2, self.alpha, self.power
            if self.alternative == "two-sided":
                self.d = float(brentq(lambda d: self._get_effect_size(d, n1, n2, alpha, power), 1e-10, 10))
            elif self.alternative == "greater":
                self.d = float(brentq(lambda d: self._get_effect_size(d, n1, n2, alpha, power), -5, 10))
            else:
                self.d = float(brentq(lambda d: self._get_effect_size(d, n1, n2, alpha, power), -10, 5))
        elif self.n1 is None:
            if self.n2 is None or self.d is None or self.alpha is None or self.power is None:
                raise ValueError("n2, d, alpha, and power must be provided to solve for n1")
            n2, d, alpha, power = self.n2, self.d, self.alpha, self.power
            self.n1 = ceil(float(brentq(lambda n1: self._get_n1(n1, n2, d, alpha, power), 2 + 1e-10, 1e09)))
        elif self.n2 is None:
            if self.n1 is None or self.d is None or self.alpha is None or self.power is None:
                raise ValueError("n1, d, alpha, and power must be provided to solve for n2")
            n1, d, alpha, power = self.n1, self.d, self.alpha, self.power
            self.n2 = ceil(float(brentq(lambda n2: self._get_n2(n2, n1, d, alpha, power), 2 + 1e-10, 1e09)))
        else:
            if self.n1 is None or self.n2 is None or self.d is None or self.power is None:
                raise ValueError("n1, n2, d, and power must be provided to solve for alpha")
            n1, n2, d, power = self.n1, self.n2, self.d, self.power
            self.alpha = float(brentq(lambda alpha: self._get_alpha(alpha, n1, n2, d, power), 1e-10, 1 - 1e-10))
        return {
            "effect_size": self.d,
            "n1": self.n1,
            "n2": self.n2,
            "alpha": self.alpha,
            "power": self.power,
            "alternative": self.alternative,
            "method": self.method,
            "note": self.note,
            "url": self.url,
        }
