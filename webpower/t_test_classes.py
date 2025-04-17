from math import ceil, sqrt
from typing import Dict, Optional

from scipy.stats import nct, t as t_dist
from scipy.optimize import brentq


class WpOneT:
    def __init__(
        self,
        n: Optional[int] = None,
        d: Optional[float] = None,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
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

    def _get_power(self) -> float:
        nu = (self.n - 1) * self.t_sample
        if self.alternative == "two-sided":
            qu = t_dist.isf(self.alpha / 2, nu)
            power = nct.sf(qu, nu, sqrt(self.n / self.t_sample) * self.d) + nct.cdf(
                -qu, nu, sqrt(self.n / self.t_sample) * self.d
            )
        elif self.alternative == "greater":
            power = nct.sf(
                t_dist.isf(self.alpha, nu),
                nu,
                sqrt(self.n / self.t_sample) * self.d,
            )
        else:
            power = nct.cdf(
                t_dist.ppf(self.alpha, nu),
                nu,
                sqrt(self.n / self.t_sample) * self.d,
            )
        return power

    def _get_effect_size(self, effect_size: float) -> float:
        nu = (self.n - 1) * self.t_sample
        if self.alternative == "two-sided":
            qu = t_dist.isf(self.alpha / 2, nu)
            effect_size = (
                nct.sf(qu, nu, sqrt(self.n / self.t_sample) * effect_size)
                + nct.cdf(-qu, nu, sqrt(self.n / self.t_sample) * effect_size)
                - self.power
            )
        elif self.alternative == "greater":
            effect_size = (
                nct.sf(
                    t_dist.isf(self.alpha, nu),
                    nu,
                    sqrt(self.n / self.t_sample) * effect_size,
                )
                - self.power
            )
        else:
            effect_size = (
                nct.cdf(
                    t_dist.ppf(self.alpha, nu),
                    nu,
                    sqrt(self.n / self.t_sample) * effect_size,
                )
                - self.power
            )
        return effect_size

    def _get_n(self, n: int) -> float:
        nu = (n - 1) * self.t_sample
        if self.alternative == "two-sided":
            qu = t_dist.isf(self.alpha / 2, nu)
            n = (
                nct.sf(qu, nu, sqrt(n / self.t_sample) * self.d)
                + nct.cdf(-qu, nu, sqrt(n / self.t_sample) * self.d)
                - self.power
            )
        elif self.alternative == "greater":
            n = nct.sf(t_dist.isf(self.alpha, nu), nu, sqrt(n / self.t_sample) * self.d) - self.power
        else:
            n = nct.cdf(t_dist.ppf(self.alpha, nu), nu, sqrt(n / self.t_sample) * self.d) - self.power
        return n

    def _get_alpha(self, alpha: float) -> float:
        nu = (self.n - 1) * self.t_sample
        if self.alternative == "two-sided":
            qu = t_dist.isf(alpha / 2, nu)
            alpha = (
                nct.sf(qu, nu, sqrt(self.n / self.t_sample) * self.d)
                + nct.cdf(-qu, nu, sqrt(self.n / self.t_sample) * self.d)
                - self.power
            )
        elif self.alternative == "greater":
            alpha = nct.sf(t_dist.isf(alpha, nu), nu, sqrt(self.n / self.t_sample) * self.d) - self.power
        else:
            alpha = nct.cdf(t_dist.ppf(alpha, nu), nu, sqrt(self.n / self.t_sample) * self.d) - self.power
        return alpha

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.d is None:
            if self.alternative == "two-sided":
                self.d = brentq(self._get_effect_size, 1e-07, 10)
            elif self.alternative == "greater":
                self.d = brentq(self._get_effect_size, -5, 10)
            else:
                self.d = brentq(self._get_effect_size, -10, 5)
        elif self.n is None:
            self.n = ceil(brentq(self._get_n, 2 + 1e-10, 1e09))
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
        if self.note is not None:
            return {
                "n": self.n,
                "effect_size": self.d,
                "alpha": self.alpha,
                "power": self.power,
                "alternative": self.alternative,
                "method": "{} t test power calculation".format(self.method),
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
                "method": "{} t test power calculation".format(self.method),
                "url": self.url,
            }


class WpTwoT:
    def __init__(
        self,
        n1: Optional[int] = None,
        n2: Optional[int] = None,
        d: Optional[float] = None,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
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

    def _get_power(self) -> float:
        nu = self.n1 + self.n2 - 2
        if self.alternative == "two-sided":
            qu = t_dist.isf(self.alpha / 2, nu)
            power = nct.sf(qu, nu, self.d * (1 / sqrt(1 / self.n1 + 1 / self.n2))) + nct.cdf(
                -qu, nu, self.d * (1 / sqrt(1 / self.n1 + 1 / self.n2))
            )
        elif self.alternative == "greater":
            power = nct.sf(
                t_dist.isf(self.alpha, nu),
                nu,
                self.d * (1 / sqrt(1 / self.n1 + 1 / self.n2)),
            )
        else:
            power = nct.cdf(
                t_dist.ppf(self.alpha, nu),
                nu,
                self.d * (1 / sqrt(1 / self.n1 + 1 / self.n2)),
            )
        return power

    def _get_effect_size(self, effect_size: float) -> float:
        nu = self.n1 + self.n2 - 2
        if self.alternative == "two-sided":
            qu = t_dist.isf(self.alpha / 2, nu)
            effect_size = (
                nct.sf(qu, nu, effect_size * (1 / sqrt(1 / self.n1 + 1 / self.n2)))
                + nct.cdf(-qu, nu, effect_size * (1 / sqrt(1 / self.n1 + 1 / self.n2)))
                - self.power
            )
        elif self.alternative == "greater":
            effect_size = (
                nct.sf(
                    t_dist.isf(self.alpha, nu),
                    nu,
                    effect_size * (1 / sqrt(1 / self.n1 + 1 / self.n2)),
                )
                - self.power
            )
        else:
            effect_size = (
                nct.cdf(
                    t_dist.ppf(self.alpha, nu),
                    nu,
                    effect_size * (1 / sqrt(1 / self.n1 + 1 / self.n2)),
                )
                - self.power
            )
        return effect_size

    def _get_n1(self, n1: int) -> float:
        nu = n1 + self.n2 - 2
        if self.alternative == "two-sided":
            qu = t_dist.isf(self.alpha / 2, nu)
            n1 = (
                nct.sf(qu, nu, self.d * (1 / sqrt(1 / n1 + 1 / self.n2)))
                + nct.cdf(-qu, nu, self.d * (1 / sqrt(1 / n1 + 1 / self.n2)))
                - self.power
            )
        elif self.alternative == "greater":
            n1 = (
                nct.sf(
                    t_dist.isf(self.alpha, nu),
                    nu,
                    self.d * (1 / sqrt(1 / n1 + 1 / self.n2)),
                )
                - self.power
            )
        else:
            n1 = (
                nct.cdf(
                    t_dist.ppf(self.alpha, nu),
                    nu,
                    self.d * (1 / sqrt(1 / n1 + 1 / self.n2)),
                )
                - self.power
            )
        return n1

    def _get_n2(self, n2: int) -> float:
        nu = self.n1 + n2 - 2
        if self.alternative == "two-sided":
            qu = t_dist.isf(self.alpha / 2, nu)
            n2 = (
                nct.sf(qu, nu, self.d * (1 / sqrt(1 / self.n1 + 1 / n2)))
                + nct.cdf(-qu, nu, self.d * (1 / sqrt(1 / self.n1 + 1 / n2)))
                - self.power
            )
        elif self.alternative == "greater":
            n2 = (
                nct.sf(
                    t_dist.isf(self.alpha, nu),
                    nu,
                    self.d * (1 / sqrt(1 / self.n1 + 1 / n2)),
                )
                - self.power
            )
        else:
            n2 = (
                nct.cdf(
                    t_dist.ppf(self.alpha, nu),
                    nu,
                    self.d * (1 / sqrt(1 / self.n1 + 1 / n2)),
                )
                - self.power
            )
        return n2

    def _get_alpha(self, alpha: float) -> float:
        nu = self.n1 + self.n2 - 2
        if self.alternative == "two-sided":
            qu = t_dist.isf(alpha / 2, nu)
            alpha = (
                nct.sf(qu, nu, self.d * (1 / sqrt(1 / self.n1 + 1 / self.n2)))
                + nct.cdf(-qu, nu, self.d * (1 / sqrt(1 / self.n1 + 1 / self.n2)))
                - self.power
            )
        elif self.alternative == "greater":
            alpha = (
                nct.sf(
                    t_dist.isf(alpha, nu),
                    nu,
                    self.d * (1 / sqrt(1 / self.n1 + 1 / self.n2)),
                )
                - self.power
            )
        else:
            alpha = (
                nct.cdf(
                    t_dist.ppf(alpha, nu),
                    nu,
                    self.d * (1 / sqrt(1 / self.n1 + 1 / self.n2)),
                )
                - self.power
            )
        return alpha

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.d is None:
            if self.alternative == "two-sided":
                self.d = brentq(self._get_effect_size, 1e-10, 10)
            elif self.alternative == "greater":
                self.d = brentq(self._get_effect_size, -5, 10)
            else:
                self.d = brentq(self._get_effect_size, -10, 5)
        elif self.n1 is None:
            self.n1 = ceil(brentq(self._get_n1, 2 + 1e-10, 1e09))
        elif self.n2 is None:
            self.n2 = ceil(brentq(self._get_n2, 2 + 1e-10, 1e09))
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
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
