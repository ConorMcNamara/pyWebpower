from math import ceil, sqrt

from scipy.stats import f as f_dist
from scipy.stats import ncf, nct
from scipy.stats import t as t_dist

from webpower.utils import nuniroot


class WpMRT2Arm:
    def __init__(
        self,
        n: int | None = None,
        f: float | None = None,
        J: int | None = None,
        tau00: float = 1.0,
        tau11: float = 1.0,
        sg2: float = 1.0,
        power: float | None = None,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        test_type: str = "main",
    ) -> None:
        self.n = n
        self.f = f
        self.J = J
        self.tau00 = tau00
        self.tau11 = tau11
        self.sg2 = sg2
        self.power = power
        self.alpha = alpha
        self.alternative = alternative.casefold()
        self.test_type = test_type.casefold()
        self.note = "n is the number of subjects per cluster"
        self.method = "Power analysis for Multilevel model Multisite randomized trials with 2 arms"
        self.url = "http://psychstat.org/mrt2arm"

    def _get_power(self) -> float:
        df = self.J - 1
        if self.test_type == "main":
            lamda1 = sqrt(self.J) * self.f / sqrt(4 / self.n + self.tau11 / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                power = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1)
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                power = nct.sf(t0, df, lamda1)
        else:
            df2 = self.J * (self.n - 2)
            f0 = f_dist.ppf(1 - self.alpha, df, df2)
            if self.test_type == "site":
                power = f_dist.sf(f0 / (self.n * self.tau00 / self.sg2 + 1), df, df2)
            else:
                power = f_dist.sf(f0 / (self.n * self.tau11 / self.sg2 / 4 + 1), df, df2)
        return float(power)

    def _get_n(self, n: int) -> float:
        df = self.J - 1
        if self.test_type == "main":
            lamda1 = sqrt(self.J) * self.f / sqrt(4 / n + self.tau11 / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                n = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                n = nct.sf(t0, df, lamda1) - self.power
        else:
            df2 = self.J * (self.n - 2)
            f0 = f_dist.ppf(1 - self.alpha, df, df2)
            if self.test_type == "site":
                n = f_dist.sf(f0 / (self.n * self.tau00 / self.sg2 + 1), df, df2) - self.power
            else:
                n = f_dist.sf(f0 / (self.n * self.tau11 / self.sg2 / 4 + 1), df, df2) - self.power
        return float(n)

    def _get_J(self, J: int) -> float:
        df = J - 1
        if self.test_type == "main":
            lamda1 = sqrt(J) * self.f / sqrt(4 / self.n + self.tau11 / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                J = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                J = nct.sf(t0, df, lamda1) - self.power
        else:
            df2 = J * (self.n - 2)
            f0 = f_dist.ppf(1 - self.alpha, df, df2)
            if self.test_type == "site":
                J = f_dist.sf(f0 / (self.n * self.tau00 / self.sg2 + 1), df, df2) - self.power
            else:
                J = f_dist.sf(f0 / (self.n * self.tau11 / self.sg2 / 4 + 1), df, df2) - self.power
        return float(J)

    def _get_f(self, f: float) -> float:
        df = self.J - 1
        if self.test_type == "main":
            lamda1 = sqrt(self.J) * f / sqrt(4 / self.n + self.tau11 / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                f = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                f = nct.sf(t0, df, lamda1) - self.power
        else:
            raise NotImplementedError("f not needed for `site` or `variance` tests")
        return float(f)

    def pwr_test(self) -> dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.n is None:
            self.n = ceil(nuniroot(self._get_n, 3 - 1e-10, 1e06))
        elif self.J is None:
            self.J = ceil(nuniroot(self._get_J, 1 + 1e-10, 1e3))
        else:
            self.f = nuniroot(self._get_f, 1e-07, 1e07)
        return {
            "J": self.J,
            "n": self.n,
            "effect_size": self.f,
            "tau00": self.tau00,
            "tau11": self.tau11,
            "sg2": self.sg2,
            "power": self.power,
            "alpha": self.alpha,
            "note": self.note,
            "method": self.method,
            "url": self.url,
        }


class WpMRT3Arm:
    def __init__(
        self,
        n: int | None = None,
        f1: float | None = None,
        f2: float = 0.0,
        J: int | None = None,
        tau: float = 1.0,
        sg2: float = 1.0,
        power: float | None = None,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        test_type: str = "main",
    ) -> None:
        self.n = n
        self.f1 = f1
        self.f2 = f2
        self.J = J
        self.tau = tau
        self.sg2 = sg2
        self.power = power
        self.alpha = alpha
        self.alternative = alternative.casefold()
        self.test_type = test_type.casefold()
        self.note = "n is the number of subjects per cluster"
        self.method = "Multisite randomized trials with 3 arms"
        self.url = "http://psychstat.org/mrt3arm"

    def _get_power(self) -> float:
        df = self.J - 1
        if self.test_type == "main":
            lamda1 = sqrt(self.J) * self.f1 / sqrt(4.5 / self.n + 1.5 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                power = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1)
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                power = nct.sf(t0, df, lamda1)
        elif self.test_type == "treatment":
            lamda2 = sqrt(self.J) * self.f2 / sqrt(6 / self.n + 2 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                power = nct.sf(t0, df, lamda2) + nct.cdf(-t0, df, lamda2)
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                power = nct.sf(t0, df, lamda2)
        else:
            df1 = 2
            df2 = 2 * (self.J - 1)
            lamda1 = sqrt(self.J) * self.f1 / sqrt(4.5 / self.n + 1.5 * self.tau / self.sg2)
            lamda2 = sqrt(self.J) * self.f2 / sqrt(6 / self.n + 2 * self.tau / self.sg2)
            lamda3 = lamda1**2 + lamda2**2
            f0 = f_dist.ppf(1 - self.alpha, df1, df2)
            power = ncf.sf(f0, df1, df2, lamda3)
        return float(power)

    def _get_n(self, n: int) -> float:
        df = self.J - 1
        if self.test_type == "main":
            lamda1 = sqrt(self.J) * self.f1 / sqrt(4.5 / n + 1.5 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                n = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                n = nct.sf(t0, df, lamda1) - self.power
        elif self.test_type == "treatment":
            lamda2 = sqrt(self.J) * self.f2 / sqrt(6 / n + 2 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                n = nct.sf(t0, df, lamda2) + nct.cdf(-t0, df, lamda2) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                n = nct.sf(t0, df, lamda2) - self.power
        else:
            df1 = 2
            df2 = 2 * (self.J - 1)
            lamda1 = sqrt(self.J) * self.f1 / sqrt(4.5 / n + 1.5 * self.tau / self.sg2)
            lamda2 = sqrt(self.J) * self.f2 / sqrt(6 / n + 2 * self.tau / self.sg2)
            lamda3 = lamda1**2 + lamda2**2
            f0 = f_dist.ppf(1 - self.alpha, df1, df2)
            n = ncf.sf(f0, df1, df2, lamda3) - self.power
        return float(n)

    def _get_f1(self, f1: float) -> float:
        df = self.J - 1
        if self.test_type == "main":
            lamda1 = sqrt(self.J) * f1 / sqrt(4.5 / self.n + 1.5 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                f1 = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                f1 = nct.sf(t0, df, lamda1) - self.power
        elif self.test_type == "treatment":
            raise ValueError("f1 not used if test_type is `treatment`")
        else:
            df1 = 2
            df2 = 2 * (self.J - 1)
            lamda1 = sqrt(self.J) * f1 / sqrt(4.5 / self.n + 1.5 * self.tau / self.sg2)
            lamda2 = sqrt(self.J) * self.f2 / sqrt(6 / self.n + 2 * self.tau / self.sg2)
            lamda3 = lamda1**2 + lamda2**2
            f0 = f_dist.ppf(1 - self.alpha, df1, df2)
            f1 = ncf.sf(f0, df1, df2, lamda3) - self.power
        return float(f1)

    def _get_J(self, J: int) -> float:
        df = J - 1
        if self.test_type == "main":
            lamda1 = sqrt(J) * self.f1 / sqrt(4.5 / self.n + 1.5 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                J = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                J = nct.sf(t0, df, lamda1) - self.power
        elif self.test_type == "treatment":
            lamda2 = sqrt(J) * self.f2 / sqrt(6 / self.n + 2 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                J = nct.sf(t0, df, lamda2) + nct.cdf(-t0, df, lamda2) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                J = nct.sf(t0, df, lamda2) - self.power
        else:
            df1 = 2
            df2 = 2 * (J - 1)
            lamda1 = sqrt(J) * self.f1 / sqrt(4.5 / self.n + 1.5 * self.tau / self.sg2)
            lamda2 = sqrt(J) * self.f2 / sqrt(6 / self.n + 2 * self.tau / self.sg2)
            lamda3 = lamda1**2 + lamda2**2
            f0 = f_dist.ppf(1 - self.alpha, df1, df2)
            J = ncf.sf(f0, df1, df2, lamda3) - self.power
        return float(J)

    def pwr_test(self) -> dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.J is None:
            self.J = ceil(nuniroot(self._get_J, 2 - 1e-10, 1e03))
        elif self.n is None:
            self.n = ceil(nuniroot(self._get_n, 3 - 1e-10, 1e07))
        else:
            self.f1 = nuniroot(self._get_f1, 1e-07, 1e07)
        return {
            "power": self.power,
            "J": self.J,
            "n": self.n,
            "f1": self.f1,
            "f2": self.f2,
            "alpha": self.alpha,
            "tau": self.tau,
            "sg2": self.sg2,
            "note": self.note,
            "method": self.method,
            "url": self.url,
        }


class WpCRT2Arm:
    def __init__(
        self,
        n: int | None = None,
        f: float | None = None,
        J: int | None = None,
        icc: float | None = None,
        power: float | None = None,
        alpha: float | None = None,
        alternative: str = "two-sided",
    ) -> None:
        self.n = n
        self.f = f
        self.J = J
        self.icc = icc
        self.power = power
        self.alpha = alpha
        self.alternative = alternative.casefold()
        self.method = "Cluster randomized trials with 2 arms"
        self.note = "n is the number of subjects per cluster."
        self.url = "http://psychstat.org/crt2arm"

    def _get_power(self) -> float:
        df = self.J - 2
        lamda = sqrt(self.J * self.f**2 / (4 * self.icc + 4 * (1 - self.icc) / self.n))
        if self.alternative == "two-sided":
            z_t = t_dist.ppf(1 - self.alpha / 2, df)
            power = nct.sf(z_t, df, lamda) + nct.cdf(-z_t, df, lamda)
        else:
            z_t = t_dist.ppf(1 - self.alpha, df)
            power = nct.sf(z_t, df, lamda)
        return float(power)

    def _get_effect_size(self, effect_size: float) -> float:
        df = self.J - 2
        lamda = sqrt(self.J * effect_size**2 / (4 * self.icc + 4 * (1 - self.icc) / self.n))
        if self.alternative == "two-sided":
            z_t = t_dist.ppf(1 - self.alpha / 2, df)
            effect_size = nct.sf(z_t, df, lamda) + nct.cdf(-z_t, df, lamda) - self.power
        else:
            z_t = t_dist.ppf(1 - self.alpha, df)
            effect_size = nct.sf(z_t, df, lamda) - self.power
        return float(effect_size)

    def _get_n(self, n: int) -> float:
        df = self.J - 2
        lamda = sqrt(self.J * self.f**2 / (4 * self.icc + 4 * (1 - self.icc) / n))
        if self.alternative == "two-sided":
            z_t = t_dist.ppf(1 - self.alpha / 2, df)
            n = nct.sf(z_t, df, lamda) + nct.cdf(-z_t, df, lamda) - self.power
        else:
            z_t = t_dist.ppf(1 - self.alpha, df)
            n = nct.sf(z_t, df, lamda) - self.power
        return float(n)

    def _get_J(self, J: int) -> float:
        df = J - 2
        lamda = sqrt(J * self.f**2 / (4 * self.icc + 4 * (1 - self.icc) / self.n))
        if self.alternative == "two-sided":
            z_t = t_dist.ppf(1 - self.alpha / 2, df)
            J = nct.sf(z_t, df, lamda) + nct.cdf(-z_t, df, lamda) - self.power
        else:
            z_t = t_dist.ppf(1 - self.alpha, df)
            J = nct.sf(z_t, df, lamda) - self.power
        return float(J)

    def _get_icc(self, icc: float) -> float:
        df = self.J - 2
        lamda = sqrt(self.J * self.f**2 / (4 * icc + 4 * (1 - icc) / self.n))
        if self.alternative == "two-sided":
            z_t = t_dist.ppf(1 - self.alpha / 2, df)
            icc = nct.sf(z_t, df, lamda) + nct.cdf(-z_t, df, lamda) - self.power
        else:
            z_t = t_dist.ppf(1 - self.alpha, df)
            icc = nct.sf(z_t, df, lamda) - self.power
        return float(icc)

    def _get_alpha(self, alpha: float) -> float:
        df = self.J - 2
        lamda = sqrt(self.J * self.f**2 / (4 * self.icc + 4 * (1 - self.icc) / self.n))
        if self.alternative == "two-sided":
            z_t = t_dist.ppf(1 - alpha / 2, df)
            alpha = nct.sf(z_t, df, lamda) + nct.cdf(-z_t, df, lamda) - self.power
        else:
            z_t = t_dist.ppf(1 - alpha, df)
            alpha = nct.sf(z_t, df, lamda) - self.power
        return float(alpha)

    def pwr_test(self) -> dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.n is None:
            self.n = ceil(nuniroot(self._get_n, 1, 1e06))
        elif self.J is None:
            self.J = ceil(nuniroot(self._get_J, 2 + 1e-10, 1_000))
        elif self.f is None:
            self.f = nuniroot(self._get_effect_size, 1e-07, 1e07)
        elif self.icc is None:
            self.icc = nuniroot(self._get_icc, 0, 1)
        else:
            self.alpha = nuniroot(self._get_alpha, 1e-10, 1 - 1e-10)
        return {
            "J": self.J,
            "n": self.n,
            "effect_size": self.f,
            "icc": self.icc,
            "power": self.power,
            "alpha": self.alpha,
            "note": self.note,
            "method": self.method,
            "url": self.url,
            "alternative": self.alternative,
        }


class WpCRT3Arm:
    def __init__(
        self,
        n: int | None = None,
        f: float | None = None,
        J: int | None = None,
        icc: float | None = None,
        power: float | None = None,
        alpha: float | None = None,
        alternative: str = "two-sided",
        test_type: str = "main",
    ) -> None:
        self.n = n
        self.f = f
        self.J = J
        self.icc = icc
        self.power = power
        self.alpha = alpha
        self.alternative = alternative.casefold()
        self.test_type = test_type.casefold()
        self.note = "n is the number of subjects per cluster."
        self.method = "Cluster randomized trials with 3 arms"
        self.url = "http://psychstat.org/crt3arm"

    def _get_power(self) -> float:
        df = self.J - 3
        if self.test_type == "main":
            lambda1 = sqrt(self.J) * self.f / sqrt(4.5 * (self.icc + (1 - self.icc) / self.n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                power = nct.sf(t0, df, lambda1) + nct.cdf(-t0, df, lambda1)
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                power = nct.sf(t0, df, lambda1)
        elif self.test_type == "treatment":
            lambda2 = sqrt(self.J) * self.f / sqrt(6 * (self.icc + (1 - self.icc) / self.n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                power = nct.sf(t0, df, lambda2) + nct.cdf(-t0, df, lambda2)
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                power = nct.sf(t0, df, lambda2)
        else:
            df1 = 2
            lambda3 = self.J * self.f**2 / (self.icc + (1 - self.icc) / self.n)
            f0 = f_dist.ppf(1 - self.alpha, df, df1)
            power = ncf.sf(f0, df1, df, lambda3)
        return float(power)

    def _get_effect_size(self, effect_size: float) -> float:
        df = self.J - 3
        if self.test_type == "main":
            lambda1 = sqrt(self.J) * effect_size / sqrt(4.5 * (self.icc + (1 - self.icc) / self.n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                effect_size = nct.sf(t0, df, lambda1) + nct.cdf(-t0, df, lambda1) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                effect_size = nct.sf(t0, df, lambda1) - self.power
        elif self.test_type == "treatment":
            lambda2 = sqrt(self.J) * effect_size / sqrt(6 * (self.icc + (1 - self.icc) / self.n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                effect_size = nct.sf(t0, df, lambda2) + nct.cdf(-t0, df, lambda2) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                effect_size = nct.sf(t0, df, lambda2) - self.power
        else:
            df1 = 2
            lambda3 = self.J * effect_size**2 / (self.icc + (1 - self.icc) / self.n)
            f0 = f_dist.ppf(1 - self.alpha, df1, df)
            effect_size = ncf.sf(f0, df1, df, lambda3) - self.power
        return float(effect_size)

    def _get_n(self, n: int) -> float:
        df = self.J - 3
        if self.test_type == "main":
            lambda1 = sqrt(self.J) * self.f / sqrt(4.5 * (self.icc + (1 - self.icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                n = nct.sf(t0, df, lambda1) + nct.cdf(-t0, df, lambda1) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                n = nct.sf(t0, df, lambda1) - self.power
        elif self.test_type == "treatment":
            lambda2 = sqrt(self.J) * self.f / sqrt(6 * (self.icc + (1 - self.icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                n = nct.sf(t0, df, lambda2) + nct.cdf(-t0, df, lambda2) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                n = nct.sf(t0, df, lambda2) - self.power
        else:
            df1 = 2
            lambda3 = self.J * self.f**2 / (self.icc + (1 - self.icc) / n)
            f0 = f_dist.ppf(1 - self.alpha, df1, df)
            n = ncf.sf(f0, df1, df, lambda3) - self.power
        return float(n)

    def _get_J(self, J: int) -> float:
        df = J - 3
        if self.test_type == "main":
            lambda1 = sqrt(J) * self.f / sqrt(4.5 * (self.icc + (1 - self.icc) / self.n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                J = nct.sf(t0, df, lambda1) + nct.cdf(-t0, df, lambda1) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                J = nct.sf(t0, df, lambda1) - self.power
        elif self.test_type == "treatment":
            lambda2 = sqrt(J) * self.f / sqrt(6 * (self.icc + (1 - self.icc) / self.n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                J = nct.sf(t0, df, lambda2) + nct.cdf(-t0, df, lambda2) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                J = nct.sf(t0, df, lambda2) - self.power
        else:
            df1 = 2
            lambda3 = J * self.f**2 / (self.icc + (1 - self.icc) / self.n)
            f0 = f_dist.ppf(1 - self.alpha, df1, df)
            J = ncf.sf(f0, df1, df, lambda3) - self.power
        return float(J)

    def _get_icc(self, icc: float) -> float:
        df = self.J - 3
        if self.test_type == "main":
            lambda1 = sqrt(self.J) * self.f / sqrt(4.5 * (icc + (1 - icc) / self.n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                icc = nct.sf(t0, df, lambda1) + nct.cdf(-t0, df, lambda1) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                icc = nct.sf(t0, df, lambda1) - self.power
        elif self.test_type == "treatment":
            lambda2 = sqrt(self.J) * self.f / sqrt(6 * (icc + (1 - icc) / self.n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                icc = nct.sf(t0, df, lambda2) + nct.cdf(-t0, df, lambda2) - self.power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                icc = nct.sf(t0, df, lambda2) - self.power
        else:
            df1 = 2
            lambda3 = self.J * self.f**2 / (icc + (1 - icc) / self.n)
            f0 = f_dist.ppf(1 - self.alpha, df, df1)
            icc = ncf.sf(f0, df1, df, lambda3) - self.power
        return float(icc)

    def _get_alpha(self, alpha: float) -> float:
        df = self.J - 3
        if self.test_type == "main":
            lambda1 = sqrt(self.J) * self.f / sqrt(4.5 * (self.icc + (1 - self.icc) / self.n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                alpha = nct.sf(t0, df, lambda1) + nct.cdf(-t0, df, lambda1) - self.power
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                alpha = nct.sf(t0, df, lambda1) - self.power
        elif self.test_type == "treatment":
            lambda2 = sqrt(self.J) * self.f / sqrt(6 * (self.icc + (1 - self.icc) / self.n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                alpha = nct.sf(t0, df, lambda2) + nct.cdf(-t0, df, lambda2) - self.power
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                alpha = nct.sf(t0, df, lambda2) - self.power
        else:
            df1 = 2
            lambda3 = self.J * self.f**2 / (self.icc + (1 - self.icc) / self.n)
            f0 = f_dist.ppf(1 - alpha, df, df1)
            alpha = ncf.sf(f0, df1, df, lambda3) - self.power
        return float(alpha)

    def pwr_test(self) -> dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.f is None:
            self.f = nuniroot(self._get_effect_size, 1e-07, 1e07)
        elif self.n is None:
            self.n = ceil(nuniroot(self._get_n, 2 + 1e-10, 1e06))
        elif self.J is None:
            self.J = ceil(nuniroot(self._get_J, 3 + 1e-10, 1_000))
        elif self.icc is None:
            self.icc = nuniroot(self._get_icc, 0, 1)
        else:
            self.alpha = nuniroot(self._get_alpha, 1e-10, 1 - 1e-10)
        return {
            "J": self.J,
            "n": self.n,
            "effect_size": self.f,
            "icc": self.icc,
            "power": self.power,
            "alpha": self.alpha,
            "note": self.note,
            "method": self.method,
            "url": self.url,
        }
