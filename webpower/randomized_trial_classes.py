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

    def _get_power(self, J: int, f: float, n: int) -> float:
        df = J - 1
        if self.test_type == "main":
            lamda1 = sqrt(J) * f / sqrt(4 / n + self.tau11 / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                power = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1)
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                power = nct.sf(t0, df, lamda1)
        else:
            df2 = J * (n - 2)
            f0 = f_dist.ppf(1 - self.alpha, df, df2)
            if self.test_type == "site":
                power = f_dist.sf(f0 / (n * self.tau00 / self.sg2 + 1), df, df2)
            else:
                power = f_dist.sf(f0 / (n * self.tau11 / self.sg2 / 4 + 1), df, df2)
        return float(power)

    def _get_n(self, n: float, f: float, J: int, power: float) -> float:
        df = J - 1
        if self.test_type == "main":
            lamda1 = sqrt(J) * f / sqrt(4 / n + self.tau11 / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                result = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1) - power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                result = nct.sf(t0, df, lamda1) - power
        else:
            df2 = J * (n - 2)
            f0 = f_dist.ppf(1 - self.alpha, df, df2)
            if self.test_type == "site":
                result = f_dist.sf(f0 / (n * self.tau00 / self.sg2 + 1), df, df2) - power
            else:
                result = f_dist.sf(f0 / (n * self.tau11 / self.sg2 / 4 + 1), df, df2) - power
        return float(result)

    def _get_J(self, J: float, f: float, n: int, power: float) -> float:
        df = J - 1
        if self.test_type == "main":
            lamda1 = sqrt(J) * f / sqrt(4 / n + self.tau11 / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                result = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1) - power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                result = nct.sf(t0, df, lamda1) - power
        else:
            df2 = J * (n - 2)
            f0 = f_dist.ppf(1 - self.alpha, df, df2)
            if self.test_type == "site":
                result = f_dist.sf(f0 / (n * self.tau00 / self.sg2 + 1), df, df2) - power
            else:
                result = f_dist.sf(f0 / (n * self.tau11 / self.sg2 / 4 + 1), df, df2) - power
        return float(result)

    def _get_f(self, f: float, J: int, n: int, power: float) -> float:
        df = J - 1
        if self.test_type == "main":
            lamda1 = sqrt(J) * f / sqrt(4 / n + self.tau11 / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                result = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1) - power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                result = nct.sf(t0, df, lamda1) - power
        else:
            raise NotImplementedError("f not needed for `site` or `variance` tests")
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.J is None or self.n is None:
                raise ValueError("J and n must be provided to compute power")
            if self.test_type == "main" and self.f is None:
                raise ValueError("f must be provided to compute power for main test")
            f = self.f if self.f is not None else 0.0
            self.power = self._get_power(self.J, f, self.n)
        elif self.n is None:
            if self.f is None or self.J is None or self.power is None:
                raise ValueError("f, J, and power must be provided to solve for n")
            f, J, power = self.f, self.J, self.power
            self.n = ceil(nuniroot(lambda n: self._get_n(n, f, J, power), 3 - 1e-10, 1e06))
        elif self.J is None:
            if self.f is None or self.n is None or self.power is None:
                raise ValueError("f, n, and power must be provided to solve for J")
            f, n, power = self.f, self.n, self.power
            self.J = ceil(nuniroot(lambda J: self._get_J(J, f, n, power), 1 + 1e-10, 1e3))
        else:
            if self.J is None or self.n is None or self.power is None:
                raise ValueError("J, n, and power must be provided to solve for f")
            J, n, power = self.J, self.n, self.power
            self.f = nuniroot(lambda f: self._get_f(f, J, n, power), 1e-07, 1e07)
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

    def _get_power(self, J: int, n: int, f1: float) -> float:
        df = J - 1
        if self.test_type == "main":
            lamda1 = sqrt(J) * f1 / sqrt(4.5 / n + 1.5 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                power = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1)
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                power = nct.sf(t0, df, lamda1)
        elif self.test_type == "treatment":
            lamda2 = sqrt(J) * self.f2 / sqrt(6 / n + 2 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                power = nct.sf(t0, df, lamda2) + nct.cdf(-t0, df, lamda2)
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                power = nct.sf(t0, df, lamda2)
        else:
            df1 = 2
            df2 = 2 * (J - 1)
            lamda1 = sqrt(J) * f1 / sqrt(4.5 / n + 1.5 * self.tau / self.sg2)
            lamda2 = sqrt(J) * self.f2 / sqrt(6 / n + 2 * self.tau / self.sg2)
            lamda3 = lamda1**2 + lamda2**2
            f0 = f_dist.ppf(1 - self.alpha, df1, df2)
            power = ncf.sf(f0, df1, df2, lamda3)
        return float(power)

    def _get_n(self, n: float, J: int, f1: float, power: float) -> float:
        df = J - 1
        if self.test_type == "main":
            lamda1 = sqrt(J) * f1 / sqrt(4.5 / n + 1.5 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                result = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1) - power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                result = nct.sf(t0, df, lamda1) - power
        elif self.test_type == "treatment":
            lamda2 = sqrt(J) * self.f2 / sqrt(6 / n + 2 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                result = nct.sf(t0, df, lamda2) + nct.cdf(-t0, df, lamda2) - power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                result = nct.sf(t0, df, lamda2) - power
        else:
            df1 = 2
            df2 = 2 * (J - 1)
            lamda1 = sqrt(J) * f1 / sqrt(4.5 / n + 1.5 * self.tau / self.sg2)
            lamda2 = sqrt(J) * self.f2 / sqrt(6 / n + 2 * self.tau / self.sg2)
            lamda3 = lamda1**2 + lamda2**2
            f0 = f_dist.ppf(1 - self.alpha, df1, df2)
            result = ncf.sf(f0, df1, df2, lamda3) - power
        return float(result)

    def _get_f1(self, f1: float, J: int, n: int, power: float) -> float:
        df = J - 1
        if self.test_type == "main":
            lamda1 = sqrt(J) * f1 / sqrt(4.5 / n + 1.5 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                result = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1) - power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                result = nct.sf(t0, df, lamda1) - power
        elif self.test_type == "treatment":
            raise ValueError("f1 not used if test_type is `treatment`")
        else:
            df1 = 2
            df2 = 2 * (J - 1)
            lamda1 = sqrt(J) * f1 / sqrt(4.5 / n + 1.5 * self.tau / self.sg2)
            lamda2 = sqrt(J) * self.f2 / sqrt(6 / n + 2 * self.tau / self.sg2)
            lamda3 = lamda1**2 + lamda2**2
            f0 = f_dist.ppf(1 - self.alpha, df1, df2)
            result = ncf.sf(f0, df1, df2, lamda3) - power
        return float(result)

    def _get_J(self, J: float, f1: float, n: int, power: float) -> float:
        df = J - 1
        if self.test_type == "main":
            lamda1 = sqrt(J) * f1 / sqrt(4.5 / n + 1.5 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                result = nct.sf(t0, df, lamda1) + nct.cdf(-t0, df, lamda1) - power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                result = nct.sf(t0, df, lamda1) - power
        elif self.test_type == "treatment":
            lamda2 = sqrt(J) * self.f2 / sqrt(6 / n + 2 * self.tau / self.sg2)
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - self.alpha / 2, df)
                result = nct.sf(t0, df, lamda2) + nct.cdf(-t0, df, lamda2) - power
            else:
                t0 = t_dist.ppf(1 - self.alpha, df)
                result = nct.sf(t0, df, lamda2) - power
        else:
            df1 = 2
            df2 = 2 * (J - 1)
            lamda1 = sqrt(J) * f1 / sqrt(4.5 / n + 1.5 * self.tau / self.sg2)
            lamda2 = sqrt(J) * self.f2 / sqrt(6 / n + 2 * self.tau / self.sg2)
            lamda3 = lamda1**2 + lamda2**2
            f0 = f_dist.ppf(1 - self.alpha, df1, df2)
            result = ncf.sf(f0, df1, df2, lamda3) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.J is None or self.n is None or self.f1 is None:
                raise ValueError("J, n, and f1 must be provided to compute power")
            self.power = self._get_power(self.J, self.n, self.f1)
        elif self.J is None:
            if self.f1 is None or self.n is None or self.power is None:
                raise ValueError("f1, n, and power must be provided to solve for J")
            f1, n, power = self.f1, self.n, self.power
            self.J = ceil(nuniroot(lambda J: self._get_J(J, f1, n, power), 2 - 1e-10, 1e03))
        elif self.n is None:
            if self.f1 is None or self.J is None or self.power is None:
                raise ValueError("f1, J, and power must be provided to solve for n")
            f1, J, power = self.f1, self.J, self.power
            self.n = ceil(nuniroot(lambda n: self._get_n(n, J, f1, power), 3 - 1e-10, 1e07))
        else:
            if self.J is None or self.n is None or self.power is None:
                raise ValueError("J, n, and power must be provided to solve for f1")
            J, n, power = self.J, self.n, self.power
            self.f1 = nuniroot(lambda f1: self._get_f1(f1, J, n, power), 1e-07, 1e07)
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

    def _get_power(self, J: int, f: float, icc: float, n: int, alpha: float) -> float:
        df = J - 2
        lamda = sqrt(J * f**2 / (4 * icc + 4 * (1 - icc) / n))
        if self.alternative == "two-sided":
            z_t = t_dist.ppf(1 - alpha / 2, df)
            power = nct.sf(z_t, df, lamda) + nct.cdf(-z_t, df, lamda)
        else:
            z_t = t_dist.ppf(1 - alpha, df)
            power = nct.sf(z_t, df, lamda)
        return float(power)

    def _get_effect_size(self, effect_size: float, J: int, icc: float, n: int, alpha: float, power: float) -> float:
        df = J - 2
        lamda = sqrt(J * effect_size**2 / (4 * icc + 4 * (1 - icc) / n))
        if self.alternative == "two-sided":
            z_t = t_dist.ppf(1 - alpha / 2, df)
            result = nct.sf(z_t, df, lamda) + nct.cdf(-z_t, df, lamda) - power
        else:
            z_t = t_dist.ppf(1 - alpha, df)
            result = nct.sf(z_t, df, lamda) - power
        return float(result)

    def _get_n(self, n: float, J: int, f: float, icc: float, alpha: float, power: float) -> float:
        df = J - 2
        lamda = sqrt(J * f**2 / (4 * icc + 4 * (1 - icc) / n))
        if self.alternative == "two-sided":
            z_t = t_dist.ppf(1 - alpha / 2, df)
            result = nct.sf(z_t, df, lamda) + nct.cdf(-z_t, df, lamda) - power
        else:
            z_t = t_dist.ppf(1 - alpha, df)
            result = nct.sf(z_t, df, lamda) - power
        return float(result)

    def _get_J(self, J: float, f: float, icc: float, n: int, alpha: float, power: float) -> float:
        df = J - 2
        lamda = sqrt(J * f**2 / (4 * icc + 4 * (1 - icc) / n))
        if self.alternative == "two-sided":
            z_t = t_dist.ppf(1 - alpha / 2, df)
            result = nct.sf(z_t, df, lamda) + nct.cdf(-z_t, df, lamda) - power
        else:
            z_t = t_dist.ppf(1 - alpha, df)
            result = nct.sf(z_t, df, lamda) - power
        return float(result)

    def _get_icc(self, icc: float, J: int, f: float, n: int, alpha: float, power: float) -> float:
        df = J - 2
        lamda = sqrt(J * f**2 / (4 * icc + 4 * (1 - icc) / n))
        if self.alternative == "two-sided":
            z_t = t_dist.ppf(1 - alpha / 2, df)
            result = nct.sf(z_t, df, lamda) + nct.cdf(-z_t, df, lamda) - power
        else:
            z_t = t_dist.ppf(1 - alpha, df)
            result = nct.sf(z_t, df, lamda) - power
        return float(result)

    def _get_alpha(self, alpha: float, J: int, f: float, icc: float, n: int, power: float) -> float:
        df = J - 2
        lamda = sqrt(J * f**2 / (4 * icc + 4 * (1 - icc) / n))
        if self.alternative == "two-sided":
            z_t = t_dist.ppf(1 - alpha / 2, df)
            result = nct.sf(z_t, df, lamda) + nct.cdf(-z_t, df, lamda) - power
        else:
            z_t = t_dist.ppf(1 - alpha, df)
            result = nct.sf(z_t, df, lamda) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.J is None or self.f is None or self.icc is None or self.n is None or self.alpha is None:
                raise ValueError("J, f, icc, n, and alpha must be provided to compute power")
            self.power = self._get_power(self.J, self.f, self.icc, self.n, self.alpha)
        elif self.n is None:
            if self.J is None or self.f is None or self.icc is None or self.alpha is None or self.power is None:
                raise ValueError("J, f, icc, alpha, and power must be provided to solve for n")
            J, f, icc, alpha, power = self.J, self.f, self.icc, self.alpha, self.power
            self.n = ceil(nuniroot(lambda n: self._get_n(n, J, f, icc, alpha, power), 1, 1e06))
        elif self.J is None:
            if self.f is None or self.icc is None or self.n is None or self.alpha is None or self.power is None:
                raise ValueError("f, icc, n, alpha, and power must be provided to solve for J")
            f, icc, n, alpha, power = self.f, self.icc, self.n, self.alpha, self.power
            self.J = ceil(nuniroot(lambda J: self._get_J(J, f, icc, n, alpha, power), 2 + 1e-10, 1_000))
        elif self.f is None:
            if self.J is None or self.icc is None or self.n is None or self.alpha is None or self.power is None:
                raise ValueError("J, icc, n, alpha, and power must be provided to solve for f")
            J, icc, n, alpha, power = self.J, self.icc, self.n, self.alpha, self.power
            self.f = nuniroot(lambda f: self._get_effect_size(f, J, icc, n, alpha, power), 1e-07, 1e07)
        elif self.icc is None:
            if self.J is None or self.f is None or self.n is None or self.alpha is None or self.power is None:
                raise ValueError("J, f, n, alpha, and power must be provided to solve for icc")
            J, f, n, alpha, power = self.J, self.f, self.n, self.alpha, self.power
            self.icc = nuniroot(lambda icc: self._get_icc(icc, J, f, n, alpha, power), 0, 1)
        else:
            if self.J is None or self.f is None or self.icc is None or self.n is None or self.power is None:
                raise ValueError("J, f, icc, n, and power must be provided to solve for alpha")
            J, f, icc, n, power = self.J, self.f, self.icc, self.n, self.power
            self.alpha = nuniroot(lambda alpha: self._get_alpha(alpha, J, f, icc, n, power), 1e-10, 1 - 1e-10)
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

    def _get_power(self, J: int, f: float, icc: float, n: int, alpha: float) -> float:
        df = J - 3
        if self.test_type == "main":
            lambda1 = sqrt(J) * f / sqrt(4.5 * (icc + (1 - icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                power = nct.sf(t0, df, lambda1) + nct.cdf(-t0, df, lambda1)
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                power = nct.sf(t0, df, lambda1)
        elif self.test_type == "treatment":
            lambda2 = sqrt(J) * f / sqrt(6 * (icc + (1 - icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                power = nct.sf(t0, df, lambda2) + nct.cdf(-t0, df, lambda2)
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                power = nct.sf(t0, df, lambda2)
        else:
            df1 = 2
            lambda3 = J * f**2 / (icc + (1 - icc) / n)
            f0 = f_dist.ppf(1 - alpha, df1, df)
            power = ncf.sf(f0, df1, df, lambda3)
        return float(power)

    def _get_effect_size(self, effect_size: float, J: int, icc: float, n: int, alpha: float, power: float) -> float:
        df = J - 3
        if self.test_type == "main":
            lambda1 = sqrt(J) * effect_size / sqrt(4.5 * (icc + (1 - icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                result = nct.sf(t0, df, lambda1) + nct.cdf(-t0, df, lambda1) - power
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                result = nct.sf(t0, df, lambda1) - power
        elif self.test_type == "treatment":
            lambda2 = sqrt(J) * effect_size / sqrt(6 * (icc + (1 - icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                result = nct.sf(t0, df, lambda2) + nct.cdf(-t0, df, lambda2) - power
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                result = nct.sf(t0, df, lambda2) - power
        else:
            df1 = 2
            lambda3 = J * effect_size**2 / (icc + (1 - icc) / n)
            f0 = f_dist.ppf(1 - alpha, df1, df)
            result = ncf.sf(f0, df1, df, lambda3) - power
        return float(result)

    def _get_n(self, n: float, J: int, f: float, icc: float, alpha: float, power: float) -> float:
        df = J - 3
        if self.test_type == "main":
            lambda1 = sqrt(J) * f / sqrt(4.5 * (icc + (1 - icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                result = nct.sf(t0, df, lambda1) + nct.cdf(-t0, df, lambda1) - power
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                result = nct.sf(t0, df, lambda1) - power
        elif self.test_type == "treatment":
            lambda2 = sqrt(J) * f / sqrt(6 * (icc + (1 - icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                result = nct.sf(t0, df, lambda2) + nct.cdf(-t0, df, lambda2) - power
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                result = nct.sf(t0, df, lambda2) - power
        else:
            df1 = 2
            lambda3 = J * f**2 / (icc + (1 - icc) / n)
            f0 = f_dist.ppf(1 - alpha, df1, df)
            result = ncf.sf(f0, df1, df, lambda3) - power
        return float(result)

    def _get_J(self, J: float, f: float, icc: float, n: int, alpha: float, power: float) -> float:
        df = J - 3
        if self.test_type == "main":
            lambda1 = sqrt(J) * f / sqrt(4.5 * (icc + (1 - icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                result = nct.sf(t0, df, lambda1) + nct.cdf(-t0, df, lambda1) - power
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                result = nct.sf(t0, df, lambda1) - power
        elif self.test_type == "treatment":
            lambda2 = sqrt(J) * f / sqrt(6 * (icc + (1 - icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                result = nct.sf(t0, df, lambda2) + nct.cdf(-t0, df, lambda2) - power
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                result = nct.sf(t0, df, lambda2) - power
        else:
            df1 = 2
            lambda3 = J * f**2 / (icc + (1 - icc) / n)
            f0 = f_dist.ppf(1 - alpha, df1, df)
            result = ncf.sf(f0, df1, df, lambda3) - power
        return float(result)

    def _get_icc(self, icc: float, J: int, f: float, n: int, alpha: float, power: float) -> float:
        df = J - 3
        if self.test_type == "main":
            lambda1 = sqrt(J) * f / sqrt(4.5 * (icc + (1 - icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                result = nct.sf(t0, df, lambda1) + nct.cdf(-t0, df, lambda1) - power
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                result = nct.sf(t0, df, lambda1) - power
        elif self.test_type == "treatment":
            lambda2 = sqrt(J) * f / sqrt(6 * (icc + (1 - icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                result = nct.sf(t0, df, lambda2) + nct.cdf(-t0, df, lambda2) - power
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                result = nct.sf(t0, df, lambda2) - power
        else:
            df1 = 2
            lambda3 = J * f**2 / (icc + (1 - icc) / n)
            f0 = f_dist.ppf(1 - alpha, df1, df)
            result = ncf.sf(f0, df1, df, lambda3) - power
        return float(result)

    def _get_alpha(self, alpha: float, J: int, f: float, icc: float, n: int, power: float) -> float:
        df = J - 3
        if self.test_type == "main":
            lambda1 = sqrt(J) * f / sqrt(4.5 * (icc + (1 - icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                result = nct.sf(t0, df, lambda1) + nct.cdf(-t0, df, lambda1) - power
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                result = nct.sf(t0, df, lambda1) - power
        elif self.test_type == "treatment":
            lambda2 = sqrt(J) * f / sqrt(6 * (icc + (1 - icc) / n))
            if self.alternative == "two-sided":
                t0 = t_dist.ppf(1 - alpha / 2, df)
                result = nct.sf(t0, df, lambda2) + nct.cdf(-t0, df, lambda2) - power
            else:
                t0 = t_dist.ppf(1 - alpha, df)
                result = nct.sf(t0, df, lambda2) - power
        else:
            df1 = 2
            lambda3 = J * f**2 / (icc + (1 - icc) / n)
            f0 = f_dist.ppf(1 - alpha, df1, df)
            result = ncf.sf(f0, df1, df, lambda3) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.J is None or self.f is None or self.icc is None or self.n is None or self.alpha is None:
                raise ValueError("J, f, icc, n, and alpha must be provided to compute power")
            self.power = self._get_power(self.J, self.f, self.icc, self.n, self.alpha)
        elif self.f is None:
            if self.J is None or self.icc is None or self.n is None or self.alpha is None or self.power is None:
                raise ValueError("J, icc, n, alpha, and power must be provided to solve for f")
            J, icc, n, alpha, power = self.J, self.icc, self.n, self.alpha, self.power
            self.f = nuniroot(lambda f: self._get_effect_size(f, J, icc, n, alpha, power), 1e-07, 1e07)
        elif self.n is None:
            if self.J is None or self.f is None or self.icc is None or self.alpha is None or self.power is None:
                raise ValueError("J, f, icc, alpha, and power must be provided to solve for n")
            J, f, icc, alpha, power = self.J, self.f, self.icc, self.alpha, self.power
            self.n = ceil(nuniroot(lambda n: self._get_n(n, J, f, icc, alpha, power), 2 + 1e-10, 1e06))
        elif self.J is None:
            if self.f is None or self.icc is None or self.n is None or self.alpha is None or self.power is None:
                raise ValueError("f, icc, n, alpha, and power must be provided to solve for J")
            f, icc, n, alpha, power = self.f, self.icc, self.n, self.alpha, self.power
            self.J = ceil(nuniroot(lambda J: self._get_J(J, f, icc, n, alpha, power), 3 + 1e-10, 1_000))
        elif self.icc is None:
            if self.J is None or self.f is None or self.n is None or self.alpha is None or self.power is None:
                raise ValueError("J, f, n, alpha, and power must be provided to solve for icc")
            J, f, n, alpha, power = self.J, self.f, self.n, self.alpha, self.power
            self.icc = nuniroot(lambda icc: self._get_icc(icc, J, f, n, alpha, power), 0, 1)
        else:
            if self.J is None or self.f is None or self.icc is None or self.n is None or self.power is None:
                raise ValueError("J, f, icc, n, and power must be provided to solve for alpha")
            J, f, icc, n, power = self.J, self.f, self.icc, self.n, self.power
            self.alpha = nuniroot(lambda alpha: self._get_alpha(alpha, J, f, icc, n, power), 1e-10, 1 - 1e-10)
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
