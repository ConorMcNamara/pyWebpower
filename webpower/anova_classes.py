from math import ceil, sqrt

from scipy.optimize import bisect
from scipy.stats import chi2, ncf, nct, ncx2
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from webpower.utils import brentq


class WpAnovaClass:
    def __init__(
        self,
        k: int | None = None,
        n: int | None = None,
        f: float | None = None,
        alpha: float | None = None,
        power: float | None = None,
        test_type: str = "overall",
    ) -> None:
        self.k = k
        self.n = n
        self.f = f
        self.alpha = alpha
        self.power = power
        self.test_type = test_type.casefold()
        self.method = "Power for One-way ANOVA"
        if self.test_type == "overall":
            self.note = "n is the total sample size (overall)"
        elif self.test_type == "greater":
            self.note = "n is the total sample size (contrast, greater)"
        elif self.test_type == "lower":
            self.note = "n is the total sample size (contrast, less)"
        else:
            self.note = "n is the total sample size (contrast, two-sided)"
        self.url = "http://psychstat.org/anova"

    def _get_power(self, n: int, f: float, k: int, alpha: float) -> float:
        if self.test_type == "overall":
            lambda_ = n * f**2
            power = ncf.sf(
                f_dist.isf(alpha, k - 1, n - k),
                k - 1,
                n - k,
                lambda_,
            )
        elif self.test_type == "two-sided":
            lambda_ = n * f**2
            power = ncf.sf(
                f_dist.isf(alpha, k - 1, n - k),
                1,
                n - k,
                lambda_,
            )
        elif self.test_type == "greater":
            lambda_ = sqrt(n) * f
            power = nct.sf(t_dist.isf(alpha, n - k), n - k, lambda_)
        else:
            lambda_ = sqrt(n) * f
            power = nct.cdf(t_dist.ppf(alpha, n - k), n - k, lambda_)
        return float(power)

    def _get_groups(self, k: float, n: int, f: float, alpha: float, power: float) -> float:
        if self.test_type == "overall":
            lambda_ = n * f**2
            result = (
                ncf.sf(
                    f_dist.isf(alpha, k - 1, n - k),
                    k - 1,
                    n - k,
                    lambda_,
                )
                - power
            )
        elif self.test_type == "two-sided":
            lambda_ = n * f**2
            result = (
                ncf.sf(
                    f_dist.isf(alpha, k - 1, n - k),
                    1,
                    n - k,
                    lambda_,
                )
                - power
            )
        elif self.test_type == "greater":
            lambda_ = sqrt(n) * f
            result = nct.sf(t_dist.isf(alpha, n - k), n - k, lambda_) - power
        else:
            lambda_ = sqrt(n) * f
            result = nct.cdf(t_dist.ppf(alpha, n - k), n - k, lambda_) - power
        return float(result)

    def _get_sample_size(self, n: float, f: float, k: int, alpha: float, power: float) -> float:
        if self.test_type == "overall":
            lambda_ = n * f**2
            result = (
                ncf.sf(
                    f_dist.isf(alpha, k - 1, n - k),
                    k - 1,
                    n - k,
                    lambda_,
                )
                - power
            )
        elif self.test_type == "two-sided":
            lambda_ = n * f**2
            result = (
                ncf.sf(
                    f_dist.isf(alpha, k - 1, n - k),
                    1,
                    n - k,
                    lambda_,
                )
                - power
            )
        elif self.test_type == "greater":
            lambda_ = sqrt(n) * f
            result = nct.sf(t_dist.isf(alpha, n - k), n - k, lambda_) - power
        else:
            lambda_ = sqrt(n) * f
            result = nct.cdf(t_dist.ppf(alpha, n - k), n - k, lambda_) - power
        return float(result)

    def _get_effect_size(self, f: float, n: int, k: int, alpha: float, power: float) -> float:
        if self.test_type == "overall":
            lambda_ = n * f**2
            result = (
                ncf.sf(
                    f_dist.isf(alpha, k - 1, n - k),
                    k - 1,
                    n - k,
                    lambda_,
                )
                - power
            )
        elif self.test_type == "two-sided":
            lambda_ = n * f**2
            result = (
                ncf.sf(
                    f_dist.isf(alpha, k - 1, n - k),
                    1,
                    n - k,
                    lambda_,
                )
                - power
            )
        elif self.test_type == "greater":
            lambda_ = sqrt(n) * f
            result = nct.sf(t_dist.isf(alpha, n - k), n - k, lambda_) - power
        else:
            lambda_ = sqrt(n) * f
            result = nct.cdf(t_dist.ppf(alpha, n - k), n - k, lambda_) - power
        return float(result)

    def _get_alpha(self, alpha: float, n: int, f: float, k: int, power: float) -> float:
        if self.test_type == "overall":
            lambda_ = n * f**2
            result = (
                ncf.sf(
                    f_dist.isf(alpha, k - 1, n - k),
                    k - 1,
                    n - k,
                    lambda_,
                )
                - power
            )
        elif self.test_type == "two-sided":
            lambda_ = n * f**2
            result = (
                ncf.sf(
                    f_dist.isf(alpha, k - 1, n - k),
                    1,
                    n - k,
                    lambda_,
                )
                - power
            )
        elif self.test_type == "greater":
            lambda_ = sqrt(n) * f
            result = nct.sf(t_dist.isf(alpha, n - k), n - k, lambda_) - power
        else:
            lambda_ = sqrt(n) * f
            result = nct.cdf(t_dist.ppf(alpha, n - k), n - k, lambda_) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.n is None or self.f is None or self.k is None or self.alpha is None:
                raise ValueError("n, f, k, and alpha must be provided to compute power")
            self.power = self._get_power(self.n, self.f, self.k, self.alpha)
        elif self.k is None:
            if self.n is None or self.f is None or self.alpha is None or self.power is None:
                raise ValueError("n, f, alpha, and power must be provided to solve for k")
            n, f, alpha, power = self.n, self.f, self.alpha, self.power
            self.k = ceil(bisect(lambda k: self._get_groups(k, n, f, alpha, power), 2 + 1e-10, 100))
        elif self.n is None:
            if self.k is None or self.f is None or self.alpha is None or self.power is None:
                raise ValueError("k, f, alpha, and power must be provided to solve for n")
            k, f, alpha, power = self.k, self.f, self.alpha, self.power
            self.n = ceil(brentq(lambda n: self._get_sample_size(n, f, k, alpha, power), 2 + self.k + 1e-10, 1e05))
        elif self.f is None:
            if self.n is None or self.k is None or self.alpha is None or self.power is None:
                raise ValueError("n, k, alpha, and power must be provided to solve for f")
            n, k, alpha, power = self.n, self.k, self.alpha, self.power
            self.f = bisect(lambda f: self._get_effect_size(f, n, k, alpha, power), 1e-07, 1e07)
        else:
            if self.n is None or self.f is None or self.k is None or self.power is None:
                raise ValueError("n, f, k, and power must be provided to solve for alpha")
            n, f, k, power = self.n, self.f, self.k, self.power
            self.alpha = brentq(lambda alpha: self._get_alpha(alpha, n, f, k, power), 1e-10, 1 - 1e-10)
        return {
            "k": self.k,
            "n": self.n,
            "effect_size": self.f,
            "alpha": self.alpha,
            "power": self.power,
            "method": self.method,
            "note": self.note,
            "url": self.url,
        }


class WpAnovaBinaryClass:
    def __init__(
        self,
        k: int | None = None,
        n: int | None = None,
        V: float | None = None,
        alpha: float | None = None,
        power: float | None = None,
    ) -> None:
        self.k = k
        self.n = n
        self.V = V
        self.alpha = alpha
        self.power = power
        self.method = "One-way Analogous ANOVA with Binary Data"
        self.url = "http://psychstat.org/anovabinary"
        self.note = "n is the total sample size"

    def _get_power(self, V: float, n: int, k: int, alpha: float) -> float:
        chi = V**2 * n * (k - 1)
        df = k - 1
        crit_value = chi2.ppf(1 - alpha, df)
        power = ncx2.sf(crit_value, df, chi)
        return float(power)

    def _get_groups(self, k: float, V: float, n: int, alpha: float, power: float) -> float:
        chi = V**2 * n * (k - 1)
        df = k - 1
        crit_value = chi2.ppf(1 - alpha, df)
        result = ncx2.sf(crit_value, df, chi) - power
        return float(result)

    def _get_sample_size(self, n: float, V: float, k: int, alpha: float, power: float) -> float:
        chi = V**2 * n * (k - 1)
        df = k - 1
        crit_value = chi2.ppf(1 - alpha, df)
        result = ncx2.sf(crit_value, df, chi) - power
        return float(result)

    def _get_effect_size(self, V: float, n: int, k: int, alpha: float, power: float) -> float:
        chi = V**2 * n * (k - 1)
        df = k - 1
        crit_value = chi2.ppf(1 - alpha, df)
        result = ncx2.sf(crit_value, df, chi) - power
        return float(result)

    def _get_alpha(self, alpha: float, V: float, n: int, k: int, power: float) -> float:
        chi = V**2 * n * (k - 1)
        df = k - 1
        crit_value = chi2.ppf(1 - alpha, df)
        result = ncx2.sf(crit_value, df, chi) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.V is None or self.n is None or self.k is None or self.alpha is None:
                raise ValueError("V, n, k, and alpha must be provided to compute power")
            self.power = self._get_power(self.V, self.n, self.k, self.alpha)
        elif self.k is None:
            if self.V is None or self.n is None or self.alpha is None or self.power is None:
                raise ValueError("V, n, alpha, and power must be provided to solve for k")
            V, n, alpha, power = self.V, self.n, self.alpha, self.power
            self.k = ceil(bisect(lambda k: self._get_groups(k, V, n, alpha, power), 2 + 1e-10, 100))
        elif self.n is None:
            if self.V is None or self.k is None or self.alpha is None or self.power is None:
                raise ValueError("V, k, alpha, and power must be provided to solve for n")
            V, k, alpha, power = self.V, self.k, self.alpha, self.power
            self.n = ceil(brentq(lambda n: self._get_sample_size(n, V, k, alpha, power), 2 + self.k + 1e-10, 1e05))
        elif self.V is None:
            if self.n is None or self.k is None or self.alpha is None or self.power is None:
                raise ValueError("n, k, alpha, and power must be provided to solve for V")
            n, k, alpha, power = self.n, self.k, self.alpha, self.power
            self.V = bisect(lambda V: self._get_effect_size(V, n, k, alpha, power), 1e-07, 1e07)
        else:
            if self.V is None or self.n is None or self.k is None or self.power is None:
                raise ValueError("V, n, k, and power must be provided to solve for alpha")
            V, n, k, power = self.V, self.n, self.k, self.power
            self.alpha = brentq(lambda alpha: self._get_alpha(alpha, V, n, k, power), 1e-10, 1 - 1e-10)
        return {
            "k": self.k,
            "n": self.n,
            "effect_size": self.V,
            "alpha": self.alpha,
            "power": self.power,
            "method": self.method,
            "note": self.note,
            "url": self.url,
        }


class WpAnovaCountClass(WpAnovaBinaryClass):
    def __init__(
        self,
        k: int | None = None,
        n: int | None = None,
        V: float | None = None,
        alpha: float | None = None,
        power: float | None = None,
    ) -> None:
        super().__init__(k, n, V, alpha, power)
        self.method = "One-way Analogous ANOVA with Count Data"
        self.url = "http://psychstat.org/anovacount"
        self.note = "n is the total sample size"


class WpKAnovaClass:
    def __init__(
        self,
        n: int | None = None,
        ndf: int | None = None,
        f: float | None = None,
        ng: int | None = None,
        alpha: float | None = None,
        power: float | None = None,
    ) -> None:
        self.n = n
        self.ndf = ndf
        self.f = f
        self.ng = ng
        self.alpha = alpha
        self.power = power
        self.method = "Multiple way ANOVA analysis"
        self.url = "http://psychstat.org/kanova"
        self.note = "Sample size is the total sample size"

    def _get_power(self, f: float, n: int, ng: int, ndf: int, alpha: float) -> float:
        lambda_ = f**2 * n
        ddf = n - ng
        power = ncf.sf(f_dist.isf(alpha, ndf, ddf), ndf, ddf, lambda_)
        return float(power)

    def _get_sample_size(self, n: float, f: float, ng: int, ndf: int, alpha: float, power: float) -> float:
        lambda_ = f**2 * n
        ddf = n - ng
        result = ncf.sf(f_dist.isf(alpha, ndf, ddf), ndf, ddf, lambda_) - power
        return float(result)

    def _get_numerator_df(self, ndf: float, f: float, n: int, ng: int, alpha: float, power: float) -> float:
        lambda_ = f**2 * n
        ddf = n - ng
        result = ncf.sf(f_dist.isf(alpha, ndf, ddf), ndf, ddf, lambda_) - power
        return float(result)

    def _get_effect_size(self, f: float, n: int, ng: int, ndf: int, alpha: float, power: float) -> float:
        lambda_ = f**2 * n
        ddf = n - ng
        result = ncf.sf(f_dist.isf(alpha, ndf, ddf), ndf, ddf, lambda_) - power
        return float(result)

    def _get_groups(self, ng: float, f: float, n: int, ndf: int, alpha: float, power: float) -> float:
        lambda_ = f**2 * n
        ddf = n - ng
        result = ncf.sf(f_dist.isf(alpha, ndf, ddf), ndf, ddf, lambda_) - power
        return float(result)

    def _get_alpha(self, alpha: float, f: float, n: int, ng: int, ndf: int, power: float) -> float:
        lambda_ = f**2 * n
        ddf = n - ng
        result = ncf.sf(f_dist.isf(alpha, ndf, ddf), ndf, ddf, lambda_) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.f is None or self.n is None or self.ng is None or self.ndf is None or self.alpha is None:
                raise ValueError("f, n, ng, ndf, and alpha must be provided to compute power")
            self.power = self._get_power(self.f, self.n, self.ng, self.ndf, self.alpha)
        elif self.n is None:
            if self.f is None or self.ng is None or self.ndf is None or self.alpha is None or self.power is None:
                raise ValueError("f, ng, ndf, alpha, and power must be provided to solve for n")
            f, ng, ndf, alpha, power = self.f, self.ng, self.ndf, self.alpha, self.power
            self.n = ceil(brentq(lambda n: self._get_sample_size(n, f, ng, ndf, alpha, power), 1 + ng, 1e07))
        elif self.ndf is None:
            if self.f is None or self.n is None or self.ng is None or self.alpha is None or self.power is None:
                raise ValueError("f, n, ng, alpha, and power must be provided to solve for ndf")
            f, n, ng, alpha, power = self.f, self.n, self.ng, self.alpha, self.power
            self.ndf = ceil(bisect(lambda ndf: self._get_numerator_df(ndf, f, n, ng, alpha, power), 1 + 1e-10, 1e05))
        elif self.ng is None:
            if self.f is None or self.n is None or self.ndf is None or self.alpha is None or self.power is None:
                raise ValueError("f, n, ndf, alpha, and power must be provided to solve for ng")
            f, n, ndf, alpha, power = self.f, self.n, self.ndf, self.alpha, self.power
            self.ng = ceil(brentq(lambda ng: self._get_groups(ng, f, n, ndf, alpha, power), 1e-07, 1e07))
        elif self.f is None:
            if self.n is None or self.ng is None or self.ndf is None or self.alpha is None or self.power is None:
                raise ValueError("n, ng, ndf, alpha, and power must be provided to solve for f")
            n, ng, ndf, alpha, power = self.n, self.ng, self.ndf, self.alpha, self.power
            self.f = bisect(lambda f: self._get_effect_size(f, n, ng, ndf, alpha, power), 1e-07, 1e07)
        else:
            if self.f is None or self.n is None or self.ng is None or self.ndf is None or self.power is None:
                raise ValueError("f, n, ng, ndf, and power must be provided to solve for alpha")
            f, n, ng, ndf, power = self.f, self.n, self.ng, self.ndf, self.power
            self.alpha = brentq(lambda alpha: self._get_alpha(alpha, f, n, ng, ndf, power), 1e-10, 1 - 1e-10)
        n = self.n
        ng = self.ng
        if n is None or ng is None:
            raise ValueError("n and ng must be set after pwr_test")
        ddf = ceil(n - ng)
        return {
            "n": self.n,
            "ndf": self.ndf,
            "ddf": ddf,
            "effect_size": self.f,
            "ng": self.ng,
            "alpha": self.alpha,
            "power": self.power,
            "method": self.method,
            "note": self.note,
            "url": self.url,
        }


class WpRMAnovaClass:
    def __init__(
        self,
        n: int | None = None,
        ng: int | None = None,
        nm: int | None = None,
        f: float | None = None,
        nscor: float = 1,
        alpha: float | None = None,
        power: float | None = None,
        test_type: str = "between",
    ) -> None:
        self.n = n
        self.ng = ng
        self.nm = nm
        self.f = f
        self.nscor = nscor
        self.alpha = alpha
        self.power = power
        self.test_type = test_type
        self.method = "Repeated-measures ANOVA analysis"
        self.url = "http://psychstat.org/rmanova"
        if self.test_type == "between":
            self.note = "Power analysis for between-effect test"
        elif self.test_type == "within":
            self.note = "Power analysis for within-effect test"
        else:
            self.note = "Power analysis for interaction-effect test"

    def _get_power(self, f: float, n: int, ng: int, nm: int, alpha: float) -> float:
        if self.test_type == "between":
            df_1 = ng - 1
            df_2 = n - ng
        elif self.test_type == "within":
            df_1 = (nm - 1) * self.nscor
            df_2 = (n - ng) * df_1
        else:
            df_1 = (ng - 1) * (nm - 1) * self.nscor
            df_2 = (n - ng) * (nm - 1) * self.nscor
        lambda_ = f**2 * n * self.nscor
        power = ncf.sf(f_dist.isf(alpha, df_1, df_2), df_1, df_2, lambda_)
        return float(power)

    def _get_groups(self, ng: float, f: float, n: int, nm: int, alpha: float, power: float) -> float:
        if self.test_type == "between":
            df_1 = ng - 1
            df_2 = n - ng
        elif self.test_type == "within":
            df_1 = (nm - 1) * self.nscor
            df_2 = (n - ng) * df_1
        else:
            df_1 = (ng - 1) * (nm - 1) * self.nscor
            df_2 = (n - ng) * (nm - 1) * self.nscor
        lambda_ = f**2 * n * self.nscor
        result = ncf.sf(f_dist.isf(alpha, df_1, df_2), df_1, df_2, lambda_) - power
        return float(result)

    def _get_nm(self, nm: float, f: float, n: int, ng: int, alpha: float, power: float) -> float:
        if self.test_type == "between":
            raise ValueError("nm is not defined for between-effects")
        elif self.test_type == "within":
            df_1 = (nm - 1) * self.nscor
            df_2 = (n - ng) * df_1
        else:
            df_1 = (ng - 1) * (nm - 1) * self.nscor
            df_2 = (n - ng) * (nm - 1) * self.nscor
        lambda_ = f**2 * n * self.nscor
        result = ncf.sf(f_dist.isf(alpha, df_1, df_2), df_1, df_2, lambda_) - power
        return float(result)

    def _get_sample_size(self, n: float, f: float, ng: int, nm: int, alpha: float, power: float) -> float:
        if self.test_type == "between":
            df_1 = ng - 1
            df_2 = n - ng
        elif self.test_type == "within":
            df_1 = (nm - 1) * self.nscor
            df_2 = (n - ng) * df_1
        else:
            df_1 = (ng - 1) * (nm - 1) * self.nscor
            df_2 = (n - ng) * (nm - 1) * self.nscor
        lambda_ = f**2 * n * self.nscor
        result = ncf.sf(f_dist.isf(alpha, df_1, df_2), df_1, df_2, lambda_) - power
        return float(result)

    def _get_effect_size(self, f: float, n: int, ng: int, nm: int, alpha: float, power: float) -> float:
        if self.test_type == "between":
            df_1 = ng - 1
            df_2 = n - ng
        elif self.test_type == "within":
            df_1 = (nm - 1) * self.nscor
            df_2 = (n - ng) * df_1
        else:
            df_1 = (ng - 1) * (nm - 1) * self.nscor
            df_2 = (n - ng) * (nm - 1) * self.nscor
        lambda_ = f**2 * n * self.nscor
        result = ncf.sf(f_dist.isf(alpha, df_1, df_2), df_1, df_2, lambda_) - power
        return float(result)

    def _get_alpha(self, alpha: float, f: float, n: int, ng: int, nm: int, power: float) -> float:
        if self.test_type == "between":
            df_1 = ng - 1
            df_2 = n - ng
        elif self.test_type == "within":
            df_1 = (nm - 1) * self.nscor
            df_2 = (n - ng) * df_1
        else:
            df_1 = (ng - 1) * (nm - 1) * self.nscor
            df_2 = (n - ng) * (nm - 1) * self.nscor
        lambda_ = f**2 * n * self.nscor
        result = ncf.sf(f_dist.isf(alpha, df_1, df_2), df_1, df_2, lambda_) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.f is None or self.n is None or self.ng is None or self.nm is None or self.alpha is None:
                raise ValueError("f, n, ng, nm, and alpha must be provided to compute power")
            self.power = self._get_power(self.f, self.n, self.ng, self.nm, self.alpha)
        elif self.n is None:
            if self.f is None or self.ng is None or self.nm is None or self.alpha is None or self.power is None:
                raise ValueError("f, ng, nm, alpha, and power must be provided to solve for n")
            f, ng, nm, alpha, power = self.f, self.ng, self.nm, self.alpha, self.power
            self.n = ceil(brentq(lambda n: self._get_sample_size(n, f, ng, nm, alpha, power), 5, 1e07))
        elif self.nm is None:
            if self.f is None or self.n is None or self.ng is None or self.alpha is None or self.power is None:
                raise ValueError("f, n, ng, alpha, and power must be provided to solve for nm")
            f, n, ng, alpha, power = self.f, self.n, self.ng, self.alpha, self.power
            self.nm = ceil(bisect(lambda nm: self._get_nm(nm, f, n, ng, alpha, power), 1 + 1e-10, 1e05))
        elif self.ng is None:
            if self.f is None or self.n is None or self.nm is None or self.alpha is None or self.power is None:
                raise ValueError("f, n, nm, alpha, and power must be provided to solve for ng")
            f, n, nm, alpha, power = self.f, self.n, self.nm, self.alpha, self.power
            self.ng = ceil(bisect(lambda ng: self._get_groups(ng, f, n, nm, alpha, power), 1 + 1e-10, 1e05))
        elif self.f is None:
            if self.n is None or self.ng is None or self.nm is None or self.alpha is None or self.power is None:
                raise ValueError("n, ng, nm, alpha, and power must be provided to solve for f")
            n, ng, nm, alpha, power = self.n, self.ng, self.nm, self.alpha, self.power
            self.f = bisect(lambda f: self._get_effect_size(f, n, ng, nm, alpha, power), 1e-07, 1e07)
        else:
            if self.f is None or self.n is None or self.ng is None or self.nm is None or self.power is None:
                raise ValueError("f, n, ng, nm, and power must be provided to solve for alpha")
            f, n, ng, nm, power = self.f, self.n, self.ng, self.nm, self.power
            self.alpha = brentq(lambda alpha: self._get_alpha(alpha, f, n, ng, nm, power), 1e-10, 1 - 1e-10)
        return {
            "n": self.n,
            "nm": self.nm,
            "effect_size": self.f,
            "nscor": self.nscor,
            "ng": self.ng,
            "alpha": self.alpha,
            "power": self.power,
            "method": self.method,
            "note": self.note,
            "url": self.url,
        }
