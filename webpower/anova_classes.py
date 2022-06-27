from math import ceil, sqrt, pow
from typing import Dict, Optional

from scipy.stats import chi2, ncx2, ncf, nct, f as f_dist, t as t_dist
from scipy.optimize import brentq, bisect


class WpAnovaClass:
    def __init__(
        self,
        k: Optional[int] = None,
        n: Optional[int] = None,
        f: Optional[float] = None,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
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

    def _get_power(self) -> float:
        if self.test_type == "overall":
            lambda_ = self.n * pow(self.f, 2)
            power = ncf.sf(
                f_dist.isf(self.alpha, self.k - 1, self.n - self.k),
                self.k - 1,
                self.n - self.k,
                lambda_,
            )
        elif self.test_type == "two-sided":
            lambda_ = self.n * pow(self.f, 2)
            power = ncf.sf(
                f_dist.isf(self.alpha, self.k - 1, self.n - self.k),
                1,
                self.n - self.k,
                lambda_,
            )
        elif self.test_type == "greater":
            lambda_ = sqrt(self.n) * self.f
            power = nct.sf(
                t_dist.isf(self.alpha, self.n - self.k), self.n - self.k, lambda_
            )
        else:
            lambda_ = sqrt(self.n) * self.f
            power = nct.cdf(
                t_dist.ppf(self.alpha, self.n - self.k), self.n - self.k, lambda_
            )
        return power

    def _get_groups(self, k) -> float:
        if self.test_type == "overall":
            lambda_ = self.n * pow(self.f, 2)
            k = (
                ncf.sf(
                    f_dist.isf(self.alpha, k - 1, self.n - k),
                    k - 1,
                    self.n - k,
                    lambda_,
                )
                - self.power
            )
        elif self.test_type == "two-sided":
            lambda_ = self.n * pow(self.f, 2)
            k = (
                ncf.sf(
                    f_dist.isf(self.alpha, k - 1, self.n - k),
                    1,
                    self.n - self.k,
                    lambda_,
                )
                - self.power
            )
        elif self.test_type == "greater":
            lambda_ = sqrt(self.n) * self.f
            k = (
                nct.sf(t_dist.isf(self.alpha, self.n - k), self.n - k, lambda_)
                - self.power
            )
        else:
            lambda_ = sqrt(self.n) * self.f
            k = (
                nct.cdf(t_dist.ppf(self.alpha, self.n - k), self.n - k, lambda_)
                - self.power
            )
        return k

    def _get_sample_size(self, n) -> float:
        if self.test_type == "overall":
            lambda_ = n * pow(self.f, 2)
            n = (
                ncf.sf(
                    f_dist.isf(self.alpha, self.k - 1, n - self.k),
                    self.k - 1,
                    n - self.k,
                    lambda_,
                )
                - self.power
            )
        elif self.test_type == "two-sided":
            lambda_ = n * pow(self.f, 2)
            n = (
                ncf.sf(
                    f_dist.isf(self.alpha, self.k - 1, n - self.k),
                    1,
                    n - self.k,
                    lambda_,
                )
                - self.power
            )
        elif self.test_type == "greater":
            lambda_ = sqrt(n) * self.f
            n = (
                nct.sf(t_dist.isf(self.alpha, n - self.k), n - self.k, lambda_)
                - self.power
            )
        else:
            lambda_ = sqrt(n) * self.f
            n = (
                nct.cdf(t_dist.ppf(self.alpha, n - self.k), n - self.k, lambda_)
                - self.power
            )
        return n

    def _get_effect_size(self, f) -> float:
        if self.test_type == "overall":
            lambda_ = self.n * pow(f, 2)
            f = (
                ncf.sf(
                    f_dist.isf(self.alpha, self.k - 1, self.n - self.k),
                    self.k - 1,
                    self.n - self.k,
                    lambda_,
                )
                - self.power
            )
        elif self.test_type == "two-sided":
            lambda_ = self.n * pow(f, 2)
            f = (
                ncf.sf(
                    f_dist.isf(self.alpha, self.k - 1, self.n - self.k),
                    1,
                    self.n - self.k,
                    lambda_,
                )
                - self.power
            )
        elif self.test_type == "greater":
            lambda_ = sqrt(self.n) * f
            f = (
                nct.sf(
                    t_dist.isf(self.alpha, self.n - self.k), self.n - self.k, lambda_
                )
                - self.power
            )
        else:
            lambda_ = sqrt(self.n) * f
            f = (
                nct.cdf(
                    t_dist.ppf(self.alpha, self.n - self.k), self.n - self.k, lambda_
                )
                - self.power
            )
        return f

    def _get_alpha(self, alpha) -> float:
        if self.test_type == "overall":
            lambda_ = self.n * pow(self.f, 2)
            alpha = (
                ncf.sf(
                    f_dist.isf(alpha, self.k - 1, self.n - self.k),
                    self.k - 1,
                    self.n - self.k,
                    lambda_,
                )
                - self.power
            )
        elif self.test_type == "two-sided":
            lambda_ = self.n * pow(self.f, 2)
            alpha = (
                ncf.sf(
                    f_dist.isf(alpha, self.k - 1, self.n - self.k),
                    1,
                    self.n - self.k,
                    lambda_,
                )
                - self.power
            )
        elif self.test_type == "greater":
            lambda_ = sqrt(self.n) * self.f
            alpha = (
                nct.sf(t_dist.isf(alpha, self.n - self.k), self.n - self.k, lambda_)
                - self.power
            )
        else:
            lambda_ = sqrt(self.n) * self.f
            alpha = (
                nct.cdf(t_dist.ppf(alpha, self.n - self.k), self.n - self.k, lambda_)
                - self.power
            )
        return alpha

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.k is None:
            self.k = ceil(bisect(self._get_groups, 2 + 1e-10, 100))
        elif self.n is None:
            self.n = ceil(brentq(self._get_sample_size, 2 + self.k + 1e-10, 1e05))
        elif self.f is None:
            self.f = bisect(self._get_effect_size, 1e-07, 1e07)
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
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
        k: Optional[int] = None,
        n: Optional[int] = None,
        V: Optional[float] = None,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
    ) -> None:
        self.k = k
        self.n = n
        self.V = V
        self.alpha = alpha
        self.power = power
        self.method = "One-way Analogous ANOVA with Binary Data"
        self.url = "http://psychstat.org/anovabinary"
        self.note = "n is the total sample size"

    def _get_power(self) -> float:
        chi = pow(self.V, 2) * self.n * (self.k - 1)
        df = self.k - 1
        crit_value = chi2.ppf(1 - self.alpha, df)
        power = 1 - ncx2.cdf(crit_value, df, chi)
        return power

    def _get_groups(self, k) -> float:
        chi = pow(self.V, 2) * self.n * (k - 1)
        df = k - 1
        crit_value = chi2.ppf(1 - self.alpha, df)
        k = 1 - ncx2.cdf(crit_value, df, chi) - self.power
        return k

    def _get_sample_size(self, n) -> float:
        chi = pow(self.V, 2) * n * (self.k - 1)
        df = self.k - 1
        crit_value = chi2.ppf(1 - self.alpha, df)
        n = 1 - ncx2.cdf(crit_value, df, chi) - self.power
        return n

    def _get_effect_size(self, V) -> float:
        chi = pow(V, 2) * self.n * (self.k - 1)
        df = self.k - 1
        crit_value = chi2.ppf(1 - self.alpha, df)
        V = 1 - ncx2.cdf(crit_value, df, chi) - self.power
        return V

    def _get_alpha(self, alpha) -> float:
        chi = pow(self.V, 2) * self.n * (self.k - 1)
        df = self.k - 1
        crit_value = chi2.ppf(1 - alpha, df)
        alpha = 1 - ncx2.cdf(crit_value, df, chi) - self.power
        return alpha

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.k is None:
            self.k = ceil(bisect(self._get_groups, 2 + 1e-10, 100))
        elif self.n is None:
            self.n = ceil(brentq(self._get_sample_size, 2 + self.k + 1e-10, 1e05))
        elif self.V is None:
            self.V = bisect(self._get_effect_size, 1e-07, 1e07)
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
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
        k: Optional[int] = None,
        n: Optional[int] = None,
        V: Optional[float] = None,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
    ) -> None:
        super().__init__(k, n, V, alpha, power)
        self.method = "One-way Analogous ANOVA with Count Data"
        self.url = "http://psychstat.org/anovacount"
        self.note = "n is the total sample size"


class WpKAnovaClass:
    def __init__(
        self,
        n: Optional[int] = None,
        ndf: Optional[int] = None,
        f: Optional[float] = None,
        ng: Optional[int] = None,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
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

    def _get_power(self) -> float:
        lambda_ = pow(self.f, 2) * self.n
        ddf = self.n - self.ng
        power = ncf.sf(f_dist.isf(self.alpha, self.ndf, ddf), self.ndf, ddf, lambda_)
        return power

    def _get_sample_size(self, n) -> float:
        lambda_ = pow(self.f, 2) * n
        ddf = n - self.ng
        n = (
            ncf.sf(f_dist.isf(self.alpha, self.ndf, ddf), self.ndf, ddf, lambda_)
            - self.power
        )
        return n

    def _get_numerator_df(self, ndf) -> float:
        lambda_ = pow(self.f, 2) * self.n
        ddf = self.n - self.ng
        ndf = ncf.sf(f_dist.isf(self.alpha, ndf, ddf), ndf, ddf, lambda_) - self.power
        return ndf

    def _get_effect_size(self, f) -> float:
        lambda_ = pow(f, 2) * self.n
        ddf = self.n - self.ng
        f = (
            ncf.sf(f_dist.isf(self.alpha, self.ndf, ddf), self.ndf, ddf, lambda_)
            - self.power
        )
        return f

    def _get_groups(self, ng) -> float:
        lambda_ = pow(self.f, 2) * self.n
        ddf = self.n - ng
        ng = (
            ncf.sf(f_dist.isf(self.alpha, self.ndf, ddf), self.ndf, ddf, lambda_)
            - self.power
        )
        return ng

    def _get_alpha(self, alpha) -> float:
        lambda_ = pow(self.f, 2) * self.n
        ddf = self.n - self.ng
        alpha = (
            ncf.sf(f_dist.isf(alpha, self.ndf, ddf), self.ndf, ddf, lambda_)
            - self.power
        )
        return alpha

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.n is None:
            self.n = ceil(brentq(self._get_sample_size, 1 + self.ng, 1e07))
        elif self.ndf is None:
            self.ndf = ceil(bisect(self._get_numerator_df, 1 + 1e-10, 1e05))
        elif self.ng is None:
            self.ng = ceil(brentq(self._get_groups, 1e-07, 1e07))
        elif self.f is None:
            self.f = bisect(self._get_effect_size, 1e-07, 1e07)
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
        ddf = ceil(self.n - self.ng)
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
        n: Optional[int] = None,
        ng: Optional[int] = None,
        nm: Optional[int] = None,
        f: Optional[float] = None,
        nscor: float = 1,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
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

    def _get_power(self) -> float:
        if self.test_type == "between":
            df_1 = self.ng - 1
            df_2 = self.n - self.ng
        elif self.test_type == "within":
            df_1 = (self.nm - 1) * self.nscor
            df_2 = (self.n - self.ng) * df_1
        else:
            df_1 = (self.ng - 1) * (self.nm - 1) * self.nscor
            df_2 = (self.n - self.ng) * (self.nm - 1) * self.nscor
        lambda_ = pow(self.f, 2) * self.n * self.nscor
        power = ncf.sf(f_dist.isf(self.alpha, df_1, df_2), df_1, df_2, lambda_)
        return power

    def _get_groups(self, ng) -> float:
        if self.test_type == "between":
            df_1 = ng - 1
            df_2 = self.n - ng
        elif self.test_type == "within":
            df_1 = (self.nm - 1) * self.nscor
            df_2 = (self.n - ng) * df_1
        else:
            df_1 = (ng - 1) * (self.nm - 1) * self.nscor
            df_2 = (self.n - ng) * (self.nm - 1) * self.nscor
        lambda_ = pow(self.f, 2) * self.n * self.nscor
        ng = (
            ncf.sf(f_dist.isf(self.alpha, df_1, df_2), df_1, df_2, lambda_) - self.power
        )
        return ng

    def _get_nm(self, nm):
        if self.test_type == "between":
            raise ValueError("nm is not defined for between-effects")
        elif self.test_type == "within":
            df_1 = (nm - 1) * self.nscor
            df_2 = (self.n - self.ng) * df_1
        else:
            df_1 = (self.ng - 1) * (nm - 1) * self.nscor
            df_2 = (self.n - self.ng) * (nm - 1) * self.nscor
        lambda_ = pow(self.f, 2) * self.n * self.nscor
        nm = (
            ncf.sf(f_dist.isf(self.alpha, df_1, df_2), df_1, df_2, lambda_) - self.power
        )
        return nm

    def _get_sample_size(self, n):
        if self.test_type == "between":
            df_1 = self.ng - 1
            df_2 = n - self.ng
        elif self.test_type == "within":
            df_1 = (self.nm - 1) * self.nscor
            df_2 = (n - self.ng) * df_1
        else:
            df_1 = (self.ng - 1) * (self.nm - 1) * self.nscor
            df_2 = (n - self.ng) * (self.nm - 1) * self.nscor
        lambda_ = pow(self.f, 2) * n * self.nscor
        n = ncf.sf(f_dist.isf(self.alpha, df_1, df_2), df_1, df_2, lambda_) - self.power
        return n

    def _get_effect_size(self, f):
        if self.test_type == "between":
            df_1 = self.ng - 1
            df_2 = self.n - self.ng
        elif self.test_type == "within":
            df_1 = (self.nm - 1) * self.nscor
            df_2 = (self.n - self.ng) * df_1
        else:
            df_1 = (self.ng - 1) * (self.nm - 1) * self.nscor
            df_2 = (self.n - self.ng) * (self.nm - 1) * self.nscor
        lambda_ = pow(f, 2) * self.n * self.nscor
        f = ncf.sf(f_dist.isf(self.alpha, df_1, df_2), df_1, df_2, lambda_) - self.power
        return f

    def _get_alpha(self, alpha):
        if self.test_type == "between":
            df_1 = self.ng - 1
            df_2 = self.n - self.ng
        elif self.test_type == "within":
            df_1 = (self.nm - 1) * self.nscor
            df_2 = (self.n - self.ng) * df_1
        else:
            df_1 = (self.ng - 1) * (self.nm - 1) * self.nscor
            df_2 = (self.n - self.ng) * (self.nm - 1) * self.nscor
        lambda_ = pow(self.f, 2) * self.n * self.nscor
        alpha = ncf.sf(f_dist.isf(alpha, df_1, df_2), df_1, df_2, lambda_) - self.power
        return alpha

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.n is None:
            self.n = ceil(brentq(self._get_sample_size, 5, 1e07))
        elif self.nm is None:
            self.nm = ceil(bisect(self._get_nm, 1 + 1e-10, 1e05))
        elif self.ng is None:
            self.ng = ceil(bisect(self._get_groups, 1 + 1e-10, 1e05))
        elif self.f is None:
            self.f = bisect(self._get_effect_size, 1e-07, 1e07)
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
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
