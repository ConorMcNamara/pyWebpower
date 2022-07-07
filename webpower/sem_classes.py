from math import ceil
from typing import Dict, Optional

from scipy.stats import ncx2, chi2
from scipy.optimize import brentq


class WPSEMChisq:
    def __init__(
        self,
        n: Optional[int] = None,
        df: Optional[int] = None,
        effect: Optional[float] = None,
        alpha: Optional[float] = None,
        power: Optional[float] = None,
    ) -> None:
        self.n = n
        self.df = df
        self.effect = effect
        self.power = power
        self.alpha = alpha
        self.method = "Power for SEM (Satorra & Saris, 1985)"
        self.url = "http://psychstat.org/semchisq"

    def _get_power(self) -> float:
        ncp = (self.n - 1) * self.effect
        c_alpha = chi2.ppf(1 - self.alpha, self.df)
        power = ncx2.sf(c_alpha, self.df, ncp)
        return power

    def _get_n(self, n: int) -> float:
        ncp = (n - 1) * self.effect
        c_alpha = chi2.ppf(1 - self.alpha, self.df)
        n = ncx2.sf(c_alpha, self.df, ncp) - self.power
        return n

    def _get_df(self, df: int) -> float:
        ncp = (self.n - 1) * self.effect
        c_alpha = chi2.ppf(1 - self.alpha, df)
        df = ncx2.sf(c_alpha, df, ncp) - self.power
        return df

    def _get_alpha(self, alpha: float) -> float:
        ncp = (self.n - 1) * self.effect
        c_alpha = chi2.ppf(1 - alpha, self.df)
        alpha = ncx2.sf(c_alpha, self.df, ncp) - self.power
        return alpha

    def _get_effect_size(self, effect: float) -> float:
        ncp = (self.n - 1) * effect
        c_alpha = chi2.ppf(1 - self.alpha, self.df)
        effect = ncx2.sf(c_alpha, self.df, ncp) - self.power
        return effect

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.effect is None:
            self.effect = brentq(self._get_effect_size, 0, 1)
        elif self.n is None:
            self.n = ceil(brentq(self._get_n, 10 + 1e-10, 1e09))
        elif self.df is None:
            self.df = ceil(brentq(self._get_df, 1, 1e04))
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
        return {
            "n": self.n,
            "df": self.df,
            "effect_size": self.effect,
            "alpha": self.alpha,
            "power": self.power,
            "method": self.method,
            "url": self.url,
        }


class WPSEMRMSEA:
    def __init__(
            self,
            n: Optional[int] = None,
            df: Optional[int] = None,
            rmsea0: Optional[float] = None,
            rmsea1: Optional[float] = None,
            power: Optional[float] = None,
            alpha: Optional[float] = None,
            test_type: str = "close",
    ) -> None:
        self.n = n
        self.df = df
        self.rmsea0 = rmsea0
        self.rmsea1 = rmsea1
        self.power = power
        self.alpha = alpha
        self.test_type = test_type.casefold()
        self.method = "Power for SEM based on RMSEA"
        self.url = "http://psychstat.org/rmsea"

    def _get_power(self) -> float:
        ncp0 = (self.n - 1) * self.df * pow(self.rmsea0, 2)
        ncp1 = (self.n - 1) * self.df * pow(self.rmsea1, 2)
        if self.test_type == "close":
            c_alpha = ncx2.ppf(1 - self.alpha, self.df, ncp0)
        else:
            c_alpha = ncx2.ppf(self.alpha, self.df, ncp0)
        power = ncx2.sf(c_alpha, self.df, ncp1)
        return power

    def _get_n(self, n: int) -> float:
        ncp0 = (n - 1) * self.df * pow(self.rmsea0, 2)
        ncp1 = (n - 1) * self.df * pow(self.rmsea1, 2)
        if self.test_type == "close":
            c_alpha = ncx2.ppf(1 - self.alpha, self.df, ncp0)
        else:
            c_alpha = ncx2.ppf(self.alpha, self.df, ncp0)
        n = ncx2.sf(c_alpha, self.df, ncp1) - self.power
        return n

    def _get_df(self, df: int) -> float:
        ncp0 = (self.n - 1) * df * pow(self.rmsea0, 2)
        ncp1 = (self.n - 1) * df * pow(self.rmsea1, 2)
        if self.test_type == "close":
            c_alpha = ncx2.ppf(1 - self.alpha, df, ncp0)
        else:
            c_alpha = ncx2.ppf(self.alpha, df, ncp0)
        df = ncx2.sf(c_alpha, df, ncp1) - self.power
        return df

    def _get_rmsea0(self, rmsea0: float) -> float:
        ncp0 = (self.n - 1) * self.df * pow(rmsea0, 2)
        ncp1 = (self.n - 1) * self.df * pow(self.rmsea1, 2)
        if self.test_type == "close":
            c_alpha = ncx2.ppf(1 - self.alpha, self.df, ncp0)
        else:
            c_alpha = ncx2.ppf(self.alpha, self.df, ncp0)
        rmsea0 = ncx2.sf(c_alpha, self.df, ncp1) - self.power
        return rmsea0

    def _get_rmsea1(self, rmsea1: float) -> float:
        ncp0 = (self.n - 1) * self.df * pow(self.rmsea0, 2)
        ncp1 = (self.n - 1) * self.df * pow(rmsea1, 2)
        if self.test_type == "close":
            c_alpha = ncx2.ppf(1 - self.alpha, self.df, ncp0)
        else:
            c_alpha = ncx2.ppf(self.alpha, self.df, ncp0)
        rmsea1 = ncx2.sf(c_alpha, self.df, ncp1) - self.power
        return rmsea1

    def _get_alpha(self, alpha: float) -> float:
        ncp0 = (self.n - 1) * self.df * pow(self.rmsea0, 2)
        ncp1 = (self.n - 1) * self.df * pow(self.rmsea1, 2)
        if self.test_type == "close":
            c_alpha = ncx2.ppf(1 - alpha, self.df, ncp0)
        else:
            c_alpha = ncx2.ppf(alpha, self.df, ncp0)
        alpha = ncx2.sf(c_alpha, self.df, ncp1) - self.power
        return alpha

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.rmsea0 is None:
            self.rmsea0 = brentq(self._get_rmsea0, 0, 1)
        elif self.rmsea1 is None:
            self.rmsea1 = brentq(self._get_rmsea1, 0, 1)
        elif self.n is None:
            self.n = ceil(brentq(self._get_n, 2 + 1e-10, 1e09))
        elif self.df is None:
            self.df = ceil(brentq(self._get_df, 1, 1e04))
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
        return {
            "n": self.n,
            "df": self.df,
            "rmsea0": self.rmsea0,
            "rmsea1": self.rmsea1,
            "alpha": self.alpha,
            "power": self.power,
            "method": self.method,
            "url": self.url,
        }
