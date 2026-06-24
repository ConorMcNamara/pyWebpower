"""Power-analysis classes for structural equation modeling (SEM)."""

from math import ceil

from scipy.stats import chi2, ncx2

from webpower.utils import brentq


class WPSEMChisq:
    """Power analysis for structural equation modeling (SEM) based on the chi-squared test.

    Structural equation modeling (SEM) is a multivariate technique used to analyze relationships among observed and
    latent variables. This class implements SEM power analysis based on the likelihood ratio test proposed by Satorra
    and Saris (1985).

    Parameters
    ----------
    n : int, default=None
        Sample size
    df : int, default=None
        The degrees of freedom for our test, based on the Chi-Squared Test.
    effect : float, default=None
        Effect size. It specifies the population misfit of a SEM model, which is the difference between two SEM models:
        a full model (Mf) and a reduced model (Mr). A convienient way to get the effect size is to fit the reduced model
        using SEM software such R package 'lavaan' (Rossel, 2012). Then the effect size is calculated as the
        chi-squared statistics dividing by the sample size.
    alpha : float, default=None
        Significance level chosen for the test.
    power : float, default=None
        Statistical power.
    """

    def __init__(
        self,
        n: int | None = None,
        df: int | None = None,
        effect: float | None = None,
        alpha: float | None = None,
        power: float | None = None,
    ) -> None:
        self.n = n
        self.df = df
        self.effect = effect
        self.power = power
        self.alpha = alpha
        self.method = "Power for SEM (Satorra & Saris, 1985)"
        self.url = "http://psychstat.org/semchisq"

    def _get_power(self, n: int, effect: float, df: int, alpha: float) -> float:
        ncp = (n - 1) * effect
        c_alpha = chi2.ppf(1 - alpha, df)
        power = ncx2.sf(c_alpha, df, ncp)
        return float(power)

    def _get_n(self, n: float, effect: float, df: int, alpha: float, power: float) -> float:
        ncp = (n - 1) * effect
        c_alpha = chi2.ppf(1 - alpha, df)
        result = ncx2.sf(c_alpha, df, ncp) - power
        return float(result)

    def _get_df(self, df: float, n: int, effect: float, alpha: float, power: float) -> float:
        ncp = (n - 1) * effect
        c_alpha = chi2.ppf(1 - alpha, df)
        result = ncx2.sf(c_alpha, df, ncp) - power
        return float(result)

    def _get_alpha(self, alpha: float, n: int, effect: float, df: int, power: float) -> float:
        ncp = (n - 1) * effect
        c_alpha = chi2.ppf(1 - alpha, df)
        result = ncx2.sf(c_alpha, df, ncp) - power
        return float(result)

    def _get_effect_size(self, effect: float, n: int, df: int, alpha: float, power: float) -> float:
        ncp = (n - 1) * effect
        c_alpha = chi2.ppf(1 - alpha, df)
        result = ncx2.sf(c_alpha, df, ncp) - power
        return float(result)

    def pwr_test(self) -> dict:
        """Solve for the unspecified parameter and return the power-analysis results.

        Returns
        -------
        A dictionary containing n, df, effect_size, alpha, power, and the method and url metadata of the test.
        """
        if self.power is None:
            if self.n is None or self.effect is None or self.df is None or self.alpha is None:
                raise ValueError("n, effect, df, and alpha must be provided to compute power")
            self.power = self._get_power(self.n, self.effect, self.df, self.alpha)
        elif self.effect is None:
            if self.n is None or self.df is None or self.alpha is None or self.power is None:
                raise ValueError("n, df, alpha, and power must be provided to solve for effect")
            n, df, alpha, power = self.n, self.df, self.alpha, self.power
            self.effect = brentq(lambda effect: self._get_effect_size(effect, n, df, alpha, power), 0, 1)
        elif self.n is None:
            if self.effect is None or self.df is None or self.alpha is None or self.power is None:
                raise ValueError("effect, df, alpha, and power must be provided to solve for n")
            effect, df, alpha, power = self.effect, self.df, self.alpha, self.power
            self.n = ceil(brentq(lambda n: self._get_n(n, effect, df, alpha, power), 10 + 1e-10, 1e09))
        elif self.df is None:
            if self.n is None or self.effect is None or self.alpha is None or self.power is None:
                raise ValueError("n, effect, alpha, and power must be provided to solve for df")
            n, effect, alpha, power = self.n, self.effect, self.alpha, self.power
            self.df = ceil(brentq(lambda df: self._get_df(df, n, effect, alpha, power), 1, 1e04))
        else:
            if self.n is None or self.effect is None or self.df is None or self.power is None:
                raise ValueError("n, effect, df, and power must be provided to solve for alpha")
            n, effect, df, power = self.n, self.effect, self.df, self.power
            self.alpha = brentq(lambda alpha: self._get_alpha(alpha, n, effect, df, power), 1e-10, 1 - 1e-10)
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
    """Power analysis for structural equation modeling (SEM) based on RMSEA.

    Structural equation modeling (SEM) is a multivariate technique used to analyze relationships among observed and
    latent variables. This class implements SEM power analysis based on RMSEA proposed by MacCallum et al. (1996).

    Parameters
    ----------
    n : int, default=None
        Sample size
    df : int, default=None
        The degrees of freedom for our test, based on the Chi-Squared Test.
    rmsea0 : float, default=None
        The RMSE for H0. It usually equals 0.
    rmsea1 : float, default=None
        The RMSE for H1
    power : float, default=None
        Statistical power.
    alpha : float, default=None
        Significance level chosen for the test.
    test_type : {'close', 'notclose'}, default='close'
        Close fit or not-close fit.
    """

    def __init__(
        self,
        n: int | None = None,
        df: int | None = None,
        rmsea0: float | None = None,
        rmsea1: float | None = None,
        power: float | None = None,
        alpha: float | None = None,
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

    def _get_power(self, n: int, df: int, rmsea0: float, rmsea1: float, alpha: float) -> float:
        ncp0 = (n - 1) * df * rmsea0**2
        ncp1 = (n - 1) * df * rmsea1**2
        if self.test_type == "close":
            c_alpha = ncx2.ppf(1 - alpha, df, ncp0)
        else:
            c_alpha = ncx2.ppf(alpha, df, ncp0)
        power = ncx2.sf(c_alpha, df, ncp1)
        return float(power)

    def _get_n(self, n: float, df: int, rmsea0: float, rmsea1: float, alpha: float, power: float) -> float:
        ncp0 = (n - 1) * df * rmsea0**2
        ncp1 = (n - 1) * df * rmsea1**2
        if self.test_type == "close":
            c_alpha = ncx2.ppf(1 - alpha, df, ncp0)
        else:
            c_alpha = ncx2.ppf(alpha, df, ncp0)
        result = ncx2.sf(c_alpha, df, ncp1) - power
        return float(result)

    def _get_df(self, df: float, n: int, rmsea0: float, rmsea1: float, alpha: float, power: float) -> float:
        ncp0 = (n - 1) * df * rmsea0**2
        ncp1 = (n - 1) * df * rmsea1**2
        if self.test_type == "close":
            c_alpha = ncx2.ppf(1 - alpha, df, ncp0)
        else:
            c_alpha = ncx2.ppf(alpha, df, ncp0)
        result = ncx2.sf(c_alpha, df, ncp1) - power
        return float(result)

    def _get_rmsea0(self, rmsea0: float, n: int, df: int, rmsea1: float, alpha: float, power: float) -> float:
        ncp0 = (n - 1) * df * rmsea0**2
        ncp1 = (n - 1) * df * rmsea1**2
        if self.test_type == "close":
            c_alpha = ncx2.ppf(1 - alpha, df, ncp0)
        else:
            c_alpha = ncx2.ppf(alpha, df, ncp0)
        result = ncx2.sf(c_alpha, df, ncp1) - power
        return float(result)

    def _get_rmsea1(self, rmsea1: float, n: int, df: int, rmsea0: float, alpha: float, power: float) -> float:
        ncp0 = (n - 1) * df * rmsea0**2
        ncp1 = (n - 1) * df * rmsea1**2
        if self.test_type == "close":
            c_alpha = ncx2.ppf(1 - alpha, df, ncp0)
        else:
            c_alpha = ncx2.ppf(alpha, df, ncp0)
        result = ncx2.sf(c_alpha, df, ncp1) - power
        return float(result)

    def _get_alpha(self, alpha: float, n: int, df: int, rmsea0: float, rmsea1: float, power: float) -> float:
        ncp0 = (n - 1) * df * rmsea0**2
        ncp1 = (n - 1) * df * rmsea1**2
        if self.test_type == "close":
            c_alpha = ncx2.ppf(1 - alpha, df, ncp0)
        else:
            c_alpha = ncx2.ppf(alpha, df, ncp0)
        result = ncx2.sf(c_alpha, df, ncp1) - power
        return float(result)

    def pwr_test(self) -> dict:
        """Solve for the unspecified parameter and return the power-analysis results.

        Returns
        -------
        A dictionary containing n, df, rmsea0, rmsea1, alpha, power, and the method and url metadata of the test.
        """
        if self.power is None:
            if self.n is None or self.df is None or self.rmsea0 is None or self.rmsea1 is None or self.alpha is None:
                raise ValueError("n, df, rmsea0, rmsea1, and alpha must be provided to compute power")
            self.power = self._get_power(self.n, self.df, self.rmsea0, self.rmsea1, self.alpha)
        elif self.rmsea0 is None:
            if self.n is None or self.df is None or self.rmsea1 is None or self.alpha is None or self.power is None:
                raise ValueError("n, df, rmsea1, alpha, and power must be provided to solve for rmsea0")
            n, df, rmsea1, alpha, power = self.n, self.df, self.rmsea1, self.alpha, self.power
            self.rmsea0 = brentq(lambda rmsea0: self._get_rmsea0(rmsea0, n, df, rmsea1, alpha, power), 0, 1)
        elif self.rmsea1 is None:
            if self.n is None or self.df is None or self.rmsea0 is None or self.alpha is None or self.power is None:
                raise ValueError("n, df, rmsea0, alpha, and power must be provided to solve for rmsea1")
            n, df, rmsea0, alpha, power = self.n, self.df, self.rmsea0, self.alpha, self.power
            self.rmsea1 = brentq(lambda rmsea1: self._get_rmsea1(rmsea1, n, df, rmsea0, alpha, power), 0, 1)
        elif self.n is None:
            if (
                self.df is None
                or self.rmsea0 is None
                or self.rmsea1 is None
                or self.alpha is None
                or self.power is None
            ):  # noqa: E501
                raise ValueError("df, rmsea0, rmsea1, alpha, and power must be provided to solve for n")
            df, rmsea0, rmsea1, alpha, power = self.df, self.rmsea0, self.rmsea1, self.alpha, self.power
            self.n = ceil(brentq(lambda n: self._get_n(n, df, rmsea0, rmsea1, alpha, power), 2 + 1e-10, 1e09))
        elif self.df is None:
            if self.n is None or self.rmsea0 is None or self.rmsea1 is None or self.alpha is None or self.power is None:
                raise ValueError("n, rmsea0, rmsea1, alpha, and power must be provided to solve for df")
            n, rmsea0, rmsea1, alpha, power = self.n, self.rmsea0, self.rmsea1, self.alpha, self.power
            self.df = ceil(brentq(lambda df: self._get_df(df, n, rmsea0, rmsea1, alpha, power), 1, 1e04))
        else:
            if self.n is None or self.df is None or self.rmsea0 is None or self.rmsea1 is None or self.power is None:
                raise ValueError("n, df, rmsea0, rmsea1, and power must be provided to solve for alpha")
            n, df, rmsea0, rmsea1, power = self.n, self.df, self.rmsea0, self.rmsea1, self.power
            self.alpha = brentq(lambda alpha: self._get_alpha(alpha, n, df, rmsea0, rmsea1, power), 1e-10, 1 - 1e-10)
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
