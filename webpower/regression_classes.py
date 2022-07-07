import numpy as np

from math import ceil, log, exp, sqrt
from typing import Dict, Optional, Tuple, Union

from scipy.stats import ncf, f as f_dist, norm, lognorm, poisson, expon
from scipy.optimize import brentq
from scipy.integrate import quad


class WPRegression:
    def __init__(
            self,
            n: Optional[int] = None,
            p1: int = 1,
            p2: int = 0,
            f2: Optional[float] = None,
            alpha: Optional[float] = None,
            power: Optional[float] = None,
            test_type: str = "regular",
    ) -> None:
        self.n = n
        self.p1 = p1
        self.p2 = p2
        self.f2 = f2
        self.alpha = alpha
        self.power = power
        self.test_type = test_type.casefold()
        self.method = "Power for multiple regression"
        self.url = "http://psychstat.org/regression"
        self.u = p1 - p2

    def _get_power(self) -> float:
        v = self.n - self.p1 - 1
        if self.test_type == "cohen":
            lambda_ = self.f2 * (self.u + v + 1)
        else:
            lambda_ = self.f2 * self.n
        power = ncf.sf(f_dist.isf(self.alpha, self.u, v), self.u, v, lambda_)
        return power

    def _get_effect_size(self, f2: float) -> float:
        v = self.n - self.p1 - 1
        if self.test_type == "cohen":
            lambda_ = f2 * (self.u + v + 1)
        else:
            lambda_ = f2 * self.n
        f2 = ncf.sf(f_dist.isf(self.alpha, self.u, v), self.u, v, lambda_) - self.power
        return f2

    def _get_n(self, n: int) -> float:
        v = n - self.p1 - 1
        if self.test_type == "cohen":
            lambda_ = self.f2 * (self.u + v + 1)
        else:
            lambda_ = self.f2 * n
        n = ncf.sf(f_dist.isf(self.alpha, self.u, v), self.u, v, lambda_) - self.power
        return n

    def _get_alpha(self, alpha: float) -> float:
        v = self.n - self.p1 - 1
        if self.test_type == "cohen":
            lambda_ = self.f2 * (self.u + v + 1)
        else:
            lambda_ = self.f2 * self.n
        alpha = ncf.sf(f_dist.isf(alpha, self.u, v), self.u, v, lambda_) - self.power
        return alpha

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.n is None:
            self.n = ceil(brentq(self._get_n, 5 + self.p1 + 1e-10, 1e05))
        elif self.f2 is None:
            self.f2 = brentq(self._get_effect_size, 1e-07, 1e07)
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
        return {
            "effect_size": self.f2,
            "n": self.n,
            "p1": self.p1,
            "p2": self.p2,
            "alpha": self.alpha,
            "power": self.power,
            "method": self.method,
            "url": self.url,
        }


class WpPoisson:
    def __init__(
            self,
            n: Optional[int] = None,
            exp0: float = 1,
            exp1: float = 0.5,
            alpha: Optional[float] = None,
            power: Optional[float] = None,
            alternative: str = "two-sided",
            family: str = "Bernoulli",
            parameter: Optional[Union[int, float, list, tuple]] = None,
    ) -> None:
        self.n = n
        self.exp0 = exp0
        self.exp1 = exp1
        if alpha is not None:
            self.alpha = alpha / 2 if alternative.casefold() == "two-sided" else alpha
        else:
            self.alpha = alpha
        self.power = power
        self.alternative = alternative.casefold()
        self.family = family.casefold()
        if parameter is None:
            if self.family == "bernoulli":
                self.parameter = 0.5
            elif self.family in ["exponential", "poisson"]:
                self.parameter = 1
            elif self.family in ["lognormal", "normal", "uniform"]:
                self.parameter = [0, 1]
            else:
                raise ValueError(f"Do not recognize {family} for Poisson Regression")
        else:
            self.parameter = parameter
        self.method = "Power for Poisson regression"
        self.url = "http://psychstat.org/poisson"

    def _get_values(self) -> Tuple:
        beta1 = log(self.exp1)
        beta0 = log(self.exp0)
        if self.family == "bernoulli":
            d = (1 - self.parameter) * exp(beta0) + self.parameter * exp(beta0 + beta1)
            e = self.parameter * exp(beta0 + beta1)
            f = e
        elif self.family == "exponential":
            d = quad(
                lambda x: exp(beta0 + (beta1 - self.parameter) * x) * self.parameter,
                0,
                np.inf,
            )[0]
            e = quad(
                lambda x: x
                          * exp(beta0 + (beta1 - self.parameter) * x)
                          * self.parameter,
                0,
                np.inf,
            )[0]
            f = quad(
                lambda x: pow(x, 2)
                          * exp(beta0 + (beta1 - self.parameter) * x)
                          * self.parameter,
                0,
                np.inf,
            )[0]
        elif self.family == "lognormal":
            mu = self.parameter[0]
            sigma = self.parameter[1]
            d = quad(
                lambda x: exp(beta0 + beta1 * x) * lognorm.pdf(x, sigma, 0, exp(mu)), 0, np.inf
            )[0]
            e = quad(
                lambda x: x * exp(beta0 + beta1 * x) * lognorm.pdf(x, sigma, 0, exp(mu)),
                0,
                np.inf,
            )[0]
            f = quad(
                lambda x: pow(x, 2)
                          * exp(beta0 + beta1 * x)
                          * lognorm.pdf(x, sigma, 0, exp(mu)),
                0,
                np.inf,
            )[0]
        elif self.family == "normal":
            mu = self.parameter[0]
            sigma = self.parameter[1]
            d = quad(
                lambda x: exp(beta0 + beta1 * x) * norm.pdf(x, mu, sigma),
                -np.inf,
                np.inf,
            )[0]
            e = quad(
                lambda x: x * exp(beta0 + beta1 * x) * norm.pdf(x, mu, sigma),
                -np.inf,
                np.inf,
            )[0]
            f = quad(
                lambda x: pow(x, 2) * exp(beta0 + beta1 * x) * norm.pdf(x, mu, sigma),
                -np.inf,
                np.inf,
            )[0]
        elif self.family == "poisson":
            val_range = np.arange(0, int(1e05) + 1)
            d = np.sum(
                np.exp(beta0 + beta1 * val_range)
                * poisson.pmf(val_range, self.parameter)
            )
            e = np.sum(
                val_range
                * np.exp(beta0 + beta1 * val_range)
                * poisson.pmf(val_range, self.parameter)
            )
            f = np.sum(
                np.square(val_range)
                * np.exp(beta0 + beta1 * val_range)
                * poisson.pmf(val_range, self.parameter)
            )
        elif self.family == "uniform":
            l = self.parameter[0]
            r = self.parameter[1]
            d = quad(lambda x: exp(beta0 + beta1 * x) / (r - l), l, r)[0]
            e = quad(lambda x: x * exp(beta0 + beta1 * x) / (r - l), l, r)[0]
            f = quad(lambda x: pow(x, 2) * exp(beta0 + beta1 * x) / (r - l), l, r)[0]
        else:
            raise ValueError(f"Do not recognize {self.family} for Poisson Regression")
        v1 = d / (d * f - pow(e, 2))
        if self.alternative == "less":
            s = 1
            t = 0
        elif self.alternative == "two-sided":
            s = 1
            t = 1
        else:
            s = 0
            t = 1
        return s, t, v1, beta1

    def _get_power(self) -> float:
        s, t, v1, beta1 = self._get_values()
        power = s * norm.cdf(
            -norm.ppf(1 - self.alpha) - sqrt(self.n) / sqrt(v1) * beta1
        ) + t * norm.cdf(-norm.ppf(1 - self.alpha) + sqrt(self.n) / sqrt(v1) * beta1)
        return power

    def _get_n(self, n: int) -> float:
        s, t, v1, beta1 = self._get_values()
        n = (
                s * norm.cdf(-norm.ppf(1 - self.alpha) - sqrt(n) / sqrt(v1) * beta1)
                + t * norm.cdf(-norm.ppf(1 - self.alpha) + sqrt(n) / sqrt(v1) * beta1)
                - self.power
        )
        return n

    def _get_alpha(self, alpha: float) -> float:
        s, t, v1, beta1 = self._get_values()
        alpha = (
                s * norm.cdf(-norm.ppf(1 - alpha) - sqrt(self.n) / sqrt(v1) * beta1)
                + t * norm.cdf(-norm.ppf(1 - alpha) + sqrt(self.n) / sqrt(v1) * beta1)
                - self.power
        )
        return alpha

    def pwr_test(self) -> Dict:
        if self.power is None:
            self.power = self._get_power()
        elif self.n is None:
            self.n = ceil(brentq(self._get_n, 2 + 1e-10, 1e07))
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
        return {
            "n": self.n,
            "power": self.power,
            "alpha": self.alpha,
            "exp0": self.exp0,
            "exp1": self.exp1,
            "beta0": log(self.exp0),
            "beta1": log(self.exp1),
            "method": self.method,
            "url": self.url,
        }


class WpLogistic:
    def __init__(
            self,
            n: Optional[int] = None,
            p0: float = 0.5,
            p1: float = 0.5,
            alpha: Optional[float] = None,
            power: Optional[float] = None,
            alternative: str = "two-sided",
            family: str = "Bernoulli",
            parameter: Optional[Union[int, float, list, tuple]] = None,
    ) -> None:
        self.n = n
        if abs(p0) > 1:
            raise ValueError("p0 must be a float between 0 and 1")
        self.p0 = p0
        if abs(p1) > 1:
            raise ValueError("p1 must be a float between 0 and 1")
        self.p1 = p1
        if alpha is not None:
            self.alpha = alpha / 2 if alternative.casefold() == "two-sided" else alpha
        else:
            self.alpha = alpha
        self.power = power
        self.alternative = alternative.casefold()
        self.family = family.casefold()

        if parameter is None:
            if self.family == "bernoulli":
                self.parameter = 0.5
            elif self.family in ["exponential", "poisson"]:
                self.parameter = 1
            elif self.family in ["lognormal", "normal", "uniform"]:
                self.parameter = [0, 1]
            else:
                raise ValueError(f"Do not recognize {family} for Poisson Regression")
        else:
            self.parameter = parameter
        self.method = "Power for Logistic regression"
        self.url = "http://psychstat.org/logistic"

    def _get_values(self) -> Tuple:
        g = 0
        odds = (self.p1 / (1 - self.p1)) / (self.p0 / (1 - self.p0))
        self.beta1 = log(odds)
        self.beta0 = log(self.p0 / (1 - self.p0))
        if self.family == "bernoulli":
            d = self.parameter * self.p1 * (1 - self.p1) + (
                    1 - self.parameter
            ) * self.p0 * (1 - self.p0)
            e = self.parameter * self.p1 * (1 - self.p1)
            v1 = d / (d * e - pow(e, 2))
            mu1 = self.parameter * self.p1 + (1 - self.parameter) * self.p0
            i00 = log(mu1 / (1 - mu1))
            pn = 1 / (1 + exp(-i00))
            a = pn * (1 - pn)
            b = self.parameter * a
            v0 = b / (a * b - pow(b, 2))
        elif self.family == "exponential":
            d = quad(
                lambda x: (1 - 1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                          * 1
                          / (1 + exp(-self.beta0 - self.beta1 * x))
                          * expon.pdf(x, scale=self.parameter),
                a=0,
                b=100,
                limit=100,
            )[0]
            e = quad(
                lambda x: x
                          * (1 - 1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                          * 1
                          / (1 + exp(-self.beta0 - self.beta1 * x))
                          * expon.pdf(x, scale=self.parameter),
                a=0,
                b=100,
                limit=100,
            )[0]
            f = quad(
                lambda x: pow(x, 2)
                          * (1 - 1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                          * 1
                          / (1 + exp(-self.beta0 - self.beta1 * x))
                          * expon.pdf(x, scale=self.parameter),
                a=0,
                b=100,
                limit=100,
            )[0]
            v1 = d / (d * f - pow(e, 2))
            mu1 = quad(
                lambda x: 1
                          / (1 + exp(-self.beta0 - self.beta1 * x))
                          * expon.pdf(x, scale=self.parameter),
                a=0,
                b=100,
                limit=100,
            )[0]
            i00 = log(mu1 / (1 - mu1))
            pn = 1 / (1 + exp(-i00))
            a = pn * (1 - pn)
            b = 2 * pow(self.parameter, -2) * pn * (1 - pn)
            c = pow(self.parameter, -1) * pn * (1 - pn)
            v0 = a / (a * b - pow(c, 2))
        elif self.family == "lognormal":
            mu = self.parameter[0]
            sigma = self.parameter[1]
            d = quad(
                lambda x: (1 - (1 / (1 + exp(-self.beta0 - self.beta1 * x))))
                          * 1
                          / (1 + exp(-self.beta0 - self.beta1 * x))
                          * lognorm.pdf(x, sigma, scale=exp(mu)),
                a=0,
                b=100,
                limit=100,
            )[0]
            e = quad(
                lambda x: x
                          * (1 - (1 / (1 + exp(-self.beta0 - self.beta1 * x))))
                          * 1
                          / (1 + exp(-self.beta0 - self.beta1 * x))
                          * lognorm.pdf(x, sigma, scale=exp(mu)),
                a=0,
                b=100,
                limit=100,
            )[0]
            f = quad(
                lambda x: pow(x, 2)
                          * (1 - (1 / (1 + exp(-self.beta0 - self.beta1 * x))))
                          * 1
                          / (1 + exp(-self.beta0 - self.beta1 * x))
                          * lognorm.pdf(x, sigma, scale=exp(mu)),
                a=0,
                b=100,
                limit=100,
            )[0]
            v1 = d / (d * f - pow(e, 2))
            mu1 = quad(
                lambda x: 1
                          / (1 + exp(-self.beta0 - self.beta1 * x))
                          * lognorm.pdf(x, sigma, scale=exp(mu)),
                a=0,
                b=100,
                limit=100,
            )[0]
            i00 = log(mu1 / (1 - mu1))
            pn = 1 / (1 + exp(-i00))
            a = pn * (1 - pn)
            b = (exp(pow(sigma, 2)) - 1) * exp(2 * mu + pow(sigma, 2)) * pn * (1 - pn)
            c = exp(mu + 0.5 * pow(sigma, 2)) * pn * (1 - pn)
            v0 = a / (a * b - pow(c, 2))
        elif self.family == "normal":
            mu = self.parameter[0]
            sigma = self.parameter[1]
            d = quad(
                lambda x: (1 - 1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                          * 1
                          / (1 + exp(-self.beta0 - self.beta1 * x))
                          * norm.pdf(x, mu, sigma),
                a=-100,
                b=100,
                limit=100,
            )[0]
            e = quad(
                lambda x: x
                          * (1 - 1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                          * 1
                          / (1 + exp(-self.beta0 - self.beta1 * x))
                          * norm.pdf(x, mu, sigma),
                a=-100,
                b=100,
                limit=100,
            )[0]
            f = quad(
                lambda x: pow(x, 2)
                          * (1 - 1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                          * 1
                          / (1 + exp(-self.beta0 - self.beta1 * x))
                          * norm.pdf(x, mu, sigma),
                a=-100,
                b=100,
                limit=100,
            )[0]
            v1 = d / (d * f - pow(e, 2))
            mu1 = quad(
                lambda x: 1
                          / (1 + exp(-self.beta0 - self.beta1 * x))
                          * norm.pdf(x, mu, sigma),
                a=-100,
                b=100,
                limit=100,
            )[0]
            i00 = log(mu1 / (1 - mu1))
            pn = 1 / (1 + exp(-i00))
            a = pn * (1 - pn)
            b = (exp(pow(sigma, 2)) - 1) * exp(2 * mu + pow(sigma, 2)) * pn * (1 - pn)
            c = exp(mu + 0.5 * pow(sigma, 2)) * pn * (1 - pn)
            v0 = a / (a * b - pow(c, 2))
        elif self.family == "poisson":
            val_range = np.arange(0, int(1e05) + 1)
            d = np.sum(
                (1 - 1 / (1 + np.exp(-self.beta0 - self.beta1 * val_range)))
                * 1
                / (1 + np.exp(-self.beta0 - self.beta1 * val_range))
                * poisson.pmf(val_range, self.parameter)
            )
            e = val_range * np.sum(
                (1 - 1 / (1 + np.exp(-self.beta0 - self.beta1 * val_range)))
                * 1
                / (1 + np.exp(-self.beta0 - self.beta1 * val_range))
                * poisson.pmf(val_range, self.parameter)
            )
            f = np.square(val_range) * np.sum(
                (1 - 1 / (1 + np.exp(-self.beta0 - self.beta1 * val_range)))
                * 1
                / (1 + np.exp(-self.beta0 - self.beta1 * val_range))
                * poisson.pmf(val_range, self.parameter)
            )
            v1 = d / (d * f - pow(e, 2))
            mu1 = np.sum(
                1
                / (1 + np.exp(-self.beta0 - self.beta1 * val_range))
                * poisson.pmf(val_range, self.parameter)
            )
            i00 = log(mu1 / (1 - mu1))
            pn = 1 / (1 + exp(-i00))
            a = pn * (1 - pn)
            b = self.parameter * pn * (1 - pn)
            c = self.parameter * pn * (1 - pn)
            v0 = a / (a * b - pow(c, 2))
        elif self.family == "uniform":
            L = self.parameter[0]
            R = self.parameter[1]
            d = quad(
                lambda x: (1 - (1 / (1 + exp(-self.beta0 - self.beta1 * x))))
                          * (1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                          / (R - L),
                a=L,
                b=R,
                limit=100,
            )[0]
            e = quad(
                lambda x: x
                          * (1 - (1 / (1 + exp(-self.beta0 - self.beta1 * x))))
                          * (1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                          / (R - L),
                a=L,
                b=R,
                limit=100,
            )[0]
            f = quad(
                lambda x: pow(x, 2)
                          * (1 - (1 / (1 + exp(-self.beta0 - self.beta1 * x))))
                          * (1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                          / (R - L),
                a=L,
                b=R,
                limit=100,
            )[0]
            v1 = d / (d * f - pow(e, 2))
            mu1 = quad(
                lambda x: 1 / (1 + exp(-self.beta0 - self.beta1 * x)) / (R - L),
                a=L,
                b=R,
                limit=100,
            )[0]
            i00 = log(mu1 / (1 - mu1))
            pn = 1 / (1 + exp(-i00))
            a = pn * (1 - pn)
            b = pow(R - L, 2) / 12 * pn * (1 - pn)
            c = pow(R - L, 2) * pn * (1 - pn)
            v0 = a / (a * b - pow(c, 2))
        else:
            raise ValueError(f"Do not recognize {self.family} for Logistic Regression")
        if self.alternative == "less":
            s = 1
            t = 0
        elif self.alternative == "two-sided":
            s = 1
            t = 1
        else:
            s = 0
            t = 1
        return s, t, g, v0, v1

    def _get_power(self) -> float:
        s, t, g, v0, v1 = self._get_values()
        power = s * norm.cdf(
            -norm.ppf(1 - self.alpha)
            - sqrt(self.n) / sqrt(g * v0 + (1 - g) * v1) * self.beta1
        ) + t * norm.cdf(
            -norm.ppf(1 - self.alpha)
            + sqrt(self.n) / sqrt(g * v0 + (1 - g) * v1) * self.beta1
        )
        return power

    def _get_n(self, n: int) -> float:
        s, t, g, v0, v1 = self._get_values()
        n = s * norm.cdf(
            -norm.ppf(1 - self.alpha)
            - sqrt(n) / sqrt(g * v0 + (1 - g) * v1) * self.beta1
        ) + t * norm.cdf(
            -norm.ppf(1 - self.alpha)
            + sqrt(n) / sqrt(g * v0 + (1 - g) * v1) * self.beta1
        ) - self.power
        return n

    def _get_alpha(self, alpha):
        s, t, g, v0, v1 = self._get_values()
        alpha = s * norm.cdf(
            -norm.ppf(1 - alpha)
            - sqrt(self.n) / sqrt(g * v0 + (1 - g) * v1) * self.beta1
        ) + t * norm.cdf(
            -norm.ppf(1 - alpha)
            + sqrt(self.n) / sqrt(g * v0 + (1 - g) * v1) * self.beta1
        ) - self.power
        return alpha

    def pwr_test(self):
        if self.power is None:
            self.power = self._get_power()
        elif self.n is None:
            self.n = ceil(brentq(self._get_n, 2 + 1e-10, 1e07))
        else:
            self.alpha = brentq(self._get_alpha, 1e-10, 1 - 1e-10)
        return {
            "n": self.n,
            "power": self.power,
            "alpha": self.alpha,
            "p0": self.p0,
            "p1": self.p1,
            "beta0": self.beta0,
            "beta1": self.beta1,
            "method": self.method,
            "url": self.url,
        }
