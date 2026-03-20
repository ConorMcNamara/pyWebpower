from math import ceil, exp, log, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import expon, lognorm, ncf, norm, poisson
from scipy.stats import f as f_dist


class WPRegression:
    def __init__(
        self,
        n: int | None = None,
        p1: int = 1,
        p2: int = 0,
        f2: float | None = None,
        alpha: float | None = None,
        power: float | None = None,
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

    def _get_power(self, n: int, f2: float, alpha: float) -> float:
        v = n - self.p1 - 1
        if self.test_type == "cohen":
            lambda_ = f2 * (self.u + v + 1)
        else:
            lambda_ = f2 * n
        power = ncf.sf(f_dist.isf(alpha, self.u, v), self.u, v, lambda_)
        return float(power)

    def _get_effect_size(self, f2: float, n: int, alpha: float, power: float) -> float:
        v = n - self.p1 - 1
        if self.test_type == "cohen":
            lambda_ = f2 * (self.u + v + 1)
        else:
            lambda_ = f2 * n
        result: float = ncf.sf(f_dist.isf(alpha, self.u, v), self.u, v, lambda_) - power
        return float(result)

    def _get_n(self, n: float, f2: float, alpha: float, power: float) -> float:
        v = n - self.p1 - 1
        if self.test_type == "cohen":
            lambda_ = f2 * (self.u + v + 1)
        else:
            lambda_ = f2 * n
        result: float = ncf.sf(f_dist.isf(alpha, self.u, v), self.u, v, lambda_) - power
        return float(result)

    def _get_alpha(self, alpha: float, n: int, f2: float, power: float) -> float:
        v = n - self.p1 - 1
        if self.test_type == "cohen":
            lambda_ = f2 * (self.u + v + 1)
        else:
            lambda_ = f2 * n
        result: float = ncf.sf(f_dist.isf(alpha, self.u, v), self.u, v, lambda_) - power
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.n is None or self.f2 is None or self.alpha is None:
                raise ValueError("n, f2, and alpha must be provided to compute power")
            self.power = self._get_power(self.n, self.f2, self.alpha)
        elif self.n is None:
            if self.f2 is None or self.alpha is None or self.power is None:
                raise ValueError("f2, alpha, and power must be provided to solve for n")
            f2, alpha, power = self.f2, self.alpha, self.power
            self.n = ceil(brentq(lambda n: self._get_n(n, f2, alpha, power), 5 + self.p1 + 1e-10, 1e05))
        elif self.f2 is None:
            if self.n is None or self.alpha is None or self.power is None:
                raise ValueError("n, alpha, and power must be provided to solve for f2")
            n, alpha, power = self.n, self.alpha, self.power
            self.f2 = brentq(lambda f2: self._get_effect_size(f2, n, alpha, power), 1e-07, 1e07)
        else:
            if self.n is None or self.f2 is None or self.power is None:
                raise ValueError("n, f2, and power must be provided to solve for alpha")
            n, f2, power = self.n, self.f2, self.power
            self.alpha = brentq(lambda alpha: self._get_alpha(alpha, n, f2, power), 1e-10, 1 - 1e-10)
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
        n: int | None = None,
        exp0: float = 1,
        exp1: float = 0.5,
        alpha: float | None = None,
        power: float | None = None,
        alternative: str = "two-sided",
        family: str = "Bernoulli",
        parameter: int | float | list | tuple | None = None,
    ) -> None:
        self.n = n
        self.exp0 = exp0
        self.exp1 = exp1
        self.alpha: float | None
        if alpha is not None:
            self.alpha = alpha / 2 if alternative.casefold() == "two-sided" else alpha
        else:
            self.alpha = alpha
        self.power = power
        self.alternative = alternative.casefold()
        self.family = family.casefold()
        self.parameter: int | float | list | tuple
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

    def _get_values(self) -> tuple:
        beta1 = log(self.exp1)
        beta0 = log(self.exp0)
        if self.family == "bernoulli":
            assert isinstance(self.parameter, (int, float))
            param_scalar: int | float = self.parameter
            d = (1 - param_scalar) * exp(beta0) + param_scalar * exp(beta0 + beta1)
            e = param_scalar * exp(beta0 + beta1)
            f = e
        elif self.family == "exponential":
            assert isinstance(self.parameter, (int, float))
            param_scalar = self.parameter
            d = quad(
                lambda x: exp(beta0 + (beta1 - param_scalar) * x) * param_scalar,
                0,
                np.inf,
            )[0]
            e = quad(
                lambda x: x * exp(beta0 + (beta1 - param_scalar) * x) * param_scalar,
                0,
                np.inf,
            )[0]
            f = quad(
                lambda x: x**2 * exp(beta0 + (beta1 - param_scalar) * x) * param_scalar,
                0,
                np.inf,
            )[0]
        elif self.family == "lognormal":
            assert isinstance(self.parameter, (list, tuple))
            param_seq: list | tuple = self.parameter
            mu = param_seq[0]
            sigma = param_seq[1]
            d = quad(lambda x: exp(beta0 + beta1 * x) * lognorm.pdf(x, sigma, 0, exp(mu)), 0, np.inf)[0]
            e = quad(
                lambda x: x * exp(beta0 + beta1 * x) * lognorm.pdf(x, sigma, 0, exp(mu)),
                0,
                np.inf,
            )[0]
            f = quad(
                lambda x: x**2 * exp(beta0 + beta1 * x) * lognorm.pdf(x, sigma, 0, exp(mu)),
                0,
                np.inf,
            )[0]
        elif self.family == "normal":
            assert isinstance(self.parameter, (list, tuple))
            param_seq = self.parameter
            mu = param_seq[0]
            sigma = param_seq[1]
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
                lambda x: x**2 * exp(beta0 + beta1 * x) * norm.pdf(x, mu, sigma),
                -np.inf,
                np.inf,
            )[0]
        elif self.family == "poisson":
            assert isinstance(self.parameter, (int, float))
            param_scalar = self.parameter
            val_range = np.arange(0, int(1e05) + 1)
            d = np.sum(np.exp(beta0 + beta1 * val_range) * poisson.pmf(val_range, param_scalar))
            e = np.sum(val_range * np.exp(beta0 + beta1 * val_range) * poisson.pmf(val_range, param_scalar))
            f = np.sum(np.square(val_range) * np.exp(beta0 + beta1 * val_range) * poisson.pmf(val_range, param_scalar))
        elif self.family == "uniform":
            assert isinstance(self.parameter, (list, tuple))
            param_seq = self.parameter
            l_var = param_seq[0]
            r = param_seq[1]
            d = quad(lambda x: exp(beta0 + beta1 * x) / (r - l_var), l_var, r)[0]
            e = quad(lambda x: x * exp(beta0 + beta1 * x) / (r - l_var), l_var, r)[0]
            f = quad(lambda x: x**2 * exp(beta0 + beta1 * x) / (r - l_var), l_var, r)[0]
        else:
            raise ValueError(f"Do not recognize {self.family} for Poisson Regression")
        v1 = d / (d * f - e**2)
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

    def _get_power(self, n: int, alpha: float) -> float:
        s, t, v1, beta1 = self._get_values()
        power = s * norm.cdf(-norm.ppf(1 - alpha) - sqrt(n) / sqrt(v1) * beta1) + t * norm.cdf(
            -norm.ppf(1 - alpha) + sqrt(n) / sqrt(v1) * beta1
        )
        return float(power)

    def _get_n(self, n: float, alpha: float, power: float) -> float:
        s, t, v1, beta1 = self._get_values()
        result: float = (
            s * norm.cdf(-norm.ppf(1 - alpha) - sqrt(n) / sqrt(v1) * beta1)
            + t * norm.cdf(-norm.ppf(1 - alpha) + sqrt(n) / sqrt(v1) * beta1)
            - power
        )
        return float(result)

    def _get_alpha(self, alpha: float, n: int, power: float) -> float:
        s, t, v1, beta1 = self._get_values()
        result: float = (
            s * norm.cdf(-norm.ppf(1 - alpha) - sqrt(n) / sqrt(v1) * beta1)
            + t * norm.cdf(-norm.ppf(1 - alpha) + sqrt(n) / sqrt(v1) * beta1)
            - power
        )
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.n is None or self.alpha is None:
                raise ValueError("n and alpha must be provided to compute power")
            self.power = self._get_power(self.n, self.alpha)
        elif self.n is None:
            if self.alpha is None or self.power is None:
                raise ValueError("alpha and power must be provided to solve for n")
            alpha, power = self.alpha, self.power
            self.n = ceil(brentq(lambda n: self._get_n(n, alpha, power), 2 + 1e-10, 1e07))
        else:
            if self.n is None or self.power is None:
                raise ValueError("n and power must be provided to solve for alpha")
            n, power = self.n, self.power
            self.alpha = brentq(lambda alpha: self._get_alpha(alpha, n, power), 1e-10, 1 - 1e-10)
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
        n: int | None = None,
        p0: float = 0.5,
        p1: float = 0.5,
        alpha: float | None = None,
        power: float | None = None,
        alternative: str = "two-sided",
        family: str = "Bernoulli",
        parameter: int | float | list | tuple | None = None,
    ) -> None:
        self.n = n
        if abs(p0) > 1:
            raise ValueError("p0 must be a float between 0 and 1")
        self.p0 = p0
        if abs(p1) > 1:
            raise ValueError("p1 must be a float between 0 and 1")
        self.p1 = p1
        self.alpha: float | None
        if alpha is not None:
            self.alpha = alpha / 2 if alternative.casefold() == "two-sided" else alpha
        else:
            self.alpha = alpha
        self.power = power
        self.alternative = alternative.casefold()
        self.family = family.casefold()
        self.parameter: int | float | list | tuple

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

    def _get_values(self) -> tuple:
        g = 0
        odds = (self.p1 / (1 - self.p1)) / (self.p0 / (1 - self.p0))
        self.beta1 = log(odds)
        self.beta0 = log(self.p0 / (1 - self.p0))
        if self.family == "bernoulli":
            assert isinstance(self.parameter, (int, float))
            param_scalar: int | float = self.parameter
            d = param_scalar * self.p1 * (1 - self.p1) + (1 - param_scalar) * self.p0 * (1 - self.p0)
            e = param_scalar * self.p1 * (1 - self.p1)
            v1 = d / (d * e - e**2)
            mu1 = param_scalar * self.p1 + (1 - param_scalar) * self.p0
            i00 = log(mu1 / (1 - mu1))
            pn = 1 / (1 + exp(-i00))
            a = pn * (1 - pn)
            b = param_scalar * a
            v0 = b / (a * b - b**2)
        elif self.family == "exponential":
            assert isinstance(self.parameter, (int, float))
            param_scalar = self.parameter
            d = quad(
                lambda x: (
                    (1 - 1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                    * 1
                    / (1 + exp(-self.beta0 - self.beta1 * x))
                    * expon.pdf(x, scale=param_scalar)
                ),
                a=0,
                b=100,
                limit=100,
            )[0]
            e = quad(
                lambda x: (
                    x
                    * (1 - 1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                    * 1
                    / (1 + exp(-self.beta0 - self.beta1 * x))
                    * expon.pdf(x, scale=param_scalar)
                ),
                a=0,
                b=100,
                limit=100,
            )[0]
            f = quad(
                lambda x: (
                    x**2
                    * (1 - 1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                    * 1
                    / (1 + exp(-self.beta0 - self.beta1 * x))
                    * expon.pdf(x, scale=param_scalar)
                ),
                a=0,
                b=100,
                limit=100,
            )[0]
            v1 = d / (d * f - e**2)
            mu1 = quad(
                lambda x: 1 / (1 + exp(-self.beta0 - self.beta1 * x)) * expon.pdf(x, scale=param_scalar),
                a=0,
                b=100,
                limit=100,
            )[0]
            i00 = log(mu1 / (1 - mu1))
            pn = 1 / (1 + exp(-i00))
            a = pn * (1 - pn)
            b = 2 * param_scalar**-2 * pn * (1 - pn)
            c = param_scalar**-1 * pn * (1 - pn)
            v0 = a / (a * b - c**2)
        elif self.family == "lognormal":
            assert isinstance(self.parameter, (list, tuple))
            param_seq: list | tuple = self.parameter
            mu = param_seq[0]
            sigma = param_seq[1]
            d = quad(
                lambda x: (
                    (1 - (1 / (1 + exp(-self.beta0 - self.beta1 * x))))
                    * 1
                    / (1 + exp(-self.beta0 - self.beta1 * x))
                    * lognorm.pdf(x, sigma, scale=exp(mu))
                ),
                a=0,
                b=100,
                limit=100,
            )[0]
            e = quad(
                lambda x: (
                    x
                    * (1 - (1 / (1 + exp(-self.beta0 - self.beta1 * x))))
                    * 1
                    / (1 + exp(-self.beta0 - self.beta1 * x))
                    * lognorm.pdf(x, sigma, scale=exp(mu))
                ),
                a=0,
                b=100,
                limit=100,
            )[0]
            f = quad(
                lambda x: (
                    x**2
                    * (1 - (1 / (1 + exp(-self.beta0 - self.beta1 * x))))
                    * 1
                    / (1 + exp(-self.beta0 - self.beta1 * x))
                    * lognorm.pdf(x, sigma, scale=exp(mu))
                ),
                a=0,
                b=100,
                limit=100,
            )[0]
            v1 = d / (d * f - e**2)
            mu1 = quad(
                lambda x: 1 / (1 + exp(-self.beta0 - self.beta1 * x)) * lognorm.pdf(x, sigma, scale=exp(mu)),
                a=0,
                b=100,
                limit=100,
            )[0]
            i00 = log(mu1 / (1 - mu1))
            pn = 1 / (1 + exp(-i00))
            a = pn * (1 - pn)
            b = (exp(sigma**2) - 1) * exp(2 * mu + sigma**2) * pn * (1 - pn)
            c = exp(mu + 0.5 * sigma**2) * pn * (1 - pn)
            v0 = a / (a * b - c**2)
        elif self.family == "normal":
            assert isinstance(self.parameter, (list, tuple))
            param_seq = self.parameter
            mu = param_seq[0]
            sigma = param_seq[1]
            d = quad(
                lambda x: (
                    (1 - 1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                    * 1
                    / (1 + exp(-self.beta0 - self.beta1 * x))
                    * norm.pdf(x, mu, sigma)
                ),
                a=-100,
                b=100,
                limit=100,
            )[0]
            e = quad(
                lambda x: (
                    x
                    * (1 - 1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                    * 1
                    / (1 + exp(-self.beta0 - self.beta1 * x))
                    * norm.pdf(x, mu, sigma)
                ),
                a=-100,
                b=100,
                limit=100,
            )[0]
            f = quad(
                lambda x: (
                    x**2
                    * (1 - 1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                    * 1
                    / (1 + exp(-self.beta0 - self.beta1 * x))
                    * norm.pdf(x, mu, sigma)
                ),
                a=-100,
                b=100,
                limit=100,
            )[0]
            v1 = d / (d * f - e**2)
            mu1 = quad(
                lambda x: 1 / (1 + exp(-self.beta0 - self.beta1 * x)) * norm.pdf(x, mu, sigma),
                a=-100,
                b=100,
                limit=100,
            )[0]
            i00 = log(mu1 / (1 - mu1))
            pn = 1 / (1 + exp(-i00))
            a = pn * (1 - pn)
            b = (exp(sigma**2) - 1) * exp(2 * mu + sigma**2) * pn * (1 - pn)
            c = exp(mu + 0.5 * sigma**2) * pn * (1 - pn)
            v0 = a / (a * b - c**2)
        elif self.family == "poisson":
            assert isinstance(self.parameter, (int, float))
            param_scalar = self.parameter
            val_range = np.arange(0, int(1e05) + 1)
            d = np.sum(
                (1 - 1 / (1 + np.exp(-self.beta0 - self.beta1 * val_range)))
                * 1
                / (1 + np.exp(-self.beta0 - self.beta1 * val_range))
                * poisson.pmf(val_range, param_scalar)
            )
            e = val_range * np.sum(
                (1 - 1 / (1 + np.exp(-self.beta0 - self.beta1 * val_range)))
                * 1
                / (1 + np.exp(-self.beta0 - self.beta1 * val_range))
                * poisson.pmf(val_range, param_scalar)
            )
            f = np.square(val_range) * np.sum(
                (1 - 1 / (1 + np.exp(-self.beta0 - self.beta1 * val_range)))
                * 1
                / (1 + np.exp(-self.beta0 - self.beta1 * val_range))
                * poisson.pmf(val_range, param_scalar)
            )
            v1 = d / (d * f - e**2)
            mu1 = np.sum(1 / (1 + np.exp(-self.beta0 - self.beta1 * val_range)) * poisson.pmf(val_range, param_scalar))
            i00 = log(mu1 / (1 - mu1))
            pn = 1 / (1 + exp(-i00))
            a = pn * (1 - pn)
            b = param_scalar * pn * (1 - pn)
            c = param_scalar * pn * (1 - pn)
            v0 = a / (a * b - c**2)
        elif self.family == "uniform":
            assert isinstance(self.parameter, (list, tuple))
            param_seq = self.parameter
            L = param_seq[0]
            R = param_seq[1]
            d = quad(
                lambda x: (
                    (1 - (1 / (1 + exp(-self.beta0 - self.beta1 * x))))
                    * (1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                    / (R - L)
                ),
                a=L,
                b=R,
                limit=100,
            )[0]
            e = quad(
                lambda x: (
                    x
                    * (1 - (1 / (1 + exp(-self.beta0 - self.beta1 * x))))
                    * (1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                    / (R - L)
                ),
                a=L,
                b=R,
                limit=100,
            )[0]
            f = quad(
                lambda x: (
                    x**2
                    * (1 - (1 / (1 + exp(-self.beta0 - self.beta1 * x))))
                    * (1 / (1 + exp(-self.beta0 - self.beta1 * x)))
                    / (R - L)
                ),
                a=L,
                b=R,
                limit=100,
            )[0]
            v1 = d / (d * f - e**2)
            mu1 = quad(
                lambda x: 1 / (1 + exp(-self.beta0 - self.beta1 * x)) / (R - L),
                a=L,
                b=R,
                limit=100,
            )[0]
            i00 = log(mu1 / (1 - mu1))
            pn = 1 / (1 + exp(-i00))
            a = pn * (1 - pn)
            b = (R - L) ** 2 / 12 * pn * (1 - pn)
            c = (R - L) ** 2 * pn * (1 - pn)
            v0 = a / (a * b - c**2)
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

    def _get_power(self, n: int, alpha: float) -> float:
        s, t, g, v0, v1 = self._get_values()
        power = s * norm.cdf(-norm.ppf(1 - alpha) - sqrt(n) / sqrt(g * v0 + (1 - g) * v1) * self.beta1) + t * norm.cdf(
            -norm.ppf(1 - alpha) + sqrt(n) / sqrt(g * v0 + (1 - g) * v1) * self.beta1
        )
        return float(power)

    def _get_n(self, n: float, alpha: float, power: float) -> float:
        s, t, g, v0, v1 = self._get_values()
        result: float = (
            s * norm.cdf(-norm.ppf(1 - alpha) - sqrt(n) / sqrt(g * v0 + (1 - g) * v1) * self.beta1)
            + t * norm.cdf(-norm.ppf(1 - alpha) + sqrt(n) / sqrt(g * v0 + (1 - g) * v1) * self.beta1)
            - power
        )
        return float(result)

    def _get_alpha(self, alpha: float, n: int, power: float) -> float:
        s, t, g, v0, v1 = self._get_values()
        result: float = (
            s * norm.cdf(-norm.ppf(1 - alpha) - sqrt(n) / sqrt(g * v0 + (1 - g) * v1) * self.beta1)
            + t * norm.cdf(-norm.ppf(1 - alpha) + sqrt(n) / sqrt(g * v0 + (1 - g) * v1) * self.beta1)
            - power
        )
        return float(result)

    def pwr_test(self) -> dict:
        if self.power is None:
            if self.n is None or self.alpha is None:
                raise ValueError("n and alpha must be provided to compute power")
            self.power = self._get_power(self.n, self.alpha)
        elif self.n is None:
            if self.alpha is None or self.power is None:
                raise ValueError("alpha and power must be provided to solve for n")
            alpha, power = self.alpha, self.power
            self.n = ceil(brentq(lambda n: self._get_n(n, alpha, power), 2 + 1e-10, 1e07))
        else:
            if self.n is None or self.power is None:
                raise ValueError("n and power must be provided to solve for alpha")
            n, power = self.n, self.power
            self.alpha = brentq(lambda alpha: self._get_alpha(alpha, n, power), 1e-10, 1 - 1e-10)
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
