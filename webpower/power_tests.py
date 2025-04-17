from typing import Dict, Optional, Union

from webpower.anova_classes import (
    WpAnovaClass,
    WpAnovaBinaryClass,
    WpAnovaCountClass,
    WpKAnovaClass,
    WpRMAnovaClass,
)
from webpower.proportion_classes import WpOneProp, WpTwoPropOneN, WpTwoPropTwoN
from webpower.t_test_classes import WpOneT, WpTwoT
from webpower.regression_classes import WPRegression, WpPoisson, WpLogistic
from webpower.sem_classes import WPSEMChisq, WPSEMRMSEA
from webpower.misc_classes import WpMediation, WpCorrelation
from webpower.randomized_trial_classes import WpMRT2Arm, WpMRT3Arm, WpCRT2Arm, WpCRT3Arm


def wp_anova_test(
    k: Optional[int] = None,
    n: Optional[int] = None,
    f: Optional[float] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    test_type: str = "overall",
    print_pretty: bool = True,
) -> Dict:
    """One-way analysis of variance (one-way ANOVA) is a technique used to compare means of two or more groups (e.g.,
    Maxwell & Delaney, 2003). The ANOVA tests the null hypothesis that samples in two or more groups are drawn from
    populations with the same mean values. The ANOVA analysis typically produces an F-statistic, the ratio of the
    between-group variance to the within-group variance.

    Parameters
    ----------
    k: int, default=None
        Number of groups
    n: int, default=None
        Sample size
    f: float, default=None
        Effect size
    alpha: float, default=None
        Significance level of the test
    power: float, default=None
        Statistical power
    test_type: {'overall', 'two-sided', 'greater', 'less'}
        The option "overall" is for the overall test of anova; "two-sided" is for a contrast anova; "greater" is testing
        the between-group variance greater than the within-group, while "less" is vis versus.
    print_pretty: bool, default=True
        Whether we want our results printed our not

    Returns
    -------
    A dictionary containing k, n, f, alpha, power and the test_type

    Notes
    -----
    Behavior is similar to pyPWR where one of the variables must be left as None and the other ones filled.
    """
    if not any(v is None for v in [k, n, f, alpha, power]):
        raise ValueError("One of k, n, f, alpha or power must be None")
    if sum([v is None for v in [k, n, f, alpha, power]]) > 1:
        raise ValueError("Only one of k, n, f, alpha or power may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    if n is not None and n < 1:
        raise ValueError("n must be a positive integer")
    if k is not None and k < 1:
        raise ValueError("k must be a positive integer")
    test_type = test_type.casefold()
    test = WpAnovaClass(k, n, f, alpha, power, test_type).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(test["k"])) - 1)
            + "k"
            + " " * (len(str(test["n"])))
            + "n"
            + " " * (len(str(round(test["effect_size"], 4))))
            + "f"
            + " "
            + "alpha"
            + " "
            + "power"
            + "\n"
            + "\t"
            + f"{test['k']}"
            + " "
            + f"{test['n']}"
            + " "
            + f"{round(test['effect_size'], 4)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " \n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test


def wp_anova_binary_test(
    k: Optional[int] = None,
    n: Optional[int] = None,
    V: Optional[float] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    print_pretty: bool = True,
) -> Dict:
    """The power analysis procedure for one-way ANOVA with binary data is introduced by Mai and Zhang (2017). One-way
    ANOVA with binary data is used for comparing means of three or more groups of binary data. Its outcome variable is
    supposed to follow Bernoulli distribution. And its overall test uses a likelihood ratio test statistics.

    Parameters
    ----------
    k: int, default=None
        Number of groups
    n: int, default=None
        Sample size
    V: float, default=None
        Effect size
    alpha: float, default=None
        Significance level of the test
    power: float, default=None
        Statistical power
    print_pretty: bool, default=True
        Whether we want our results printed or not

    Returns
    -------
    A dictionary containing k, n, V, alpha and power
    """
    if not any(v is None for v in [k, n, V, alpha, power]):
        raise ValueError("One of k, n, V, alpha or power must be None")
    if sum([v is None for v in [k, n, V, alpha, power]]) > 1:
        raise ValueError("Only one of k, n, v, alpha or power may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    if n is not None and n < 1:
        raise ValueError("n must be a positive integer")
    if k is not None and k < 1:
        raise ValueError("k must be a positive integer")
    test = WpAnovaBinaryClass(k, n, V, alpha, power).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(test["k"])) - 1)
            + "k"
            + " " * (len(str(test["n"])))
            + "n"
            + " " * (len(str(round(test["effect_size"], 4))))
            + "V"
            + " "
            + "alpha"
            + " "
            + "power"
            + "\n"
            + "\t"
            + f"{test['k']}"
            + " "
            + f"{test['n']}"
            + " "
            + f"{round(test['effect_size'], 4)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " \n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test


def wp_anova_count_test(
    k: Optional[int] = None,
    n: Optional[int] = None,
    V: Optional[float] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    print_pretty: bool = True,
) -> Dict:
    """The power analysis procedure for one-way ANOVA with count data is introduced by Mai and Zhang(2017). One-way
    ANOVA with count data is used for comparing means of three or more groups of binary data. Its outcome variable is
    supposed to follow Poisson distribution. And its overall test uses a likelihood ratio test statistics.

    Parameters
    ----------
    k: int, default=None
        Number of groups
    n: int, default=None
        Sample size
    V: float, default=None
        Effect size
    alpha: float, default=None
        Significance level of the test
    power: float, default=None
        Statistical power
    print_pretty: bool, default=True
        Whether we want our results printed or not

    Returns
    -------
    A dictionary containing k, n, V, alpha and power
    """
    if not any(v is None for v in [k, n, V, alpha, power]):
        raise ValueError("One of k, n, V, alpha or power must be None")
    if sum([v is None for v in [k, n, V, alpha, power]]) > 1:
        raise ValueError("Only one of k, n, v, alpha or power may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    if n is not None and n < 1:
        raise ValueError("n must be a positive integer")
    if k is not None and k < 1:
        raise ValueError("k must be a positive integer")
    test = WpAnovaCountClass(k, n, V, alpha, power).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(test["k"])) - 1)
            + "k"
            + " " * (len(str(test["n"])))
            + "n"
            + " " * (len(str(round(test["effect_size"], 4))))
            + "V"
            + " "
            + "alpha"
            + " "
            + "power"
            + "\n"
            + "\t"
            + f"{test['k']}"
            + " "
            + f"{test['n']}"
            + " "
            + f"{round(test['effect_size'], 4)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " \n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test


def wp_kanova_test(
    n: Optional[int] = None,
    ndf: Optional[int] = None,
    f: Optional[float] = None,
    ng: Optional[int] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    print_pretty: float = True,
) -> Dict:
    """Power analysis for k-way ANOVA.

    Parameters
    ----------
    n: int, default=None
        Sample size
    ndf: int, default=None
        Numerator degrees of freedom
    f: float, default=None
        Effect size
    ng: int, default=None
        Number of groups
    alpha: float, default=None
        Significance level of the test
    power: float, default=None
        Statistical power
    print_pretty: bool, default=True
        Whether we want our results printed or not

    Returns
    -------
    A dictionary containing n, ndf, f, ng, alpha and power
    """
    if not any(v is None for v in [n, ndf, f, ng, alpha, power]):
        raise ValueError("One of n, ndf, f, ng, alpha or power must be None")
    if sum([v is None for v in [n, ndf, f, ng, alpha, power]]) > 1:
        raise ValueError("Only one of n, ndf, f, ng, alpha or power may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    if n is not None and n < 1:
        raise ValueError("n must be a positive integer")
    if ndf is not None and ndf < 1:
        raise ValueError("ndf must be a positive integer")
    if ng is not None and ng < 1:
        raise ValueError("k must be a positive integer")
    test = WpKAnovaClass(n, ndf, f, ng, alpha, power).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(test["n"])) - 1)
            + "n"
            + " " * max(1, len(str(test["ndf"])) - 3)
            + "ndf"
            + " " * max(1, len(str(test["ddf"])) - 3)
            + "ddf"
            + " " * (len(str(round(test["effect_size"], 4))))
            + "f"
            + " " * max(1, len(str(test["ng"])) - 2)
            + "ng"
            + " "
            + "alpha"
            + " "
            + "power"
            + " \n"
            + "\t"
            + f"{test['n']}"
            + " " * max(len(str(test["ndf"])), 4 - len(str(test["ndf"])))
            + f"{test['ndf']}"
            + " " * max(1, 4 - len(str(test["ddf"])))
            + f"{test['ddf']}"
            + " "
            + f"{round(test['effect_size'], 4)}"
            + " " * 2
            + f"{test['ng']}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + "\n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test


def wp_rmanova_test(
    n: Optional[int] = None,
    ng: Optional[int] = None,
    nm: Optional[int] = None,
    f: Optional[float] = None,
    nscor: float = 1,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    test_type: str = "between",
    print_pretty: bool = True,
) -> Dict:
    """Repeated-measures ANOVA can be used to compare the means of a sequence of measurements(e.g., O’brien & Kaiser,
    1985). In a repeated-measures design, evey subject is exposed to all dif-ferent treatments, or more commonly
    measured across different time points. Power analysis for (1)the within-effect test about the mean difference among
    measurements by default. If the subjects arefrom more than one group,the power analysis is also available for (2)
    the between-effect test about mean difference among groups and (3) the interaction effect test of the measurements
    and groups.

    Parameters
    ----------
    n: int, default=None
        Sample size
    ng: int, default=None
        Number of groups
    nm: int, default=None
        Number of measurements
    f: float, default=None
        Effect size. We use the statistic f as the measure of effect size for repeated measures ANOVA
        as in Cohen(1988, p.275).
    nscor: float, default=1
        Nonsphericity correction coefficient. The nonsphericity correction coefficient is a measure of the degree of
        sphericity in the population. A coefficient of 1 means sphericity is met, while a coefficient less than 1 means
        not met. The samller value of the coefficient means the further departure from sphericity. The lowest value of
        the coefficient is 1/(nm-1) where nm is the total number of measurements. Two viable approaches for computing
        the empirical nonsphericity correction coefficient are sggested. One is by Greenhouse and Geisser (1959),
        the other is by Huynh and Feldt (1976).
    alpha: float, default=None
        Significance level of the test
    power: float, default=None
        Statistical power
    test_type: {'between', 'within', 'interaction'}
        Type of analysis.
    print_pretty: bool, default=True
        Whether we want our results printed or not

    Returns
    -------
    A dictionary containing n, ng, nm, f, alpha and power
    """
    if not any(v is None for v in [n, ng, nm, f, alpha, power]):
        raise ValueError("One of n, ng, nm, f, alpha and power must be None")
    if sum([v is None for v in [n, ng, nm, f, alpha, power]]) > 1:
        raise ValueError("Only one of n, ng, nm, f, alpha or power may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    if n is not None and n < 1:
        raise ValueError("n must be a positive integer")
    if ng is not None and ng < 1:
        raise ValueError("ndf must be a positive integer")
    if nm is not None and nm < 1:
        raise ValueError("nm must be a positive integer")
    test_type = test_type.casefold()
    if test_type not in ["between", "within", "interaction"]:
        raise ValueError(f"{test_type} not supported for test_type")
    test = WpRMAnovaClass(n, ng, nm, f, nscor, alpha, power, test_type).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(test["n"])) - 1)
            + "n"
            + " " * (len(str(round(test["effect_size"], 4))))
            + "f"
            + " " * max(1, len(str(test["ng"])) - 1)
            + "ng"
            + " " * max(1, len(str(test["nm"])) - 1)
            + "nm"
            + " "
            + "nscor"
            + " "
            + "alpha"
            + " "
            + "power"
            + "\n"
            + "\t"
            + f"{test['n']}"
            + " "
            + f"{round(test['effect_size'], 4)}"
            + " " * max(1, 3 - len(str(test["ng"])))
            + f"{test['ng']}"
            + " " * max(1, 3 - len(str(test["nm"])))
            + f"{test['nm']}"
            + " " * (6 - len(str(round(test["nscor"], 2))))
            + f"{round(test['nscor'], 2)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " \n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test


def wp_one_prop_test(
    h: Optional[float] = None,
    n: Optional[int] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    alternative: str = "two-sided",
    print_pretty: bool = True,
) -> Dict:
    """Tests of proportions are a technique used to compare proportions of success or agreement in one or two samples.
    The one-sample test of proportion tests the null proportion of success, usually 0.5. The power calculation is based
    on the arcsine transformation of the proportion (see Cohen, 1988, p.548).

    Parameters
    ----------
    h: float, default=None
        Effect size of the proportion comparison. Cohen (1992) suggested that effect size values of 0.2, 0.5, and 0.8
        represent "small", "medium", and "large" effect sizes, respectively.
    n: int, default=None
        The sample size of the group.
    alpha: float, default=None
        Significance level of the test.
    power: float, default=None
        Statistical power.
    alternative: {'two-sided', 'greater', 'less'}
        Direction of the alternative hypothesis.
    print_pretty: bool, default=True
        Whether we want our results printed or not.

    Returns
    -------
    A dictionary containing h, n, alpha, power and our alternative hypothesis
    """
    if not any(v is None for v in [h, n, alpha, power]):
        raise ValueError("One of h, n, alpha and power must be None")
    if sum([v is None for v in [h, n, alpha, power]]) > 1:
        raise ValueError("Only one of h, n, alpha or power may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    if n is not None and n < 1:
        raise ValueError("n must be a positive integer")
    alternative = alternative.casefold()
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError(f"{alternative} not supported for alternative")
    test = WpOneProp(h, n, alpha, power, alternative).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(round(test["effect_size"], 4))) - 1)
            + "h"
            + " " * len(str(test["n"]))
            + "n"
            + " "
            + "alpha"
            + " "
            + "power"
            + "\n"
            + "\t"
            + f"{round(test['effect_size'], 4)}"
            + " "
            + f"{test['n']}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " \n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test


def wp_two_prop_one_n_test(
    h: Optional[float] = None,
    n: Optional[int] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    alternative: str = "two-sided",
    print_pretty: bool = True,
) -> Dict:
    """Tests of proportions are a technique used to compare proportions of success or agreement in one or two samples.
    The two-sample test of proportions tests the null hypothesis that the two samples are drawn from populations with
    the same proportion of success. The power calculation is based on the arcsine transformation of the proportion (see
    Cohen, 1988, p.548).

    Parameters
    ----------
    h: float, default=None
        Effect size of the proportion comparison. Cohen (1992) suggested that effect size values of 0.2, 0.5, and 0.8
        represent "small", "medium", and "large" effect sizes, respectively.
    n: int, default=None
        The sample size of the group.
    alpha: float, default=None
        Significance level of the test.
    power: float, default=None
        Statistical power.
    alternative: {'two-sided', 'greater', 'less'}
        Direction of the alternative hypothesis.
    print_pretty: bool, default=True
        Whether we want our results printed or not.

    Returns
    -------
    A dictionary containing h, n, alpha, power and our alternative hypothesis
    """
    if not any(v is None for v in [h, n, alpha, power]):
        raise ValueError("One of h, n, alpha and power must be None")
    if sum([v is None for v in [h, n, alpha, power]]) > 1:
        raise ValueError("Only one of h, n, alpha or power may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    if n is not None and n < 1:
        raise ValueError("n must be a positive integer")
    alternative = alternative.casefold()
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError(f"{alternative} not supported for alternative")
    test = WpTwoPropOneN(h, n, alpha, power, alternative).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(round(test["effect_size"], 4))) - 1)
            + "h"
            + " " * len(str(test["n"]))
            + "n"
            + " "
            + "alpha"
            + " "
            + "power"
            + "\n"
            + "\t"
            + f"{round(test['effect_size'], 4)}"
            + " "
            + f"{test['n']}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " \n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test


def wp_two_prop_two_n_test(
    h: Optional[float] = None,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    alternative: str = "two-sided",
    print_pretty: bool = True,
) -> Dict:
    """Tests of proportions are a technique used to compare proportions of success or agreement in one or two samples.
    The two-sample test of proportions tests the null hypothesis that the two samples are drawn from populations with
    the same proportion of success. The power calculation is based on the arcsine transformation of the proportion (see
    Cohen, 1988, p.548).

    Parameters
    ----------
    h: float, default=None
        Effect size of the proportion comparison. Cohen (1992) suggested that effect size values of 0.2, 0.5, and 0.8
        represent "small", "medium", and "large" effect sizes, respectively.
    n1: int, default=None
        The sample size of the first group.
    n2: int, default=None
        The sample size of the second group.
    alpha: float, default=None
        Significance level of the test.
    power: float, default=None
        Statistical power.
    alternative: {'two-sided', 'greater', 'less'}
        Direction of the alternative hypothesis.
    print_pretty: bool, default=True
        Whether we want our results printed or not.

    Returns
    -------
    A dictionary containing h, n, alpha, power and our alternative hypothesis
    """
    if not any(v is None for v in [h, n1, n2, alpha, power]):
        raise ValueError("One of h, n, alpha and power must be None")
    if sum([v is None for v in [h, n1, n2, alpha, power]]) > 1:
        raise ValueError("Only one of h, n, alpha or power may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    if n1 is not None and n1 < 2:
        raise ValueError("n1 must be a positive integer greater than 1")
    if n2 is not None and n2 < 2:
        raise ValueError("n2 must be a positive integer greater than 1")
    alternative = alternative.casefold()
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError(f"{alternative} not supported for alternative")
    if alternative.casefold() == "alternative":
        h = abs(h)
    test = WpTwoPropTwoN(h, n1, n2, alpha, power, alternative).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(round(test["effect_size"], 4))) - 1)
            + "h"
            + " " * max(1, (len(str(test["n1"])) - 1))
            + "n1"
            + " " * max(1, (len(str(test["n2"])) - 1))
            + "n2"
            + " "
            + "alpha"
            + " "
            + "power"
            + "\n"
            + "\t"
            + f"{round(test['effect_size'], 4)}"
            + " " * max(1, 3 - len(str(test["n1"])))
            + f"{test['n1']}"
            + " " * max(1, 3 - len(str(test["n2"])))
            + f"{test['n2']}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " \n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test


def wp_t1_test(
    n: Optional[int] = None,
    d: Optional[float] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    test_type: str = "two-sample",
    alternative: str = "two-sided",
    print_pretty: bool = True,
) -> Dict:
    """A t-test is a statistical hypothesis test in which the test statistic follows a Student’s t distribution if the
    null hypothesis is true and follows a non-central t distribution if the alternative hypothesis is true. The t test
    can assess the statistical significance of (1) the difference between population mean and a specific value, (2) the
    difference between two independent population means, and (3) difference between means of matched pairs.

    Parameters
    ----------
    n: int, default=None
        If test_type='one-sample', then the sample size of our group; otherwise the sample size of both groups.
    d: float, default=None
        Effect size
    alpha: float, default=None
        Significance level of the test.
    power: float, default=None
        Statistical power.
    alternative: {'two-sided', 'greater', 'less'}
        Direction of the alternative hypothesis.
    test_type: {'two-sample', 'paired', 'one-sample'}
        Whether our test is a two-sample test, a paired test or a one-sample test.
    print_pretty: bool, default=True
        Whether we want our results printed or not.

    Returns
    -------
    A dictionary containing n, d, alpha, power and our alternative hypothesis
    """
    if not any(x is None for x in [n, d, alpha, power]):
        raise ValueError("One of n, d, alpha or power must be None")
    if sum([x is None for x in [n, d, alpha, power]]) > 1:
        raise ValueError("Only one of n, d, alpha or power may be None")
    if n is not None and n < 2:
        raise ValueError("Number of observations must be at least 2")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    test_type = test_type.casefold()
    if test_type not in ("two-sample", "one-sample", "paired"):
        raise ValueError(f"{test_type} not supported for a t-test")
    alternative = alternative.casefold()
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError(f"{alternative} not supported for alternative")
    test = WpOneT(n, d, alpha, power, test_type, alternative).pwr_test()
    if print_pretty:
        if "note" in test:
            print(
                f"{test['method']}"
                + "\n" * 2
                + "\t"
                + " " * len(str(test["n"]))
                + "n"
                + " " * (len(str(round(test["effect_size"], 4))) - 1)
                + "h"
                + " "
                + "alpha"
                + " "
                + "power"
                + "\n"
                + "\t"
                + f"{test['n']}"
                + " "
                + f"{round(test['effect_size'], 4)}"
                + " " * (6 - len(str(round(test["alpha"], 2))))
                + f"{round(test['alpha'], 2)}"
                + " " * (6 - len(str(round(test["power"], 2))))
                + f"{round(test['power'], 2)}"
                + " \n" * 2
                + f"Note: {test['note']}"
                + "\n"
                + f"URL: {test['url']}"
            )
        else:
            print(
                f"{test['method']}"
                + "\n" * 2
                + "\t"
                + " " * (len(str(test["n"])) - 1)
                + "n"
                + " " * (len(str(round(test["effect_size"], 4))))
                + "h"
                + " "
                + "alpha"
                + " "
                + "power"
                + "\n"
                + "\t"
                + f"{test['n']}"
                + " "
                + f"{round(test['effect_size'], 4)}"
                + " " * (6 - len(str(round(test["alpha"], 2))))
                + f"{round(test['alpha'], 2)}"
                + " " * (6 - len(str(round(test["power"], 2))))
                + f"{round(test['power'], 2)}"
                + " \n" * 2
                + f"URL: {test['url']}"
            )
    return test


def wp_t2_test(
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    d: Optional[float] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    alternative: str = "two-sided",
    print_pretty: bool = True,
) -> Dict:
    """A t-test is a statistical hypothesis test in which the test statistic follows a Student’s t distribution if the
    null hypothesis is true and follows a non-central t distribution if the alternative hypothesis is true. The t test
    can assess the statistical significance of (1) the difference between population mean and a specific value, (2) the
    difference between two independent population means, and (3) difference between means of matched pairs.

    Parameters
    ----------
    n1: int, default=None
        The sample size of our first group
    n2: int, default=None
        The sample size of our second group
    d: float, default=None
        Effect size
    alpha: float, default=None
        Significance level of the test.
    power: float, default=None
        Statistical power.
    alternative: {'two-sided', 'greater', 'less'}
        Direction of the alternative hypothesis.
    print_pretty: bool, default=True
        Whether we want our results printed or not.

    Returns
    -------
    A dictionary containing n1, n2, d, alpha, power and our alternative hypothesis
    """
    if not any(x is None for x in [n1, n2, d, alpha, power]):
        raise ValueError("One of n1, n2, d, alpha or power must be None")
    if sum([x is None for x in [n1, n2, d, alpha, power]]) > 1:
        raise ValueError("Only one of n1, n2, d, alpha or power may be None")
    if n1 is not None and n1 < 2:
        raise ValueError("Number of observations for the first group must be at least 2")
    if n2 is not None and n2 < 2:
        raise ValueError("Number of observations for the second group must be at least 2")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    alternative = alternative.casefold()
    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError(f"{alternative} not supported for alternative")
    test = WpTwoT(n1, n2, d, alpha, power, alternative).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * max(0, (len(str(test["n1"])) - 2))
            + "n1"
            + " " * max(1, (len(str(test["n2"])) - 1))
            + "n2"
            + " " * (len(str(round(test["effect_size"], 4))))
            + "d"
            + " "
            + "alpha"
            + " "
            + "power"
            + "\n"
            + "\t"
            + f"{test['n1']}"
            + " " * max(1, 3 - len(str(test["n2"])))
            + f"{test['n2']}"
            + " "
            + f"{round(test['effect_size'], 4)}"
            + " "
            + " " * (5 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " \n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test


def wp_regression_test(
    n: Optional[int] = None,
    p1: int = 1,
    p2: int = 0,
    f2: Optional[float] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    test_type: str = "regular",
    print_pretty: bool = True,
) -> Dict:
    """This function is for power analysis for regression models. Regression is a statistical technique for examining
    the relationship between one or more independent variables (or predictors) and one dependent variable (or the
    outcome). Regression provides an F-statistic that can be formulated using the ratio between variation in the outcome
    variable that is explained by the predictors and the unexplained variation (Cohen, 1988)). The test statistic can
    also be expressed in terms of comparison between Full and Reduced models (Maxwell & Delaney, 2003).

    Parameters
    ----------
    n: int, default=None
        Sample size.
    p1: int, default=1
        Number of predictors in the full model.
    p2: int, default=0
        Number of predictors in the reduced model, it is 0 by default. See the book by Maxwell and Delaney (2003)
        for the definition of the reduced model.
    f2: float, default=None
        Effect size. We use the statistic f2 as the measure of effect size for linear regression proposed by Cohen(1988, p.410).
         Cohen discussed the effect size in three different cases. The calculation of f2 can be generalized using the
         idea of a full model and a reduced model by Maxwell and Delaney (2003).
    alpha: float, default=None
        Significance level chosen for the test.
    power: float, default=None
        Statistical power.
    test_type: {"cohen", "regular"}
        If set to "cohen", the formula used in the Cohen’s book will be used (not recommended).
    print_pretty: bool, default=True
        Whether we want our results printed or not.

    Returns
    -------
    A dictionary containing n, p1, p2, f2, alpha and the power of our test
    """
    if not any(x is None for x in [n, f2, alpha, power]):
        raise ValueError("One of n, f2, alpha or power must be None")
    if sum([x is None for x in [n, f2, alpha, power]]) > 1:
        raise ValueError("Only one of n, f2, alpha or power may be None")
    if p1 < p2:
        raise ValueError("Number of predictors in the full model has to be larger than that in the reduced model")
    if p1 < 1:
        raise ValueError("Number of predictors in the full model has to be at least 1")
    if n is not None and n < 5:
        raise ValueError("Sample size must be at least 5")
    if f2 is not None and f2 < 0:
        raise ValueError("f2 must be positive")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    test = WPRegression(n, p1, p2, f2, alpha, power, test_type).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * max(0, (len(str(test["n"])) - 1))
            + "n"
            + " " * max(1, (len(str(test["p1"])) - 1))
            + "p1"
            + " " * max(1, (len(str(test["p2"])) - 1))
            + "p2"
            + " " * (len(str(round(test["effect_size"], 4))) - 1)
            + "f2"
            + " "
            + "alpha"
            + " "
            + "power"
            + "\n"
            + "\t"
            + f"{test['n']}"
            + " " * max(1, 3 - len(str(test["p1"])))
            + f"{test['p1']}"
            + " " * max(1, 3 - len(str(test["p2"])))
            + f"{test['p2']}"
            + " "
            + f"{round(test['effect_size'], 4)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " \n" * 2
            + f"URL: {test['url']}"
        )
    return test


def wp_poisson_test(
    n: Optional[int] = None,
    exp0: float = 1,
    exp1: float = 0.5,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    alternative: str = "two-sided",
    family: str = "Bernoulli",
    parameter: Optional[Union[int, float, list, tuple]] = None,
    print_pretty: bool = True,
) -> Dict:
    """This function is for Poisson regression models. Poisson regression is a type of generalized linear models where
    the outcomes are usually count data. Here, Maximum likelihood methods is used to estimate the model parameters. The
    estimated regression coefficient is assumed to follow a normal distribution. A Wald test is used to test the mean
    difference between the estimated parameter and the null parameter (typically the null hypothesis assumes it equals
    0). The procedure introduced by Demidenko (2007) is adopted here for computing the statistical power.

    Parameters
    ----------
    n: int, default=None
        Sample size
    exp0: float, default=1
        The base rate under the null hypothesis. It always takes positive value. See the article by Demidenko (2007)
        for details.
    exp1: float, default=0.5
        The relative increase of the event rate. It is used for calculation of the effect size. See the article by
        Demidenko (2007) for details.
    alpha: float, default=None
        Significance level of the test.
    power: float, default=None
        Statistical power.
    alternative: {'two-sided', 'greater', 'less'}
        Direction of the alternative hypothesis
    family: {'bernoulli', 'exponential', 'lognormal', 'normal', 'poisson', 'uniform'}
        Distribution of the predictor.
    parameter: float, int or iterable
        Corresponding parameter for the predictor’s distribution. The default is 0.5 for 'bernoulli',
        1 for 'exponential', (0,1) for 'lognormal' or normal, 1 for 'poisson', and (0,1) for 'uniform'.
    print_pretty: bool, default=True
        Whether we want our results printed or not.

    Returns
    -------
    A dict containing n, alpha and power of our test
    """
    if not any(x is None for x in [n, alpha, power]):
        raise ValueError("One of n, alpha or power must be None")
    if sum([x is None for x in [n, alpha, power]]) > 1:
        raise ValueError("Only one of n, alpha or power may be None")
    if exp0 <= 0:
        raise ValueError("exp0 cannot be less than or equal to 0")
    if exp1 <= 0:
        raise ValueError("exp1 cannot be less than or equal to 0")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    test = WpPoisson(n, exp0, exp1, alpha, power, alternative, family, parameter).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * max(0, (len(str(test["n"])) - 1))
            + "n"
            + " "
            + "power"
            + " "
            + "alpha"
            + " " * max(1, 4 - len(str(round(test["exp0"], 2))))
            + "exp0"
            + " " * max(1, 4 - len(str(round(test["exp1"], 2))))
            + "exp1"
            + " " * max(1, 5 - len(str(round(test["beta0"], 2))))
            + "beta0"
            + " " * max(1, 5 - len(str(round(test["beta1"], 2))))
            + "beta1"
            + "\n"
            + "\t"
            + f"{test['n']}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " " * (5 - len(str(round(test["exp0"], 2))))
            + f"{round(test['exp0'], 2)}"
            + " " * (5 - len(str(round(test["exp1"], 2))))
            + f"{round(test['exp1'], 2)}"
            + " " * (6 - len(str(round(test["beta0"], 2))))
            + f"{round(test['beta0'], 2)}"
            + " " * (6 - len(str(round(test["beta1"], 2))))
            + f"{round(test['beta1'], 2)}"
            + " \n" * 2
            + f"URL: {test['url']}"
        )
    return test


def wp_logistic_test(
    n: Optional[int] = None,
    p0: float = 0.5,
    p1: float = 0.5,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    alternative: str = "two-sided",
    family: str = "Bernoulli",
    parameter: Optional[Union[int, float, list, tuple]] = None,
    print_pretty: bool = True,
) -> Dict:
    """This function is for Logistic regression models. Logistic regression is a type of generalized linear models where
    the outcome variable follows Bernoulli distribution. Here, Maximum likelihood methods is used to estimate the model
    parameters. The estimated regression coefficient is assumed to follow a normal distribution. A Wald test is use to
    test the mean difference between the estimated parameter and the null parameter (typically the null hypothesis
    assumes it equals 0). The procedure introduced by Demidenko (2007) is adopted here for computing the statistical
    power.

    Parameters
    ----------
    n: int, default=None
        Sample size
    p0: float, default=0.5
        Prob(Y=1|X=0): the probability of observing 1 for the outcome variable Y when the predictor X equals 0.
    p1: float, default=0.5
        Prob(Y=1|X=1): the probability of observing 1 for the outcome variable Y when the predictor X equals 1.
    alpha: float, default=None
        Significance level chosen for the test.
    power: float, default=None
        Statistical power.
    alternative: {'two-sided', 'greater', 'less'}
        Direction of the alternative hypothesis
    family: {'bernoulli', 'exponential', 'lognormal', 'normal', 'poisson', 'uniform'}
        Distribution of the predictor
    parameter: float, int or iterable
        Corresponding parameter for the predictor’s distribution. The default is 0.5 for 'bernoulli',
        1 for 'exponential', (0,1) for 'lognormal' or normal, 1 for 'poisson', and (0,1) for 'uniform'.
    print_pretty: bool, default=True
        Whether we want our results printed or not.

    Returns
    -------
    A dict containing n, alpha and the power of our test
    """
    if not any(x is None for x in [n, alpha, power]):
        raise ValueError("One of n, alpha or power must be None")
    if sum([x is None for x in [n, alpha, power]]) > 1:
        raise ValueError("Only one of n, alpha or power may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    test = WpLogistic(n, p0, p1, alpha, power, alternative, family, parameter).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * 2
            + "p0"
            + " " * 3
            + "p1"
            + " " * max(1, (len(str(round(test["beta0"], 4))) - 4))
            + "beta0"
            + " " * max(1, (len(str(round(test["beta1"], 4))) - 4))
            + "beta1"
            + " " * (len(str(test["n"])))
            + "n"
            + " "
            + "power"
            + " "
            + "alpha"
            + "\n"
            + "\t"
            + f"{'{0:.2f}'.format(test['p0'])}"
            + " "
            + f"{'{0:.2f}'.format(test['p1'])}"
            + " "
            + f"{round(test['beta0'], 4)}"
            + " "
            + f"{round(test['beta1'], 4)}"
            + " "
            + f"{test['n']}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " \n" * 2
            + f"URL: {test['url']}"
        )
    return test


def wp_sem_chisq_test(
    n: Optional[int] = None,
    df: Optional[int] = None,
    effect: Optional[float] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    print_pretty: bool = True,
) -> Dict:
    """Structural equation modeling (SEM) is a multivariate technique used to analyze relationships among observed and
    latent variables. It can be viewed as a combination of factor analysis and multivariate regression analysis. Two
    methods are widely used in power analysis for SEM. One is based on the likelihood ratio test proposed by Satorra and
    Saris (1985). The other is based on RMSEA proposed by MacCallum et al. (1996). This function is for SEM power
    analysis based on the likelihood ratio test.

    Parameters
    ----------
    n: int, default=None
        Sample size
    df: int, default=None
        The degrees of freedom for our test, based on the Chi-Squared Test.
    effect: float, default=None
        Effect size. It specifies the population misfit of a SEM model, which is the difference between two SEM models:
        a full model (Mf) and a reduced model (Mr). A convienient way to get the effect size is to fit the reduced model
        using SEM software such R package ’lavaan’ (Rossel, 2012). Then the effect size is calculated as the
        chi-squared statistics dividing by the sample size.
    alpha: float, default=None
        Significance level chosen for the test.
    power: float, default=None
        Statistical power.
    print_pretty: bool, default=True
        Whether we want our results printed or not.

    Returns
    -------
    A dictionary containing n, df, effect, power and alpha of our test
    """
    if not any(x is None for x in [n, df, effect, power, alpha]):
        raise ValueError("One of n, df, effect, power or alpha must be None")
    if sum([x is None for x in [n, df, effect, power, alpha]]) > 1:
        raise ValueError("Only one of n, df, effect, power or alpha may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    test = WPSEMChisq(n, df, effect, alpha, power).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(test["n"])) - 1)
            + "n"
            + " " * max(1, (len(str(test["df"])) - 2))
            + "df"
            + " "
            + "effect"
            + " "
            + "power"
            + " "
            + "alpha"
            + "\n"
            + "\t"
            + f"{test['n']}"
            + " " * (3 - len(str(test["df"])))
            + f"{test['df']}"
            + " " * (7 - len(str(round(test["effect_size"], 4))))
            + f"{round(test['effect_size'], 4)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " \n" * 2
            + f"URL: {test['url']}"
        )
    return test


def wp_sem_rmsea_test(
    n: Optional[int] = None,
    df: Optional[int] = None,
    rmsea0: Optional[float] = None,
    rmsea1: Optional[float] = None,
    alpha: Optional[float] = None,
    power: Optional[float] = None,
    test_type: str = "close",
    print_pretty: bool = True,
) -> Dict:
    """Structural equation modeling (SEM) is a multivariate technique used to analyze relationships among observed and
    latent variables. It can be viewed as a combination of factor analysis and multivariate regression analysis. Two
    methods are widely used in power analysis for SEM. One is based on the likelihood ratio test proposed by Satorra and
    Saris (1985). The other is based on RMSEA proposed by MacCallum et al. (1996). This function is for SEM power
    analysis based on RMSEA.

    Parameters
    ----------
    n: int, default=None
        Sample size
    df: int, default=None
        The degrees of freedom for our test, based on the Chi-Squared Test.
    rmsea0: float, default=None
        The RMSE for H0. It usually equals 0.
    rmsea1: float, default=None
        The RMSE for H1
    alpha: float, default=None
        Significance level chosen for the test.
    power: float, default=None
        Statistical power.
    test_type: {'close' , 'notclose'}
        Close fit or not-close fit.
    print_pretty: bool, default=True
        Whether we want our results printed or not.

    Returns
    -------
    A dictionary containing n, df, rmsea0, rmsea1, alpha and the power of the test
    """
    if not any(x is None for x in [n, df, rmsea0, rmsea1, power, alpha]):
        raise ValueError("One of n, df, rmsea0, rmsea1, power or alpha must be None")
    if sum([x is None for x in [n, df, rmsea0, rmsea1, power, alpha]]) > 1:
        raise ValueError("Only one of n, df, rmsea0, rmsea1, power or alpha may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    if test_type.casefold() not in ("close", "notclose"):
        raise ValueError(f"{test_type} must be either close or notclose")
    test = WPSEMRMSEA(n, df, rmsea0, rmsea1, power, alpha, test_type).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(test["n"])) - 1)
            + "n"
            + " " * max(1, (len(str(test["df"])) - 2))
            + "df"
            + " "
            + "rmsea0"
            + " "
            + "rmsea1"
            + " "
            + "power"
            + " "
            + "alpha"
            + "\n"
            + "\t"
            + f"{test['n']}"
            + " " * (3 - len(str(test["df"])))
            + f"{test['df']}"
            + " " * (7 - len(str(round(test["rmsea0"], 4))))
            + f"{round(test['rmsea0'], 4)}"
            + " " * (7 - len(str(round(test["rmsea1"], 4))))
            + f"{round(test['rmsea1'], 4)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " \n" * 2
            + f"URL: {test['url']}"
        )
    return test


def wp_mediation_test(
    n: Optional[int] = None,
    power: Optional[float] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
    var_x: float = 1,
    var_y: Optional[float] = None,
    var_m: float = 1,
    alpha: Optional[float] = None,
    print_pretty: bool = True,
) -> Dict:
    """This function is for mediation models. Mediation models can be used to investigate the underlying mechanisms
    related to why an input variable x influences an output variable y (e.g., Hayes, 2013; MacKinnon, 2008). The
    mediation effect is calculated as a*b, where a is the path coefficent from the predictor x to the mediator m, and b
    is the path coefficent from the mediator m to the outcome variable y. Sobel test statistic (Sobel, 1982) is used to
    test whether the mediation effect is significantly different from zero.

    Parameters
    ----------
    n: int, default=None
        Sample size
    power: float, default=None
        Statistical power
    a: float, default=None
        Coefficient from x to m
    b: float, default=None
        Coefficient from m to y
    var_x: float, default=None
        Variance of x
    var_y: float, default=None
        Variance of y
    var_m: float, default=None
        Variance of m
    alpha: float, default=None
        Significance level chosen for the test
    print_pretty: bool, default=True
        Whether we want our results printed or not

    Returns
    -------
    A dictionary containing n, a, b, var_x, var_y, var_m, alpha and the power of the test
    """
    if not any(x is None for x in [n, a, b, var_x, var_y, var_m, power, alpha]):
        raise ValueError("One of n, a, b, var_x, var_y, var_m, power or alpha must be None")
    if sum([x is None for x in [n, a, b, var_x, var_y, var_m, power, alpha]]) > 1:
        raise ValueError("Only one of n, a, b, var_x, var_y, var_m, power or alpha may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    test = WpMediation(n, power, a, b, var_x, var_y, var_m, alpha).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(test["n"])) - 1)
            + "n"
            + " "
            + "power"
            + " " * (len(str(round(test["a"], 2))))
            + "a"
            + " " * (len(str(round(test["b"], 2))))
            + "b"
            + " " * max(len(str(round(test["var_x"], 2))) - 6, 1)
            + "var_x"
            + " " * max(len(str(round(test["var_m"], 2))) - 6, 1)
            + "var_m"
            + " " * max(len(str(round(test["var_y"], 2))) - 6, 1)
            + "var_y"
            + " "
            + "alpha"
            + "\n"
            + "\t"
            + f"{test['n']}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " "
            + f"{round(test['a'], 2)}"
            + " "
            + f"{round(test['b'], 2)}"
            + " " * (6 - len(str(round(test["var_x"], 2))))
            + f"{round(test['var_x'], 2)}"
            + " " * (6 - len(str(round(test["var_m"], 2))))
            + f"{round(test['var_m'], 2)}"
            + " " * (6 - len(str(round(test["var_y"], 2))))
            + f"{round(test['var_y'], 2)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " \n" * 2
            + f"URL: {test['url']}"
        )
    return test


def wp_correlation_test(
    n: Optional[int] = None,
    r: Optional[float] = None,
    power: Optional[float] = None,
    p: int = 0,
    rho0: float = 0.0,
    alpha: Optional[float] = None,
    alternative: str = "two-sided",
    print_pretty: bool = True,
) -> Dict:
    """This function is for power analysis for correlation. Correlation measures whether and how a pair of variables are
    related. The Pearson Product Moment correlation coefficient (r) is adopted here. The power calculation for
    correlation is conducted based on Fisher’s z transformation of Pearson correlation coefficent (Fisher, 1915, 1921).

    Parameters
    ----------
    n: int, default=None
        Sample size
    r: float, default=None
        Effect size or correlation. According to Cohen (1988), a correlation coefficient of 0.10, 0.30, and 0.50 are
        considered as an effect size of "small", "medium", and "large", respectively.
    power: float, default=None
        Statistical power
    p: int, default=0
        Number of variables to partial out
    rho0: float, default=0.0
        Null correlation coefficient
    alpha: float, default=None
        Significance level chosen for the test
    alternative: {'two-sided', 'greater', 'less'}
        Direction of the alternative hypothesis.
    print_pretty: bool, default=True
        Whether we want our results printed or not

    Returns
    -------
    A dictionary containing n, r, power, p, rho0, alpha and the alternative hypothesis of the test
    """
    if not any(x is None for x in [n, r, power, alpha]):
        raise ValueError("One of n, r, power or alpha must be None")
    if sum([x is None for x in [n, r, power, alpha]]) > 1:
        raise ValueError("Only one of n, r, power or alpha may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    test = WpCorrelation(n, r, power, p, rho0, alpha, alternative).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(test["n"])) - 1)
            + "n"
            + " " * (len(str(round(test["effect_size"], 2))))
            + "r"
            + " "
            + "alpha"
            + " "
            + "power"
            + "\n"
            + "\t"
            + f"{test['n']}"
            + " "
            + f"{round(test['effect_size'], 2)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " \n" * 2
            + f"URL: {test['url']}"
        )
    return test


def wp_mrt2arm_test(
    n: Optional[int] = None,
    f: Optional[float] = None,
    J: Optional[int] = None,
    tau00: float = 1.0,
    tau11: float = 1.0,
    sg2: float = 1.0,
    power: Optional[float] = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    test_type: str = "main",
    print_pretty: bool = True,
) -> Dict:
    """Multisite randomized trials (MRT) are a type of multilevel design for the situation when the entire cluster is
    randomly assigned to either a treatment arm or a control arm (Liu, 2013). The data from MRT can be analyzed in a
    two-level hierarchical linear model, where the indicator variable for treatment assignment is included in first
    level. If a study contains multiple treatments, then multiple indicators will be used. This function is for designs
    with.

    2 arms (i.e., a treatment and a control). Three types of tests are considered in the function:
        * (1) The "main" type tests treatment main effect;
        * (2) The "site" type tests the variance of cluster/site means;
        * and (3) The "variance" type tests variance of treatment effects.
    Details leading to power calculation can be found in Raudenbush (1997) and Liu (2013).

    Parameters
    ----------
    n: int, default=None
        Sample size. It is the number of individuals within each cluster.
    f: float, default=None
        Effect size. It specifies the main effect of treatment, the mean difference between the treatment clusters/sites
        and the control clusters/sites. Effect size must be positive.
    J: int, default=None
        Number of clusters / sites. It tells how many clusters are considered in the study design. At least two clusters
        are required.
    tau00: float, default=1.0
        Variance of cluster/site means. It is one of the residual variances in the second level. Its value must be positive.
    tau11: float, default=1.0
        Variance of treatment effects across sites. It is one of the residual variances in the second level. Its value
        must be positive.
    sg2: float, default=1.0
        Level-one error Variance. The residual variance in the first level.
    power: float, default=None
        Statistical power
    alpha: float, default=0.05
        Significance level chosen for the test.
    alternative: str, default='two-sided'
        Type of alternative hypothesis. The option 'one-sided' can be either 'less' or 'greater'
    test_type: {'main', 'site', 'variance'}
        Type of effect. The type "main" tests treatment main effect, no tau00 needed; Type "site" tests the variance of
        cluster/site means, no tau11 or f needed; and Type "variance" tests variance of treatment effects, no tau00 or
        f needed.
    print_pretty: bool, default=True
        Whether we want our results printed or not

    Returns
    -------
    A dictionary containing n, J, f, power and alpha of our test
    """
    test = WpMRT2Arm(n, f, J, tau00, tau11, sg2, power, alpha, alternative, test_type).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + "\n"
            + "\t"
            + " \n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test


def wp_mrt3arm_test(
    n: Optional[int] = None,
    f1: Optional[float] = None,
    f2: float = 0.0,
    J: Optional[int] = None,
    tau: float = 1.0,
    sg2: float = 1.0,
    power: Optional[float] = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    test_type: str = "main",
    print_pretty: bool = True,
) -> Dict:
    """Multisite randomized trials (MRT) are a type of multilevel design for the situation when the entire cluster is
    randomly assigned to either a treatment arm or a control arm (Liu, 2013). The data from MRT can be analyzed in a
    two-level hierarchical linear model, where the indicator variable for treatment assignment is included in first
    level. If a study contains multiple treatments, then multiple indicators will be used. This function is for designs
    with 3.

    arms (i.e., two treatments and a control). Three types of tests are considered in the function:
        * (1) The "main" type tests treatment main effect;
        * (2) The "treatment" type tests the difference between the two treatments;
        * and (3) The "omnibus" type tests whether the three arms are all equivalent.
    Details leading to power calculation can befound in Raudenbush (1997) and Liu (2013).

    Parameters
    ----------
    n: int, default=None
        Sample size. It is the number of individuals within each cluster.
    f1: float, default=None
        Effect size for treatment main effect. Effect size must be positive.
    f2: float, default=0.0
        Effect size for the difference between two treatments. Effect size must be positive.
    J: int, default=None
        Number of clusters / sites. It tells how many clusters are considered in the study design. At least two clusters
        are required.
    tau: float, default=1.0
        Variance of treatment effects across sites/clusters.
    sg2: float, default=1.0
        Level-one error Variance. The residual variance in the first level.
    power: float, default=None
        Statistical power
    alpha: float, default=0.05
        Significance level chosed for the test. It equals 0.05 by default.
    alternative: {'two-sided', 'one-sided'}
        Type of the alternative hypothesis. The option "one-sided" can be either "less" or "greater"
    test_type: {'main', 'treatment', 'omnibus'}
        The type "main" tests the difference between the average treatment arms and the control arm;
        Type "treatment" tests the difference between the two treatment arms;
        and Type "omnibus" tests whether the three arms are all equivalent.
    print_pretty: bool, default=True
        Whether we want our results printed or not

    Returns
    -------
    A dictionary containing n, r, power, p, rho0, alpha and the alternative hypothesis of the test
    """
    if not any(x is None for x in [n, f1, J, power]):
        raise ValueError("One of n, f1, J, or power must be None")
    if sum([x is None for x in [n, f1, J, power]]) > 1:
        raise ValueError("Only one of n, f1, J, or power may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    if f1 is not None and f1 < 0:
        raise ValueError("f1 must be positive")
    if f2 is not None and f2 < 0:
        raise ValueError("f2 must be positive")
    if n is not None and n < 3:
        raise ValueError("n must be greater than 2")
    if J is not None and J < 2:
        raise ValueError("Number of sites must be at least 2")
    if tau < 0:
        raise ValueError("Variance of treatment main effects across sites must be positive")
    if sg2 < 0:
        raise ValueError("Between-person variation must be a positive number")
    if alternative.casefold() not in ["two-sided", "one-sided"]:
        raise ValueError("alternative must be `two-sided` or `one-sided`")
    if test_type.casefold() not in ["main", "treatment", "omnibus"]:
        raise ValueError("test_type must be `main`, `treatment` or `omnibus`")
    test = WpMRT3Arm(n, f1, f2, J, tau, sg2, power, alpha, alternative, test_type).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(test["J"])) - 1)
            + "J"
            + " " * (len(str(test["n"])))
            + "n"
            + " " * max(1, len(str(round(test["f1"], 2))) - 1)
            + "f1"
            + " " * max(1, len(str(round(test["f2"], 2))) - 1)
            + "f2"
            + " " * max(1, len(str(round(test["tau"], 2))) - 2)
            + "tau"
            + " " * max(1, len(str(round(test["sg2"], 2))) - 2)
            + "sg2"
            + " "
            + "power"
            + " "
            + "alpha"
            + "\n"
            + "\t"
            + f"{test['J']}"
            + " "
            + f"{test['n']}"
            + " " * max(3 - len(str(round(test["f1"], 2))), 1)
            + f"{round(test['f1'], 2)}"
            + " " * max(3 - len(str(round(test["f2"], 2))), 1)
            + f"{round(test['f2'], 2)}"
            + " " * max(4 - len(str(round(test["tau"], 2))), 1)
            + f"{round(test['tau'], 2)}"
            + " " * max(4 - len(str(round(test["sg2"], 2))), 1)
            + f"{round(test['sg2'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " \n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test


def wp_crt2arm_test(
    n: Optional[int] = None,
    f: Optional[float] = None,
    J: Optional[int] = None,
    icc: Optional[float] = None,
    power: Optional[float] = None,
    alpha: Optional[float] = None,
    alternative: str = "two-sided",
    print_pretty: bool = True,
) -> Dict:
    """Cluster randomized trials (CRT) are a type of multilevel design for the situation when the entire cluster is
    randomly assigned to either a treatment arm or a contral arm (Liu, 2013). The data from CRT can be analyzed in a
    two-level hierachical linear model, where the indicator variable for treatment assignment is included in second
    level. If a study contains multiple treatments, then mutiple indicators will be used. This function is for designs
    with 2 arms (i.e., a treatment and a control). Details leading to power calculation can be found in Raudenbush
    (1997) and Liu (2013).

    Parameters
    ----------
    n: int, default=None
        Sample size. It is the number of individuals within each cluster
    f: float, default=None
        Effect size. It specifies either the main effect of treatment, or the mean difference between the treatment
        clusters and the control clusters.
    J: int, default=None
        Number of clusters / sides. It tells how many clusters are considered in the study design. At least 2 clusters
        are required.
    icc: float, default=None
        Intra-class correlation. ICC is calculated as the ratio of betwee-cluster variance to the total variance. It
        quantifies the degree to which two randomly drawn observations within a cluster are correlated.
    power: float, default=None
        Statistical power
    alpha: float, default=None
        Significance level chosen for the test.
    alternative: {"two-sided", "one-sided"}
        Type of alternative hypothesis. The option "one-sided" can be either "less" or "greater"
    print_pretty: bool, default=True
        Whether we want our results printed or not

    Returns
    -------
    A dictionary containing n, effect size, J, icc, power and alpha of our test
    """
    if not any(x is None for x in [n, f, J, icc, power, alpha]):
        raise ValueError("One of n, f, J, icc, power, or alpha must be None")
    if sum([x is None for x in [n, f, J, icc, power, alpha]]) > 1:
        raise ValueError("Only one of n, f, J, icc, power, or alpha may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    if n is not None and n < 1:
        raise ValueError("n must be at least 1")
    if J is not None and J < 3:
        raise ValueError("J must be at least 3")
    if icc is not None and (icc < 0 or icc > 1):
        raise ValueError("icc must be between 0 and 1")
    if alternative.casefold() not in ["two-sided", "one-sided"]:
        raise ValueError("alternative must be one of `two-sided` or `one-sided`")
    test = WpCRT2Arm(n, f, J, icc, power, alpha, alternative).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(test["J"])) - 1)
            + "J"
            + " " * (len(str(test["n"])))
            + "n"
            + " " * max(1, len(str(round(test["effect_size"], 2))))
            + "f"
            + " " * max(1, len(str(round(test["icc"], 2))) - 2)
            + "icc"
            + " "
            + "power"
            + " "
            + "alpha"
            + "\n"
            + "\t"
            + f"{test['J']}"
            + " "
            + f"{test['n']}"
            + " " * max(3 - len(str(round(test["effect_size"], 2))), 1)
            + f"{round(test['effect_size'], 2)}"
            + " " * max(3 - len(str(round(test["icc"], 2))), 1)
            + f"{round(test['icc'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " \n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test


def wp_crt3arm_test(
    n: Optional[int] = None,
    f: Optional[float] = None,
    J: Optional[int] = None,
    icc: Optional[float] = None,
    power: Optional[float] = None,
    alpha: Optional[float] = None,
    alternative: str = "two-sided",
    test_type: str = "main",
    print_pretty: bool = True,
) -> Dict:
    """Cluster randomized trials (CRT) are a type of multilevel design for the situation when the entire cluster is
    randomly assigned to either a treatment arm or a contral arm (Liu, 2013). The data from CRT can be analyzed in a
    two-level hierachical linear model, where the indicator variable for treatment assignment is included in second
    level. If a study contains multiple treatments, then mutiple indicators will be used. This function is for designs
    with 3 arms (i.e., two treatments and a control). Details leading to power calculation can be found in Raudenbush
    (1997) and Liu (2013).

    Parameters
    ----------
    n: int, default=None
        Sample size. It is the number of individuals within each cluster
    f: float, default=None
        Effect size. It specifies either the main effect of treatment, or the mean difference between the treatment
        clusters and the control clusters.
    J: int, default=None
        Number of clusters / sides. It tells how many clusters are considered in the study design. At least 2 clusters
        are required.
    icc: float, default=None
        Intra-class correlation. ICC is calculated as the ratio of betwee-cluster variance to the total variance. It
        quantifies the degree to which two randomly drawn observations within a cluster are correlated.
    power: float, default=None
        Statistical power
    alpha: float, default=None
        Significance level chosen for the test.
    alternative: {"two-sided", "one-sided"}
        Type of alternative hypothesis. The option "one-sided" can be either "less" or "greater"
    test_type: {"main", "treatment", "omnibus"}
        Type of effect.
            * "main" tests the difference between the average treatment arms and the control arm.
            * "treatment" tests the difference between the two treatment arms
            * "omnibus" test whether the tree arms are all equivalent
    print_pretty: bool, default=True
        Whether we want our results printed or not

    Returns
    -------
    A dictionary containing n, effect size, J, icc, power and alpha of our test
    """
    if not any(x is None for x in [n, f, J, icc, power, alpha]):
        raise ValueError("One of n, f, J, icc, power, or alpha must be None")
    if sum([x is None for x in [n, f, J, icc, power, alpha]]) > 1:
        raise ValueError("Only one of n, f, J, icc, power, or alpha may be None")
    if alpha is not None and (alpha < 0 or alpha > 1):
        raise ValueError("alpha must be between 0 and 1")
    if power is not None and (power < 0 or power > 1):
        raise ValueError("power must be between 0 and 1")
    if n is not None and n < 1:
        raise ValueError("n must be at least 1")
    if J is not None and J < 3:
        raise ValueError("J must be at least 3")
    if icc is not None and (icc < 0 or icc > 1):
        raise ValueError("icc must be between 0 and 1")
    if alternative.casefold() not in ["two-sided", "one-sided"]:
        raise ValueError("alternative must be one of `two-sided` or `one-sided`")
    if test_type.casefold() not in ["main", "treatment", "omnibus"]:
        raise ValueError("test_type must be one of `main`, `treatment` or `omnibus`")
    test = WpCRT3Arm(n, f, J, icc, power, alpha, alternative, test_type).pwr_test()
    if print_pretty:
        print(
            f"{test['method']}"
            + "\n" * 2
            + "\t"
            + " " * (len(str(test["J"])) - 1)
            + "J"
            + " " * (len(str(test["n"])))
            + "n"
            + " " * max(1, len(str(round(test["effect_size"], 2))))
            + "f"
            + " " * max(1, len(str(round(test["icc"], 2))) - 2)
            + "icc"
            + " "
            + "power"
            + " "
            + "alpha"
            + "\n"
            + "\t"
            + f"{test['J']}"
            + " "
            + f"{test['n']}"
            + " " * max(3 - len(str(round(test["effect_size"], 2))), 1)
            + f"{round(test['effect_size'], 2)}"
            + " " * max(3 - len(str(round(test["icc"], 2))), 1)
            + f"{round(test['icc'], 2)}"
            + " " * (6 - len(str(round(test["power"], 2))))
            + f"{round(test['power'], 2)}"
            + " " * (6 - len(str(round(test["alpha"], 2))))
            + f"{round(test['alpha'], 2)}"
            + " \n" * 2
            + f"Note: {test['note']}"
            + "\n"
            + f"URL: {test['url']}"
        )
    return test
