# pyWebpower

[![CI](https://github.com/ConorMcNamara/pyWebpower/actions/workflows/linter.yml/badge.svg)](https://github.com/ConorMcNamara/pyWebpower/actions/workflows/linter.yml)
[![codecov](https://codecov.io/gh/ConorMcNamara/pyWebpower/branch/main/graph/badge.svg)](https://codecov.io/gh/ConorMcNamara/pyWebpower)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Python implementation of the [WebPower](https://cran.r-project.org/web/packages/WebPower/index.html) R package — a library for calculating statistical power, sample size, and minimum detectable effect for a wide range of statistical tests.

> This is a collection of tools for conducting both basic and advanced statistical power analysis including correlation, proportion, t-test, one-way ANOVA, two-way ANOVA, linear regression, logistic regression, Poisson regression, mediation analysis, longitudinal data analysis, structural equation modeling and multilevel modeling.

## Installation

```bash
pip install pywebpower
```

## Quick Start

Leave exactly one parameter as `None` — that is the quantity to be solved for.

```python
from webpower.power_tests import wp_anova_test

# Solve for power
result = wp_anova_test(f=0.25, k=4, n=100, alpha=0.05)
print(round(result["power"], 4))  # 0.5182

# Solve for required sample size
result = wp_anova_test(f=0.25, k=4, n=None, alpha=0.05, power=0.8)
print(result["n"])  # 179
```

## Available Tests

| Function | Description |
|---|---|
| `wp_anova_test` | One-way ANOVA |
| `wp_anova_binary_test` | One-way ANOVA with binary outcome |
| `wp_anova_count_test` | One-way ANOVA with count outcome |
| `wp_kanova_test` | Multi-way ANOVA |
| `wp_rmanova_test` | Repeated-measures ANOVA |
| `wp_t1_test` | One-sample / paired / two-sample t-test |
| `wp_t2_test` | Unbalanced two-sample t-test |
| `wp_one_prop_test` | One-sample proportion test |
| `wp_two_prop_one_n_test` | Two-sample proportion test (equal n) |
| `wp_two_prop_two_n_test` | Two-sample proportion test (unequal n) |
| `wp_regression_test` | Multiple linear regression |
| `wp_poisson_test` | Poisson regression |
| `wp_logistic_test` | Logistic regression |
| `wp_mediation_test` | Simple mediation analysis |
| `wp_correlation_test` | Correlation |
| `wp_sem_chisq_test` | SEM (Satorra & Saris chi-square method) |
| `wp_sem_rmsea_test` | SEM (RMSEA-based) |
| `wp_mrt2arm_test` | Multisite randomised trial — 2 arms |
| `wp_mrt3arm_test` | Multisite randomised trial — 3 arms |
| `wp_crt2arm_test` | Cluster randomised trial — 2 arms |
| `wp_crt3arm_test` | Cluster randomised trial — 3 arms |

## Requirements

- Python 3.10+
- NumPy
- SciPy

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE)

## References

Zhang, Z., & Yuan, K.-H. (2018). *Practical Statistical Power Analysis Using Webpower and R*. ISDSA Press.
