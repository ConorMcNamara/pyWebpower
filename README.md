# pyWebpower
A python implementation of the [Webpower](https://cran.r-project.org/web/packages/WebPower/index.html) R package; a library for calculating the power, sample size and minimum detectable effect of various Statistical Tests and Experiments. 

To quote the documentation

> This is a collection of tools for conducting both basic and advanced statistical power analysis including correlation, proportion, t-test, one-way ANOVA, two-way ANOVA, linear regression, logistic regression, Poisson regression, mediation analysis, longitudinal data analysis, structural equation modeling and multilevel modeling. It also serves as the engine for conducting power analysis online at <https://webpower.psychstat.org>.

## Quick Example
```
from webpower.power_tests import wp_anova_test
results = wp_anova_test(f=0.25, k=4, n=100, alpha=0.05)
print(round(results["power"]), 4)
0.5182
```

## Notes
Whenever possible, I tried to follow the R naming and code-style to ensure as much 1-1 comparison as possible; however, some liberties were taken to ensure the code follows PEP-8 guidelines. 

## References
Zhang, Zhiyong & Yuan, Ke-Hai. (2018). Practical Statistical Power Analysis Using Webpower and R. https://cran.r-project.org/web/packages/WebPower/index.html
