import pytest

import webpower.power_tests as power_tests


# ANOVA


class TestAnova:
    @staticmethod
    def test_anova_results() -> None:
        power_results = power_tests.wp_anova_test(f=0.25, k=4, n=100, alpha=0.05)["power"]
        # wp.anova(f=0.25,k=4, n=100, alpha=0.05)
        # Power for One-way ANOVA
        #
        #     k   n    f alpha     power
        #     4 100 0.25  0.05 0.5181755
        #
        # NOTE: n is the total sample size (overall)
        # URL: http://psychstat.org/anova
        expected = 0.5181755
        assert power_results == pytest.approx(expected, abs=1e-05)

        sample_size_results = power_tests.wp_anova_test(f=0.25, k=4, n=None, alpha=0.05, power=0.8)["n"]
        # wp.anova(f=0.25,k=4, n=NULL, alpha=0.05, power=0.8)
        # Power for One-way ANOVA
        #
        #     k        n    f alpha power
        #     4 178.3971 0.25  0.05   0.8
        #
        # NOTE: n is the total sample size (overall)
        # URL: http://psychstat.org/anova
        expected = 179
        assert sample_size_results == expected

        mde_results = power_tests.wp_anova_test(f=None, k=4, n=100, alpha=0.05, power=0.8, test_type="two-sided")[
            "effect_size"
        ]
        # wp.anova(f=NULL,k=4, n=100, alpha=0.05, power=0.8, type="two.sided")
        # Power for One-way ANOVA
        #
        #     k   n         f alpha power
        #     4 100 0.2830003  0.05   0.8
        #
        # NOTE: n is the total sample size (contrast, two.sided)
        # URL: http://psychstat.org/anova
        expected = 0.2830003
        assert mde_results == pytest.approx(expected, abs=1e-01)

        k_results = power_tests.wp_anova_test(k=None, f=0.3, n=100, alpha=0.10, power=0.8, test_type="less")["k"]
        # Having issues with trying to solve for k in WebPower, it keeps returning errors
        assert k_results == 100

        alpha_results = power_tests.wp_anova_test(k=4, f=0.3, n=100, alpha=None, power=0.8, test_type="greater")[
            "alpha"
        ]
        # wp.anova(k=4, f=0.3, n=100, alpha=NULL, power=0.8, type='greater')
        # Power for One-way ANOVA
        #
        #     k   n   f      alpha power
        #     4 100 0.3 0.01687416   0.8
        #
        # NOTE: n is the total sample size (contrast, greater)
        # URL: http://psychstat.org/anova
        expected = 0.01687416
        assert alpha_results == pytest.approx(expected, abs=1e-05)


class TestAnovaBinary:
    @staticmethod
    def test_anova_results() -> None:
        power_results = power_tests.wp_anova_binary_test(k=4, n=100, V=0.15, alpha=0.05)["power"]
        # wp.anova.binary(k=4,n=100,V=0.15,alpha=0.05)
        # One-way Analogous ANOVA with Binary Data
        #
        #    k   n    V alpha     power
        #    4 100 0.15  0.05 0.5723443
        #
        # NOTE: n is the total sample size
        # URL: http://psychstat.org/anovabinary
        expected = 0.5723443
        assert power_results == pytest.approx(expected)

        sample_size_results = power_tests.wp_anova_binary_test(k=4, n=None, V=0.15, power=0.8, alpha=0.05)["n"]
        # One-way Analogous ANOVA with Binary Data
        #
        #     k        n    V alpha power
        #     4 161.5195 0.15  0.05   0.8
        #
        # NOTE: n is the total sample size
        # URL: http://psychstat.org/anovabinary
        expected = 162
        assert sample_size_results == expected

        effect_size_results = power_tests.wp_anova_binary_test(k=4, n=100, V=None, power=0.8, alpha=0.05)["effect_size"]
        # wp.anova.binary(k=4,n=100,V=NULL,power=0.8, alpha=0.05)
        # One-way Analogous ANOVA with Binary Data
        #
        #     k   n         V alpha power
        #     4 100 0.1906373  0.05   0.8
        #
        # NOTE: n is the total sample size
        # URL: http://psychstat.org/anovabinary
        expected = 0.1906373
        assert effect_size_results == pytest.approx(expected, abs=1e-05)

        groups_results = power_tests.wp_anova_binary_test(k=None, n=100, V=0.19, power=0.8, alpha=0.05)["k"]
        # wp.anova.binary(k=NULL ,n=100,V=0.19,power=0.8, alpha=0.05)
        # One-way Analogous ANOVA with Binary Data
        #
        #            k   n    V alpha power
        #     4.029187 100 0.19  0.05   0.8
        #
        # NOTE: n is the total sample size
        # URL: http://psychstat.org/anovabinary
        expected = 5
        assert groups_results == expected

        alpha_results = power_tests.wp_anova_binary_test(k=4, n=100, V=0.19, power=0.80, alpha=None)["alpha"]
        # wp.anova.binary(k=4 ,n=100,V=0.19,power=0.8, alpha=NULL)
        # One-way Analogous ANOVA with Binary Data
        #
        #     k   n    V      alpha power
        #     4 100 0.19 0.05125109   0.8
        #
        # NOTE: n is the total sample size
        # URL: http://psychstat.org/anovabinary
        expected = 0.05125109
        assert alpha_results == pytest.approx(expected, abs=1e-05)


class TestAnovaCount:
    @staticmethod
    def test_anova_results() -> None:
        power_results = power_tests.wp_anova_count_test(k=4, n=100, V=0.148, alpha=0.05)["power"]
        # wp.anova.count(k=4, n=100, V=0.148, alpha=0.05)
        # One-way Analogous ANOVA with Count Data
        #
        #     k   n     V alpha     power
        #     4 100 0.148  0.05 0.5597441
        #
        # NOTE: n is the total sample size
        # URL: http://psychstat.org/anovacount
        expected = 0.5597441
        assert power_results == pytest.approx(expected)

        sample_size_results = power_tests.wp_anova_count_test(k=4, n=None, V=0.148, power=0.8, alpha=0.05)["n"]
        # wp.anova.count(k=4, n=NULL, V=0.148, power=0.8, alpha=0.05)
        # One-way Analogous ANOVA with Count Data
        #
        #     k        n     V alpha power
        #     4 165.9143 0.148  0.05   0.8
        #
        # NOTE: n is the total sample size
        # URL: http://psychstat.org/anovacount
        expected = 166
        assert sample_size_results == expected

        effect_size_results = power_tests.wp_anova_count_test(k=4, n=100, V=None, power=0.8, alpha=0.05)["effect_size"]
        # wp.anova.count(k=4, n=100, V=NULL, power=0.8, alpha=0.05)
        # One-way Analogous ANOVA with Count Data
        #
        #     k   n         V alpha power
        #     4 100 0.1906373  0.05   0.8
        #
        # NOTE: n is the total sample size
        # URL: http://psychstat.org/anovacount
        expected = 0.1906373
        assert effect_size_results == pytest.approx(expected, abs=1e-05)

        groups_results = power_tests.wp_anova_count_test(k=None, n=166, V=0.148, power=0.8, alpha=0.05)["k"]
        # One-way Analogous ANOVA with Count Data
        #
        #            k   n     V alpha power
        #     3.997756 166 0.148  0.05   0.8
        #
        # NOTE: n is the total sample size
        # URL: http://psychstat.org/anovacount
        expected = 4
        assert groups_results == expected

        alpha_result = power_tests.wp_anova_count_test(k=4, n=100, V=0.20, power=0.8, alpha=None)["alpha"]
        # wp.anova.count(k=4, n=100, V=0.20, power=0.8, alpha=NULL)
        # One-way Analogous ANOVA with Count Data
        #
        #     k   n   V      alpha power
        #     4 100 0.2 0.03433487   0.8
        #
        # NOTE: n is the total sample size
        # URL: http://psychstat.org/anovacount
        expected = 0.03433487
        assert alpha_result == pytest.approx(expected, abs=1e-05)


class TestKAnova:
    @staticmethod
    def test_kanova_result() -> None:
        power_results = power_tests.wp_kanova_test(n=120, ndf=2, f=0.2, alpha=0.05, ng=6)["power"]
        # wp.kanova(n=120, ndf=2, f=0.2, alpha=0.05, ng=6)
        # Multiple way ANOVA analysis
        #
        #       n ndf ddf   f ng alpha     power
        #     120   2 114 0.2  6  0.05 0.4757998
        #
        # NOTE: Sample size is the total sample size
        # URL: http://psychstat.org/kanova
        expected = 0.4757998
        assert power_results == pytest.approx(expected, abs=1e-05)

        sample_size_results = power_tests.wp_kanova_test(n=None, ndf=2, f=0.2, alpha=0.05, ng=6, power=0.80)["n"]
        # wp.kanova(n=NULL, ndf=2, f=0.2, alpha=0.05, ng=6, power=0.80)
        # Multiple way ANOVA analysis
        #
        #            n ndf      ddf   f ng alpha power
        #     243.9259   2 237.9259 0.2  6  0.05   0.8
        #
        # NOTE: Sample size is the total sample size
        # URL: http://psychstat.org/kanova
        expected = 244
        assert sample_size_results == expected

        degrees_freedom_results = power_tests.wp_kanova_test(n=1000, ndf=None, f=0.2, alpha=0.05, ng=6, power=0.80)[
            "ndf"
        ]
        # wp.kanova(n=1000, ndf=NULL, f=0.2, alpha=0.05, ng=6, power=0.80)
        # Multiple way ANOVA analysis
        #
        #        n      ndf ddf   f ng alpha power
        #     1000 85.79258 994 0.2  6  0.05   0.8
        #
        # NOTE: Sample size is the total sample size
        # URL: http://psychstat.org/kanova
        expected = 86
        assert degrees_freedom_results == expected

        effect_size_results = power_tests.wp_kanova_test(n=1000, ndf=5, f=None, alpha=0.05, ng=4, power=0.80)[
            "effect_size"
        ]
        # wp.kanova(n=1000, ndf=5, f=NULL, alpha=0.05, ng=4, power=0.80)
        # Multiple way ANOVA analysis
        #
        #        n ndf ddf         f ng alpha power
        #     1000   5 996 0.1135649  4  0.05   0.8
        #
        # NOTE: Sample size is the total sample size
        # URL: http://psychstat.org/kanova
        expected = 0.1135649
        assert effect_size_results == pytest.approx(expected, abs=1e-05)

        # groups_results = power_tests.wp_kanova_test()["ng"]
        # Getting issues with WebPower, it pretty much always returns the same error no matter what values I try

        alpha_results = power_tests.wp_kanova_test(n=75, ndf=4, f=0.5, alpha=None, ng=5, power=0.80)["alpha"]
        # wp.kanova(n=75, ndf=4, f=0.5, alpha=NULL, ng=5, power=0.80)
        # Multiple way ANOVA analysis
        #
        #      n ndf ddf   f ng       alpha power
        #     75   4  70 0.5  5 0.009071081   0.8
        #
        # NOTE: Sample size is the total sample size
        # URL: http://psychstat.org/kanova
        expected = 0.009071081
        assert alpha_results == pytest.approx(expected, abs=1e-03)


class TestRMAnova:
    @staticmethod
    def test_rmanova_results() -> None:
        power_results = power_tests.wp_rmanova_test(n=30, ng=3, nm=4, f=0.36, nscor=0.7, alpha=0.05)["power"]
        # wp.rmanova(n=30, ng=3, nm=4, f=0.36, nscor=0.7)
        # Repeated-measures ANOVA analysis
        #
        # n f ng nm nscor alpha power
        # 30 0.36 3 4 0.7 0.05 0.2674167
        #
        # NOTE: Power analysis for between-effect test
        # URL: http://psychstat.org/rmanova
        expected = 0.2674167
        assert power_results == pytest.approx(expected, abs=1e-05)

        sample_size_results = power_tests.wp_rmanova_test(
            n=None,
            ng=3,
            nm=4,
            f=0.36,
            power=0.8,
            nscor=0.7,
            alpha=0.05,
            test_type="interaction",
        )["n"]
        # wp.rmanova(n=NULL, ng=3, nm=4, f=0.36, power=0.8, nscor=0.7, type=2)
        # Repeated-measures ANOVA analysis
        #
        #            n    f ng nm nscor alpha power
        #     135.9974 0.36  3  4   0.7  0.05   0.8
        #
        # NOTE: Power analysis for interaction-effect test
        # URL: http://psychstat.org/rmanova
        expected = 136
        assert sample_size_results == expected

        effect_size_results = power_tests.wp_rmanova_test(
            n=30,
            ng=3,
            nm=4,
            f=None,
            power=0.8,
            nscor=0.7,
            alpha=0.05,
            test_type="within",
        )["effect_size"]
        # wp.rmanova(n=30, ng=3, nm=4, f=NULL, power=0.8, nscor=0.7, type=1)
        # Repeated-measures ANOVA analysis
        #
        #      n         f ng nm nscor alpha power
        #     30 0.7013686  3  4   0.7  0.05   0.8
        #
        # NOTE: Power analysis for within-effect test
        # URL: http://psychstat.org/rmanova
        expected = 0.7013686
        assert effect_size_results == pytest.approx(expected, abs=1e-04)

        # Currently ng and nm are incorrectly defined in the R package so they don't return any results
        groups_results = power_tests.wp_rmanova_test(n=30, nm=4, f=0.71, power=0.8, nscor=0.7, alpha=0.05)["ng"]
        expected = 3
        assert groups_results == expected

        nm_results = power_tests.wp_rmanova_test(
            n=30, ng=3, f=0.70, power=0.8, nscor=0.7, alpha=0.05, test_type="within"
        )["nm"]
        expected = 4
        assert nm_results == expected


# PROPORTION TESTS


class TestOneProp:
    @staticmethod
    def test_oneprop_results() -> None:
        power_results = power_tests.wp_one_prop_test(h=0.25, n=100, power=None, alternative="two-sided", alpha=0.05)[
            "power"
        ]
        # wp.prop(h=0.25, n1=100,power=NULL,alternative="two.sided",type="1p")
        # Power for one-sample proportion test
        #
        #        h   n alpha    power
        #     0.52 100  0.05 0.705418
        #
        # URL: http://psychstat.org/prop
        expected = 0.705418
        assert power_results == pytest.approx(expected, abs=1e-06)

        effect_size_results = power_tests.wp_one_prop_test(h=None, n=250, power=0.8, alpha=0.1, alternative="less")[
            "effect_size"
        ]
        # wp.prop(h=NULL, n1=250, power=0.8, alpha=0.1, alternative="less")
        # Power for one-sample proportion test
        #
        #              h   n alpha power
        #     -0.1342893 250   0.1   0.8
        #
        # URL: http://psychstat.org/prop
        expected = -0.1342893
        assert effect_size_results == pytest.approx(expected, abs=1e-05)

        sample_size_results = power_tests.wp_one_prop_test(
            h=0.52, n=None, power=0.8, alpha=0.05, alternative="greater"
        )["n"]
        # wp.prop(h=0.52,n1=NULL,power=0.8,alternative="greater",type="1p")
        # Power for one-sample proportion test
        #
        #    h        n alpha power
        # 0.52 22.86449  0.05   0.8
        #
        # URL: http://psychstat.org/prop
        expected = 23
        assert sample_size_results == expected

        alpha_results = power_tests.wp_one_prop_test(h=0.1, n=500, power=0.8, alpha=None, alternative="two-sided")[
            "alpha"
        ]
        # wp.prop(h=0.1, n1=500, power=0.8, alpha=NULL, alternative="two.sided")
        # Power for one-sample proportion test
        #
        #       h   n        alpha power
        #     0.3 500    0.1630304   0.8
        #
        # URL: http://psychstat.org/prop
        expected = 0.1630304
        assert alpha_results == pytest.approx(expected, abs=1e-05)


class TestTwoPropOneN:
    @staticmethod
    def test_twoprop_onen_results() -> None:
        power_results = power_tests.wp_two_prop_one_n_test(
            h=0.1, n=500, alpha=0.10, power=None, alternative="two-sided"
        )["power"]
        # wp.prop(h=0.1, n1=500, power=NULL, alpha=0.10, alternative="two.sided", type = "2p")
        # Power for two-sample proportion (equal n)
        #
        #       h   n alpha     power
        #     0.1 500   0.1 0.4752263
        #
        # NOTE: Sample sizes for EACH group
        # URL: http://psychstat.org/prop2p
        expected = 0.4752263
        assert power_results == pytest.approx(expected)

        effect_size_results = power_tests.wp_two_prop_one_n_test(
            h=None, n=1_000, power=0.8, alpha=0.1, alternative="greater"
        )["effect_size"]
        # wp.prop(h=NULL, n1=1000, power=0.8, alpha=0.10, alternative="greater", type = "2p")
        # Power for two-sample proportion (equal n)
        #
        #              h    n alpha power
        #     0.09497019 1000   0.1   0.8
        #
        # NOTE: Sample sizes for EACH group
        # URL: http://psychstat.org/prop2p
        expected = 0.09497019
        assert effect_size_results == pytest.approx(expected, abs=1e-04)

        sample_size_results = power_tests.wp_two_prop_one_n_test(
            h=-0.1, n=None, power=0.8, alpha=0.1, alternative="less"
        )["n"]
        # wp.prop(h=-0.1, n1=NULL, power=0.8, alpha=0.10, alternative="less", type = "2p")
        # Power for two-sample proportion (equal n)
        #
        #        h        n alpha power
        #     -0.1 901.5725   0.1   0.8
        #
        # NOTE: Sample sizes for EACH group
        # URL: http://psychstat.org/prop2p
        expected = 902
        assert sample_size_results == expected

        alpha_results = power_tests.wp_two_prop_one_n_test(
            h=0.1, n=1000, power=0.8, alpha=None, alternative="two-sided"
        )["alpha"]
        # wp.prop(h=0.1, n1=1000, power=0.8, alpha=NULL, alternative="two.sided", type = "2p")
        # Power for two-sample proportion (equal n)
        #
        #       h    n     alpha power
        #     0.1 1000 0.1630304   0.8
        #
        # NOTE: Sample sizes for EACH group
        # URL: http://psychstat.org/prop2p
        expected = 0.1630304
        assert alpha_results == pytest.approx(expected, abs=1e-04)


class TestTwoPropTwoN:
    @staticmethod
    def test_twoprop_twon_results() -> None:
        power_results = power_tests.wp_two_prop_two_n_test(
            h=0.2, n1=1_000, n2=750, power=None, alpha=0.1, alternative="greater"
        )["power"]
        # wp.prop(h=0.2, n1=1000, n2=750, power=NULL, alpha=0.1, alternative="greater", type = "2p2n")
        # Power for two-sample proportion (unequal n)
        #
        #       h   n1  n2 alpha    power
        #     0.2 1000 750   0.1 0.997874
        #
        # NOTE: Sample size for each group
        # URL: http://psychstat.org/prop2p2n
        expected = 0.997874
        assert power_results == pytest.approx(expected)

        effect_size_results = power_tests.wp_two_prop_two_n_test(
            h=None, n1=10, n2=5, power=0.8, alpha=0.05, alternative="two-sided"
        )["effect_size"]
        # wp.prop(h=NULL, n1=10, n2=5, power=0.8, alpha=0.05, alternative="two.sided", type = "2p2n")
        # Power for two-sample proportion (unequal n)
        #
        #            h n1 n2 alpha power
        #     1.534504 10  5  0.05   0.8
        #
        # NOTE: Sample size for each group
        # URL: http://psychstat.org/prop2p2n
        expected = 1.534504
        assert effect_size_results == pytest.approx(expected, abs=1e-04)

        n1_results = power_tests.wp_two_prop_two_n_test(
            h=-0.2, n1=None, n2=750, power=0.8, alpha=0.1, alternative="less"
        )["n1"]
        # wp.prop(h=-0.2, n1=NULL, n2=750, power=0.8, alpha=0.1, alternative="less", type = "2p2n")
        # Power for two-sample proportion (unequal n)
        #
        #        h       n1  n2 alpha power
        #     -0.2 132.6251 750   0.1   0.8
        #
        # NOTE: Sample size for each group
        # URL: http://psychstat.org/prop2p2n
        expected = 133
        assert n1_results == expected

        n2_results = power_tests.wp_two_prop_two_n_test(
            h=0.1, n1=1_500, n2=None, power=0.8, alpha=0.1, alternative="two-sided"
        )["n2"]
        # wp.prop(h=0.1, n1=1500, n2=NULL, power=0.8, alpha=0.1, alternative="two.sided", type = "2p2n")
        # Power for two-sample proportion (unequal n)
        #
        #       h   n1       n2 alpha power
        #     0.1 1500 1051.668   0.1   0.8
        #
        # NOTE: Sample size for each group
        # URL: http://psychstat.org/prop2p2n
        expected = 1052
        assert n2_results == expected

        alpha_results = power_tests.wp_two_prop_two_n_test(
            h=0.1, n1=1_000, n2=500, power=0.8, alpha=None, alternative="two-sided"
        )["alpha"]
        # wp.prop(h=0.1, n1=1000, n2=500, power=0.8, alpha=NULL, alternative="two.sided", type = "2p2n")
        # Power for two-sample proportion (unequal n)
        #
        #       h   n1  n2     alpha power
        #     0.1 1000 500 0.3208521   0.8
        #
        # NOTE: Sample size for each group
        # URL: http://psychstat.org/prop2p2n
        expected = 0.3208521
        assert alpha_results == pytest.approx(expected)


# T TESTS


class TestOneT:
    @staticmethod
    def test_onet_results() -> None:
        power_results = power_tests.wp_t1_test(n=150, d=0.2, power=None, alpha=0.05, test_type="one-sample")["power"]
        # wp.t(n1=150, d=0.2, type="one.sample")
        # One-sample t-test
        #
        #       n   d alpha    power
        #     150 0.2  0.05 0.682153
        #
        # URL: http://psychstat.org/ttest
        expected = 0.682153
        assert power_results == pytest.approx(expected, abs=1e-05)

        sample_size_results = power_tests.wp_t1_test(
            n=None,
            d=0.4,
            power=0.8,
            alpha=0.05,
            test_type="paired",
            alternative="greater",
        )["n"]
        # wp.t(d=0.4, power=0.8, type="paired", alternative="greater")
        # Paired t-test
        #
        #            n   d alpha power
        #     40.02908 0.4  0.05   0.8
        #
        # NOTE: n is number of *pairs*
        # URL: http://psychstat.org/ttest
        expected = 41
        assert sample_size_results == expected

        effect_size_results = power_tests.wp_t1_test(
            n=250,
            d=None,
            power=0.8,
            alpha=0.1,
            test_type="one-sample",
            alternative="two-sided",
        )["effect_size"]
        # wp.t(n1=250, d=NULL, power=0.8, alpha=0.1, type="one.sample", alternative="two.sided")
        # One-sample t-test
        #
        #       n        d alpha power
        #     250 0.157657   0.1   0.8
        #
        # URL: http://psychstat.org/ttest
        expected = 0.157657
        assert effect_size_results == pytest.approx(expected, abs=1e-03)

        alpha_results = power_tests.wp_t1_test(
            n=100,
            d=-0.1,
            power=0.75,
            alpha=None,
            test_type="paired",
            alternative="less",
        )["alpha"]
        # wp.t(n1=100, d=-0.1, power=0.75, alpha=NULL, type="paired", alternative="less")
        # Paired t-test
        #
        #       n    d     alpha power
        #     100 -0.1 0.3725015  0.75
        #
        # NOTE: n is number of *pairs*
        # URL: http://psychstat.org/ttest
        expected = 0.3725015
        assert alpha_results == pytest.approx(expected, abs=1e-05)


class TestTwoT:
    @staticmethod
    def test_twot_results() -> None:
        power_results = power_tests.wp_t2_test(n1=30, n2=40, d=0.356, alpha=0.05, alternative="two-sided")["power"]
        # wp.t(n1=30, n2=40, d=0.356, type="two.sample.2n", alternative="two.sided")
        # Unbalanced two-sample t-test
        #
        #     n1 n2     d alpha     power
        #     30 40 0.356  0.05 0.3064767
        #
        # NOTE: n1 and n2 are number in *each* group
        # URL: http://psychstat.org/ttest2n
        expected = 0.3064767
        assert power_results == pytest.approx(expected)

        n1_results = power_tests.wp_t2_test(n1=None, n2=400, d=0.356, power=0.8, alpha=0.05, alternative="greater")[
            "n1"
        ]
        # wp.t(n1=NULL, n2=400, d=0.356, power=0.8, type="two.sample.2n", alternative="greater")
        # Unbalanced two-sample t-test
        #
        #           n1  n2     d alpha power
        #     55.74806 400 0.356  0.05   0.8
        #
        # NOTE: n1 and n2 are number in *each* group
        # URL: http://psychstat.org/ttest2n
        expected = 56
        assert n1_results == expected

        n2_results = power_tests.wp_t2_test(n1=1_000, n2=None, d=0.4, alpha=0.05, power=0.8)["n2"]
        # wp.t(n1=1000, n2=NULL, d=-0.4, power=0.8, type="two.sample.2n", alternative="two.sided")
        # Unbalanced two-sample t-test
        #
        #       n1      n2   d alpha power
        #     1000 51.6854 0.4  0.05   0.8
        #
        # NOTE: n1 and n2 are number in *each* group
        # URL: http://psychstat.org/ttest2n
        expected = 52
        assert n2_results == expected

        effect_size_results = power_tests.wp_t2_test(
            n1=2_000, n2=2_500, d=None, alpha=0.1, power=0.8, alternative="greater"
        )["effect_size"]
        # wp.t(n1=2000, n2=2500, d=NULL, power=0.8, alpha=0.10, type="two.sample.2n", alternative="greater")
        # Unbalanced two-sample t-test
        #
        #       n1   n2          d alpha power
        #     2000 2500 0.06370563   0.1   0.8
        #
        # NOTE: n1 and n2 are number in *each* group
        # URL: http://psychstat.org/ttest2n
        expected = 0.06370563
        assert effect_size_results == pytest.approx(expected, abs=1e-05)

        alpha_results = power_tests.wp_t2_test(n1=500, n2=50, d=-0.03, alpha=None, power=0.8, alternative="less")[
            "alpha"
        ]
        # pwr.t2n.test(n1=500, n2=50, d=-.03, sig.level=NULL, power=0.8, alternative="less")
        #
        #      t test power calculation
        #
        #              n1 = 500
        #              n2 = 50
        #               d = -0.03
        #       sig.level = 0.7387184
        #           power = 0.8
        #     alternative = less
        # Having an issue where webpower is simply reporting the same values when alternative="greater" and alternative="less",
        # which shouldn't be the case at all. Raised an issue about this.
        expected = 0.7387184
        assert alpha_results == pytest.approx(expected, abs=1e-05)


# REGRESSION


class TestRegression:
    @staticmethod
    def test_regression_results() -> None:
        power_results = power_tests.wp_regression_test(n=100, p1=3, f2=0.1, alpha=0.05, power=None)["power"]
        # wp.regression(n = 100, p1 = 3, f2 = 0.1, alpha = 0.05, power = NULL)
        # Power for multiple regression
        #
        #   n p1 p2  f2 alpha     power
        # 100  3  0 0.1  0.05 0.7420463
        #
        # URL: http://psychstat.org/regression
        expected = 0.7420463
        assert power_results == pytest.approx(expected, abs=1e-05)

        n_results = power_tests.wp_regression_test(n=None, p1=5, p2=2, f2=0.3, alpha=0.1, power=0.8, test_type="Cohen")[
            "n"
        ]
        # wp.regression(n=NULL, p1=5, p2=2, f2=0.3, alpha=0.1, power=0.8, type = "Cohen")
        # Power for multiple regression
        #
        #            n p1 p2  f2 alpha power
        #     34.69994  5  2 0.3   0.1   0.8
        #
        # URL: http://psychstat.org/regression
        expected = 35
        assert n_results == expected

        f2_results = power_tests.wp_regression_test(n=113, p1=100, p2=3, alpha=0.1, power=0.8, test_type="regular")[
            "effect_size"
        ]
        # wp.regression(n=113, p1=100, p2=3, alpha=0.1, power=0.8, type="regular")
        # Power for multiple regression
        #
        #       n  p1 p2       f2 alpha power
        #     113 100  3 1.375547   0.1   0.8
        #
        # URL: http://psychstat.org/regression
        expected = 1.375547
        assert f2_results == pytest.approx(expected, abs=1e-03)

        alpha_results = power_tests.wp_regression_test(n=130, p1=100, p2=10, f2=0.5, power=0.8, test_type="Cohen")[
            "alpha"
        ]
        # wp.regression(n=130, p1=100, p2=10, f2=0.5, power=0.8, type="Cohen", alpha=NULL)
        # Power for multiple regression
        #
        #       n  p1 p2  f2     alpha power
        #     130 100 10 0.5 0.1960784   0.8
        #
        # URL: http://psychstat.org/regression
        expected = 0.1960784
        assert alpha_results == pytest.approx(expected, abs=1e-03)


class TestPoisson:
    @staticmethod
    def test_poisson_results() -> None:
        power_results = power_tests.wp_poisson_test(
            n=4406,
            exp0=2.798,
            exp1=0.8938,
            alpha=0.05,
            power=None,
            family="Bernoulli",
            parameter=0.53,
        )["power"]
        # wp.poisson(n = 4406, exp0 = 2.798, exp1 = 0.8938, alpha = 0.05, power = NULL, family = "Bernoulli", parameter = 0.53)
        # Power for Poisson regression
        #
        #    n     power alpha  exp0   exp1    beta0      beta1 parameter
        # 4406 0.9999789  0.05 2.798 0.8938 1.028905 -0.1122732      0.53
        #
        # URL: http://psychstat.org/poisson
        expected = 0.9999789
        assert power_results == pytest.approx(expected, abs=1e-05)

        n_results = power_tests.wp_poisson_test(
            n=None,
            exp0=2.798,
            exp1=0.8938,
            alpha=0.05,
            power=0.8,
            family="exponential",
            parameter=0.53,
            alternative="less",
        )["n"]
        # wp.poisson(n = NULL, exp0 = 2.798, exp1 = 0.8938, alpha = 0.05, power = 0.8, family = "exponential", parameter = 0.53, alternative = "less")
        # Power for Poisson regression
        #
        #            n power alpha  exp0   exp1    beta0      beta1 paremeter
        #     87.62974   0.8  0.05 2.798 0.8938 1.028905 -0.1122732      0.53
        #
        # URL: http://psychstat.org/poisson
        expected = 88
        assert n_results == expected

        power_results = power_tests.wp_poisson_test(
            n=40,
            exp0=1.5,
            exp1=0.9,
            alpha=0.05,
            power=None,
            family="lognormal",
            parameter=None,
            alternative="greater",
        )["power"]
        # wp.poisson(n=40, exp0=1.5, exp1=0.9, alpha=0.05, power=NULL, family="lognormal",
        # +            parameter=NULL, alternative="greater")
        # Power for Poisson regression
        #
        #      n       power alpha exp0 exp1     beta0      beta1
        #     40 0.003109859  0.05  1.5  0.9 0.4054651 -0.1053605
        #
        # URL: http://psychstat.org/poisson
        expected = 0.003109859
        assert power_results == pytest.approx(expected)

        n_results = power_tests.wp_poisson_test(
            n=None,
            exp0=1.5,
            exp1=0.9,
            alpha=0.05,
            power=0.8,
            family="normal",
            parameter=None,
            alternative="two-sided",
        )["n"]
        # wp.poisson(n=NULL, exp0=1.5, exp1=0.9, alpha=0.05, power=0.8, family="normal",
        # +            parameter=NULL, alternative="two.sided")
        # Power for Poisson regression
        #
        #            n power alpha exp0 exp1     beta0      beta1
        #     468.7584   0.8  0.05  1.5  0.9 0.4054651 -0.1053605
        #
        # URL: http://psychstat.org/poisson
        expected = 469
        assert n_results == expected

        power_results = power_tests.wp_poisson_test(
            n=450,
            exp0=1.5,
            exp1=0.9,
            alpha=0.05,
            power=None,
            family="poisson",
            parameter=None,
            alternative="less",
        )["power"]
        # wp.poisson(n=450, exp0=1.5, exp1=0.9, alpha=0.05, power=NULL, family="Poisson",
        # +            parameter=NULL, alternative="less")
        # Power for Poisson regression
        #
        #       n     power alpha exp0 exp1     beta0      beta1
        #     450 0.7954193  0.05  1.5  0.9 0.4054651 -0.1053605
        #
        # URL: http://psychstat.org/poisson
        expected = 0.7954193
        assert power_results == pytest.approx(expected, abs=1e-05)

        n_results = power_tests.wp_poisson_test(
            n=None,
            exp0=1.5,
            exp1=0.9,
            alpha=0.05,
            power=0.7,
            family="uniform",
            parameter=None,
            alternative="two-sided",
        )["n"]
        # wp.poisson(n=NULL, exp0=1.5, exp1=0.9, alpha=0.05, power=0.7, family="uniform",
        # +            parameter=NULL, alternative="two.sided")
        # Power for Poisson regression
        #
        #           n power alpha exp0 exp1     beta0      beta1
        #     4688.99   0.7  0.05  1.5  0.9 0.4054651 -0.1053605
        #
        # URL: http://psychstat.org/poisson
        expected = 4689
        assert n_results == expected


class TestLogistic:
    @staticmethod
    def test_logistic_results() -> None:
        # The webpower logistic class currently does not calculate alpha, so cannot compare alpha results,
        # only power and n. Also, family="poisson" appears to be broken, I can't find any combination of p0 and p1 values
        # to get it to run on the R package.
        power_results = power_tests.wp_logistic_test(
            n=1_000,
            p0=0.2,
            p1=0.25,
            alpha=0.1,
            power=None,
            family="bernoulli",
            parameter=None,
            alternative="greater",
        )["power"]
        # p.logistic(n = 1000, p0 = 0.2, p1 = 0.25, alpha = 0.10, power = NULL, family = "Bernoulli", parameter = NULL,
        # +          alternative="greater")
        # Power for logistic regression
        #
        #      p0   p1     beta0     beta1    n alpha     power
        #     0.2 0.25 -1.386294 0.2876821 1000   0.1 0.7285827
        #
        # URL: http://psychstat.org/logistic
        expected = 0.7285827
        assert power_results == pytest.approx(expected, abs=1e-05)

        n_results = power_tests.wp_logistic_test(
            n=None,
            p0=0.15,
            p1=0.10,
            alpha=0.10,
            power=0.8,
            family="lognormal",
            parameter=None,
            alternative="two-sided",
        )["n"]
        # wp.logistic(n = NULL, p0=0.15, p1=0.1, alpha = 0.10, power = 0.80, family = "lognormal", parameter = NULL,
        # +           alternative="two.sided")
        # Power for logistic regression
        #
        #       p0  p1     beta0      beta1        n alpha power
        #     0.15 0.1 -1.734601 -0.4626235 468.5579   0.1   0.8
        #
        # URL: http://psychstat.org/logistic
        expected = 469
        assert n_results == expected

        power_results = power_tests.wp_logistic_test(
            n=200,
            p0=0.2,
            p1=0.1,
            alpha=0.1,
            power=None,
            family="exponential",
            parameter=None,
            alternative="two-sided",
        )["power"]
        # wp.logistic(n=200, p0=0.2, p1=0.1, alpha=0.1, power=NULL, family="exponential", parameter=NULL)
        # Power for logistic regression
        #
        #      p0  p1     beta0      beta1   n alpha    power
        #     0.2 0.1 -1.386294 -0.8109302 200   0.1 0.692976
        expected = 0.692976
        assert power_results == pytest.approx(expected, abs=1e-05)

        n_results = power_tests.wp_logistic_test(
            n=None,
            p0=0.15,
            p1=0.1,
            alpha=0.05,
            power=0.8,
            family="normal",
            parameter=[0, 1],
            alternative="two-sided",
        )["n"]
        # wp.logistic(n = NULL, p0 = 0.15, p1 = 0.1, alpha = 0.05, power = 0.8, family = "normal", parameter = c(0,1))
        # Power for logistic regression
        #
        #   p0  p1     beta0      beta1        n alpha power
        # 0.15 0.1 -1.734601 -0.4626235 298.9207  0.05   0.8
        #
        # URL: http://psychstat.org/logistic
        expected = 299
        assert n_results == expected

        power_results = power_tests.wp_logistic_test(
            n=100,
            p0=0.05,
            p1=0.25,
            alpha=0.10,
            power=None,
            family="uniform",
            parameter=None,
            alternative="less",
        )["power"]
        # wp.logistic(n = 100, p0 = 0.05, p1 = 0.25, alpha = 0.10, power = NULL, family = "uniform", parameter = NULL,
        # +           alternative="less")
        # Power for logistic regression
        #
        #       p0   p1     beta0    beta1   n alpha       power
        #     0.05 0.25 -2.944439 1.845827 100   0.1 0.001653495
        expected = 0.001653495
        assert power_results == pytest.approx(expected, abs=1e-05)


# STRUCTURAL EQUATION MODELING


class TestSEMChisq:
    @staticmethod
    def test_sem_chisq_results() -> None:
        power_results = power_tests.wp_sem_chisq_test(n=100, df=4, effect=0.054, power=None, alpha=0.05)["power"]
        # wp.sem.chisq(n=100, df=4, effect=0.054, power=NULL, alpha=0.05)
        # Power for SEM (Satorra & Saris, 1985)
        #
        #   n df effect     power alpha
        # 100  4  0.054 0.4221152  0.05
        #
        # URL: http://psychstat.org/semchisq
        expected = 0.4221152
        assert expected == pytest.approx(power_results, abs=1e-05)

        n_results = power_tests.wp_sem_chisq_test(n=None, df=4, effect=0.054, power=0.8, alpha=0.05)["n"]
        # wp.sem.chisq(n = NULL, df = 4, effect = 0.054, power = 0.8, alpha = 0.05)
        # Power for SEM (Satorra & Saris, 1985)
        #
        #        n df effect power alpha
        # 222.0238  4  0.054   0.8  0.05
        #
        # URL: http://psychstat.org/semchisq
        expected = 223
        assert expected == n_results

        effect_size_results = power_tests.wp_sem_chisq_test(n=100, df=4, effect=None, power=0.8, alpha=0.05)[
            "effect_size"
        ]
        # wp.sem.chisq(n=100, df=4, effect=NULL, power=0.8, alpha=0.05)
        # Power for SEM (Satorra & Saris, 1985)
        #
        #   n df    effect power alpha
        # 100  4 0.1205597   0.8  0.05
        #
        # URL: http://psychstat.org/semchisq
        expected = 0.1205597
        assert expected == pytest.approx(effect_size_results, abs=1e-05)

        df_results = power_tests.wp_sem_chisq_test(n=1_000, df=None, effect=0.054, power=0.8, alpha=0.05)["df"]
        # wp.sem.chisq(n=1000, df=NULL, effect=0.054, power=0.8, alpha=0.05)
        # Power for SEM (Satorra & Saris, 1985)
        #
        #        n       df effect power alpha
        #     1000 190.3743  0.054   0.8  0.05
        #
        # URL: http://psychstat.org/semchisq
        expected = 191
        assert expected == df_results

        alpha_results = power_tests.wp_sem_chisq_test(n=100, df=5, effect=0.054, power=0.8, alpha=None)["alpha"]
        # wp.sem.chisq(n=100, df=5, effect=0.054, power=0.8, alpha=NULL)
        # Power for SEM (Satorra & Saris, 1985)
        #
        #       n df effect power     alpha
        #     100  5  0.054   0.8 0.3549963
        #
        # URL: http://psychstat.org/semchisq
        expected = 0.3549963
        assert expected == pytest.approx(alpha_results, abs=1e-03)


class TestSEMRMSEA:
    @staticmethod
    def test_sem_rmsea_results() -> None:
        # Could not for the life of me find a way to make type="notclose" work for WebPower, so all my validations assume
        # test_type = "close"
        power_results = power_tests.wp_sem_rmsea_test(n=100, df=4, rmsea0=0, rmsea1=0.116, power=None, alpha=0.05)[
            "power"
        ]
        # wp.sem.rmsea (n = 100, df = 4, rmsea0 = 0, rmsea1 = 0.116, power = NULL, alpha = 0.05)
        # Power for SEM based on RMSEA
        #
        #   n df rmsea0 rmsea1     power alpha
        # 100  4      0  0.116 0.4208173  0.05
        #
        # URL: http://psychstat.org/rmsea
        expected = 0.4208173
        assert expected == pytest.approx(power_results, abs=1e-05)

        n_results = power_tests.wp_sem_rmsea_test(n=None, df=4, rmsea0=0, rmsea1=0.116, power=0.8, alpha=0.05)["n"]
        # wp.sem.rmsea (n = NULL, df = 4, rmsea0 = 0, rmsea1 = 0.116, power = 0.8, alpha = 0.05)
        # Power for SEM based on RMSEA
        #
        #        n df rmsea0 rmsea1 power alpha
        # 222.7465  4      0  0.116   0.8  0.05
        #
        # URL: http://psychstat.org/rmsea
        expected = 223
        assert expected == n_results

        rmsea1_results = power_tests.wp_sem_rmsea_test(n=100, df=4, rmsea0=0, rmsea1=None, power=0.8, alpha=0.05)[
            "rmsea1"
        ]
        # wp.sem.rmsea (n = 100, df = 4, rmsea0 = 0, rmsea1 = NULL, power = 0.8, alpha = 0.05)
        # Power for SEM based on RMSEA
        #
        #   n df rmsea0    rmsea1 power alpha
        # 100  4      0 0.1736082   0.8  0.05
        #
        # URL: http://psychstat.org/rmsea
        expected = 0.1736082
        assert expected == pytest.approx(rmsea1_results, abs=1e-05)

        df_results = power_tests.wp_sem_rmsea_test(n=100, df=None, rmsea0=0, rmsea1=0.2, power=0.8, alpha=0.1)["df"]
        # wp.sem.rmsea(n=100, df=NULL, rmsea0=0, rmsea1=0.2, power=0.8, alpha=0.05)
        # Power for SEM based on RMSEA
        #
        #       n       df rmsea0 rmsea1 power alpha
        #     100 1.92255      0    0.2   0.8  0.05
        #
        # URL: http://psychstat.org/rmsea
        expected = 2
        assert df_results == expected

        alpha_results = power_tests.wp_sem_rmsea_test(n=50, df=5, rmsea0=0, rmsea1=0.2, power=0.8, alpha=None)["alpha"]
        # wp.sem.rmsea(n=50, df=5, rmsea0=0, rmsea1=0.2, power=0.8, alpha=NULL)
        # Power for SEM based on RMSEA
        #
        #      n df rmsea0 rmsea1 power     alpha
        #     50  5      0    0.2   0.8 0.1195863
        #
        # URL: http://psychstat.org/rmsea
        expected = 0.1195863
        assert expected == pytest.approx(alpha_results, abs=1e-03)


# MISCELLANEOUS


class TestMediation:
    @staticmethod
    def test_mediation_results() -> None:
        power_results = power_tests.wp_mediation_test(
            n=100, power=None, a=0.5, b=0.5, var_x=1, var_y=1, var_m=1, alpha=0.05
        )["power"]
        # wp.mediation(n = 100, power = NULL, a = 0.5, b = 0.5, varx = 1, vary = 1, varm = 1, alpha = 0.05)
        # Power for simple mediation
        #
        #   n     power   a   b varx varm vary alpha
        # 100 0.9337271 0.5 0.5    1    1    1  0.05
        #
        # URL: http://psychstat.org/mediation
        expected = 0.9337271
        assert expected == pytest.approx(power_results, abs=1e-05)

        n_results = power_tests.wp_mediation_test(
            n=None, power=0.9, a=0.5, b=0.5, var_x=1, var_y=1, var_m=1, alpha=0.05
        )["n"]
        # wp.mediation(n = NULL, power = 0.9, a = 0.5, b = 0.5, varx = 1, vary = 1, varm = 1, alpha = 0.05)
        # Power for simple mediation
        #
        #        n power   a   b varx varm vary alpha
        # 87.56182   0.9 0.5 0.5    1    1    1  0.05
        #
        # URL: http://psychstat.org/mediation
        expected = 88
        assert expected == n_results

        a_results = power_tests.wp_mediation_test(
            n=100, power=0.9, a=None, b=0.5, var_x=1, var_y=1, var_m=1, alpha=0.05
        )["a"]
        # wp.mediation(n = 100, power = 0.9, a = NULL, b = 0.5, varx = 1, vary = 1, varm = 1, alpha = 0.05)
        # Power for simple mediation
        #
        #   n power         a   b varx varm vary alpha
        # 100   0.9 0.7335197 0.5    1    1    1  0.05
        #
        # URL: http://psychstat.org/mediation
        expected = 0.7335197
        assert expected == pytest.approx(a_results, abs=1e-03)

        b_results = power_tests.wp_mediation_test(
            n=150, power=0.8, a=0.5, b=None, var_x=1, var_y=1, var_m=1, alpha=0.05
        )["b"]
        # wp.mediation(n = 150, power = 0.8, a = 0.5, b = NULL, varx = 1, vary = 1, varm = 1, alpha = 0.05)
        # Power for simple mediation
        #
        #       n power   a          b varx varm vary alpha
        #     150   0.8 0.5 -0.2876635    1    1    1  0.05
        #
        # URL: http://psychstat.org/mediation
        expected = -0.2876635
        assert expected == pytest.approx(b_results, abs=1e-04)

        alpha_results = power_tests.wp_mediation_test(
            n=200, power=0.8, a=0.5, b=-0.2, var_x=1, var_y=1, var_m=1, alpha=None
        )["alpha"]
        # wp.mediation(n = 200, power = 0.80, a = 0.5, b = -0.2, varx = 1, vary = 1, varm = 1, alpha = NULL)
        # Power for simple mediation
        #
        #       n power   a    b varx varm vary     alpha
        #     200   0.8 0.5 -0.2    1    1    1 0.1323648
        #
        # URL: http://psychstat.org/mediation
        expected = 0.1323648
        assert expected == pytest.approx(alpha_results, abs=1e-04)

        var_y_results = power_tests.wp_mediation_test(
            n=150, power=0.8, a=0.3, b=-0.2876635, var_x=1, var_y=None, var_m=1, alpha=0.05
        )["var_y"]
        # wp.mediation(n = 150, power = 0.80, a = 0.5, b = -0.2876635, varx = 1, vary = NULL, varm = 1, alpha = 0.05)
        # Power for simple mediation
        #
        #       n power   a          b varx varm      vary alpha
        #     150   0.8 0.3 -0.2876635    1    1 0.6777307  0.05
        #
        # URL: http://psychstat.org/mediation
        expected = 0.6777307
        assert expected == pytest.approx(var_y_results, abs=1e-03)


class TestCorrelation:
    @staticmethod
    def test_correlation_results() -> None:
        power_results = power_tests.wp_correlation_test(n=50, r=0.3, power=None, alpha=0.05, alternative="two-sided")[
            "power"
        ]
        # wp.correlation(n=50, r=0.3, alternative="two.sided")
        # Power for correlation
        #
        #        n   r alpha     power
        #       50 0.3  0.05 0.5728731
        # URL: http://psychstat.org/correlation
        expected = 0.5728731
        assert power_results == pytest.approx(expected, abs=1e-05)

        n_results = power_tests.wp_correlation_test(n=None, r=0.3, power=0.8, alpha=0.05, alternative="greater")["n"]
        # wp.correlation(n=NULL, r=0.3, power=0.8, alternative="greater")
        # Power for correlation
        #
        #              n   r alpha power
        #       66.55538 0.3  0.05   0.8
        #
        # URL: http://psychstat.org/correlation
        expected = 67
        assert n_results == expected

        r_results = power_tests.wp_correlation_test(n=200, r=None, power=0.8, alpha=0.10, alternative="less")[
            "effect_size"
        ]
        # wp.correlation(n=200, r=NULL, power=0.8, alternative="less")
        # Power for correlation
        #
        #       n          r alpha power
        #     200 -0.1497613  0.05   0.8
        #
        # URL: http://psychstat.org/correlation
        expected = -0.1497613
        assert r_results == pytest.approx(expected, abs=1e-05)

        alpha_results = power_tests.wp_correlation_test(n=200, r=0.1, power=0.8, alternative="two-sided", alpha=None)[
            "alpha"
        ]
        # wp.correlation(n=200, r=0.1, power=0.8, alternative="greater", alpha=NULL)
        # Power for correlation
        #
        #       n   r     alpha power
        #     200 0.1 0.5221974   0.8
        #
        # URL: http://psychstat.org/correlation
        expected = 0.5221974
        assert alpha_results == pytest.approx(expected, abs=1e-05)


# RANDOMIZED TRIALS
class TestMRT2Arm:
    @staticmethod
    def test_mrt2arm_results() -> None:
        power_main_results = power_tests.wp_mrt2arm_test(
            n=45, f=0.5, J=20, tau11=0.5, sg2=1.25, power=None, alpha=0.05
        )["power"]
        # wp.mrt2arm(n = 45, f = 0.5, J = 20, tau11 = 0.5, sg2 = 1.25, alpha = 0.05, power = NULL)
        # Power analysis for Multileve model Multisite randomized trials with 2 arms
        #
        #        J  n   f tau11  sg2     power alpha
        #       20 45 0.5   0.5 1.25 0.8583253  0.05
        #
        # NOTE: n is the number of subjects per cluster
        # URL: http://psychstat.org/mrt2arm
        expected = 0.8583253
        assert power_main_results == pytest.approx(expected, abs=1e-05)

        power_variance_results = power_tests.wp_mrt2arm_test(
            n=45, f=0.5, J=20, tau11=0.5, sg2=1.25, alpha=0.05, power=None, test_type="variance"
        )["power"]
        # wp.mrt2arm(n = 45, f = 0.5, J = 20, tau11 = 0.5, sg2 = 1.25, alpha = 0.05, power = NULL, type = "variance")
        # Power analysis for Multileve model Multisite randomized trials with 2 arms
        #
        #        J  n   f tau11  sg2     power alpha
        #       20 45 0.5   0.5 1.25 0.9987823  0.05
        #
        # NOTE: n is the number of subjects per cluster
        # URL: http://psychstat.org/mrt2arm
        expected = 0.9987823
        assert power_variance_results == pytest.approx(expected, abs=1e-05)

        power_site_results = power_tests.wp_mrt2arm_test(
            n=5, J=20, tau00=2.5, sg2=1.25, alpha=0.05, power=None, test_type="site"
        )["power"]
        # wp.mrt2arm(n = 5, J = 20, tau00 = 2.5, sg2 = 1.25, alpha = 0.05, power = NULL, type = "site")
        # Multisite randomized trials with 2 arms
        #
        #      J n tau00  sg2     power alpha
        #     20 5   2.5 1.25 0.9999719  0.05
        #
        # NOTE: n is the number of subjects per cluster
        # URL: http://psychstat.org/mrt2arm
        expected = 0.9999719
        assert power_site_results == pytest.approx(expected, abs=1e-05)

        n_results = power_tests.wp_mrt2arm_test(
            n=None, f=0.5, J=20, tau11=0.5, sg2=1.25, alpha=0.05, power=0.8, alternative="one-sided"
        )["n"]
        # wp.mrt2arm(n = NULL, f = 0.5, J =20, tau11 = 0.5, sg2 = 1.25, alpha = 0.05, power = 0.8, alternative = 'one.sided')
        # Multisite randomized trials with 2 arms
        #
        #      J       n   f tau11  sg2 power alpha
        #     20 11.3919 0.5   0.5 1.25   0.8  0.05
        #
        # NOTE: n is the number of subjects per cluster
        # URL: http://psychstat.org/mrt2arm
        expected = 12
        assert n_results == expected

        J_results = power_tests.wp_mrt2arm_test(n=10, J=None, f=0.5, tau11=2.5, sg2=1.25, alpha=0.05, power=0.8)["J"]
        # wp.mrt2arm(n = 10, J = NULL, f = 0.5, tau11 = 2.5, sg2 = 1.25, alpha = 0.05, power = 0.8)
        # Multisite randomized trials with 2 arms
        #
        #           J  n   f tau11  sg2 power alpha
        #     77.2918 10 0.5   2.5 1.25   0.8  0.05
        #
        # NOTE: n is the number of subjects per cluster
        # URL: http://psychstat.org/mrt2arm
        expected = 78
        assert J_results == expected

        f_results = power_tests.wp_mrt2arm_test(
            n=200, J=30, f=None, tau00=1.5, tau11=1.5, sg2=1.25, alpha=0.05, power=0.8
        )["effect_size"]
        # wp.mrt2arm(n=200, J=30, f=NULL, tau00=1.5, tau11=1.5, sg2=1.25, alpha=0.05, power=0.8)
        # Multisite randomized trials with 2 arms
        #
        #      J   n         f tau00 tau11  sg2 power alpha
        #     30 200 0.5845826   1.5   1.5 1.25   0.8  0.05
        #
        # NOTE: n is the number of subjects per cluster
        # URL: http://psychstat.org/mrt2arm
        expected = 0.5845826
        assert f_results == pytest.approx(expected, abs=1e-03)


class TestMRT3Arm:
    @staticmethod
    def test_mrt3arm_results() -> None:
        power_results = power_tests.wp_mrt3arm_test(
            n=30, f1=0.43, f2=0, J=20, tau=0.4, sg2=2.25, alpha=0.05, power=None
        )["power"]
        # wp.mrt3arm(n = 30, f1 = 0.43, J = 20, tau = 0.4, sg2 = 2.25, alpha = 0.05, power = NULL)
        # Multisite randomized trials with 3 arms
        #
        #        J  n   f1 tau  sg2     power alpha
        #       20 30 0.43 0.4 2.25 0.8066964  0.05
        #
        # NOTE: n is the number of subjects per cluster
        expected = 0.8066964
        assert power_results == pytest.approx(expected, abs=1e-05)

        n_results = power_tests.wp_mrt3arm_test(
            n=None, f2=0.43, f1=0, J=20, tau=0.4, sg2=2.25, alpha=0.05, power=0.8, test_type="treatment"
        )["n"]
        # wp.mrt3arm(n = NULL, f2 = 0.43, J = 20, tau = 0.4, sg2 = 2.25, alpha = 0.05, power = 0.8, type="treatment")
        # Multisite randomized trials with 3 arms
        #
        #      J        n   f2 tau  sg2 power alpha
        #     20 87.78486 0.43 0.4 2.25   0.8  0.05
        #
        # NOTE: n is the number of subjects per cluster
        # URL: http://psychstat.org/mrt3arm
        expected = 88
        assert n_results == expected

        j_results = power_tests.wp_mrt3arm_test(
            n=200, f2=0.43, f1=0.15, J=None, tau=0.4, sg2=2.25, alpha=0.05, power=0.8, test_type="omnibus"
        )["J"]
        # wp.mrt3arm(n = 200, f2 = 0.43, f1 = 0.15, J = NULL, tau = 0.4, sg2 = 2.25, alpha = 0.05, power = 0.8, type="omnibus")
        # Multisite randomized trials with 3 arms
        #
        #            J   n   f1   f2 tau  sg2 power alpha
        #     18.82449 200 0.15 0.43 0.4 2.25   0.8  0.05
        #
        # NOTE: n is the number of subjects per cluster
        # URL: http://psychstat.org/mrt3arm
        expected = 19
        assert j_results == expected

        f1_results = power_tests.wp_mrt3arm_test(
            n=250, f1=None, f2=0, J=24, tau=0.5, sg2=3.2, alpha=0.1, power=0.8, alternative="one-sided"
        )["f1"]
        # wp.mrt3arm(n=250, f1=NULL, f2=0, J=24, tau=0.5, sg2=3.2, alpha=0.1, power=0.8, alternative="greater")
        # Multisite randomized trials with 3 arms
        #
        #      J   n        f1 f2 tau sg2 power alpha
        #     24 250 0.2217522  0 0.5 3.2   0.8   0.1
        #
        # NOTE: n is the number of subjects per cluster
        # URL: http://psychstat.org/mrt3arm
        expected = 0.2217522
        assert f1_results == pytest.approx(expected, abs=1e-05)


class TestCRT2Arm:
    @staticmethod
    def test_crt2arm_results() -> None:
        power_results = power_tests.wp_crt2arm_test(f=0.6, n=20, J=10, icc=0.1, alpha=0.05, power=None)["power"]
        # wp.crt2arm(f = 0.6, n = 20, J = 10, icc = 0.1, alpha = 0.05, power = NULL)
        # Cluster randomized trials with 2 arms
        #
        #        J  n   f icc     power alpha
        #       10 20 0.6 0.1 0.5901684  0.05
        #
        # NOTE: n is the number of subjects per cluster.
        # URL: http://psychstat.org/crt2arm
        expected = 0.5901684
        assert power_results == pytest.approx(expected, abs=1e-05)

        n_results = power_tests.wp_crt2arm_test(f=0.8, n=None, J=10, icc=0.1, alpha=0.05, power=0.8)["n"]
        # wp.crt2arm(f = 0.8, n = NULL, J = 10, icc = 0.1, alpha = 0.05, power = 0.8)
        # Cluster randomized trials with 2 arms
        #
        #        J        n   f icc power alpha
        #       10 16.02558 0.8 0.1   0.8  0.05
        #
        # NOTE: n is the number of subjects per cluster.
        # URL: http://psychstat.org/crt2arm
        expected = 17
        assert n_results == expected

        icc_results = power_tests.wp_crt2arm_test(f=0.8, n=100, J=10, icc=None, alpha=0.05, power=0.8)["icc"]
        # wp.crt2arm(f = 0.8, n = 100, J = 10, icc = NULL, alpha = 0.05, power = 0.8)
        # Cluster randomized trials with 2 arms
        #
        #      J   n   f       icc power alpha
        #     10 100 0.8 0.1476605   0.8  0.05
        #
        # NOTE: n is the number of subjects per cluster.
        # URL: http://psychstat.org/crt2arm
        expected = 0.1476605
        assert icc_results == pytest.approx(expected, abs=1e-03)

        f_results = power_tests.wp_crt2arm_test(f=None, n=200, J=20, icc=0.15, alpha=0.05, power=0.8)["effect_size"]
        # wp.crt2arm(f = NULL, n = 200, J = 20, icc = 0.15, alpha = 0.05, power = 0.8)
        # Cluster randomized trials with 2 arms
        #
        #      J   n         f  icc power alpha
        #     20 200 0.5203701 0.15   0.8  0.05
        #
        # NOTE: n is the number of subjects per cluster.
        # URL: http://psychstat.org/crt2arm
        expected = 0.5203701
        assert f_results == pytest.approx(expected, abs=1e-05)

        alpha_results = power_tests.wp_crt2arm_test(f=0.3, n=200, J=20, icc=0.15, alpha=None, power=0.8)["alpha"]
        # wp.crt2arm(f = 0.3, n = 200, J = 20, icc = 0.15, alpha = NULL, power = 0.8)
        # Cluster randomized trials with 2 arms
        #
        #      J   n   f  icc power     alpha
        #     20 200 0.3 0.15   0.8 0.3860032
        #
        # NOTE: n is the number of subjects per cluster.
        # URL: http://psychstat.org/crt2arm
        expected = 0.3860032
        assert alpha_results == pytest.approx(expected, abs=1e-03)

        # Need to figure out how to get reproducible results for J, currently cannot.


class TestCRT3Arm:
    @staticmethod
    def test_crt3arm_results() -> None:
        power_results = power_tests.wp_crt3arm_test(f=0.5, n=20, J=10, icc=0.1, alpha=0.05, power=None)["power"]
        # wp.crt3arm(f = 0.5, n = 20, J = 10, icc = 0.1, alpha = 0.05, power = NULL)
        # Cluster randomized trials with 3 arms
        #
        #        J  n   f icc     power alpha
        #       10 20 0.5 0.1 0.3940027  0.05
        #
        # NOTE: n is the number of subjects per cluster.
        # URL: http://psychstat.org/crt3arm
        expected = 0.3940027
        assert power_results == pytest.approx(expected, abs=1e-05)

        f_results = power_tests.wp_crt3arm_test(
            f=None, n=100, J=15, icc=0.15, alpha=0.05, power=0.8, alternative="one-sided", test_type="treatment"
        )["effect_size"]
        # wp.crt3arm(f = NULL, n = 100, J = 15, icc = 0.15, alpha = 0.05, power = 0.8, alternative = "one.sided", type = "treatment")
        # Cluster randomized trials with 3 arms
        #
        #      J   n         f  icc power alpha
        #     15 100 0.6646342 0.15   0.8  0.05
        #
        # NOTE: n is the number of subjects per cluster.
        # URL: http://psychstat.org/crt3arm
        expected = 0.6646342
        assert f_results == pytest.approx(expected, abs=1e-05)

        n_results = power_tests.wp_crt3arm_test(f=0.8, n=None, J=10, icc=0.1, alpha=0.05, power=0.8)["n"]
        # wp.crt3arm(f = 0.8, n = NULL, J = 10, icc = 0.1, alpha = 0.05, power = 0.8)
        # Cluster randomized trials with 3 arms
        #
        #        J        n   f icc  power alpha
        #       10 27.25145 0.8  0.1   0.8  0.05
        #
        # NOTE: n is the number of subjects per cluster
        expected = 28
        assert n_results == expected

        J_results = power_tests.wp_crt3arm_test(
            f=0.5, n=200, J=None, icc=0.4, alpha=0.05, power=0.8, test_type="omnibus", alternative="one-sided"
        )["J"]
        # wp.crt3arm(f = 0.5, n = 200, J = NULL, icc = 0.4, alpha = 0.05, power = 0.8,
        # +            alternative = "one.sided", type = "omnibus")
        # Cluster randomized trials with 3 arms
        #
        #            J   n   f icc power alpha
        #     18.87456 200 0.5 0.4   0.8  0.05
        #
        # NOTE: n is the number of subjects per cluster.
        # URL: http://psychstat.org/crt3arm
        expected = 19
        assert J_results == expected

        icc_results = power_tests.wp_crt3arm_test(
            f=0.8, n=575, J=50, icc=None, alpha=0.05, power=0.8, test_type="main"
        )["icc"]
        # wp.crt3arm(f = 0.8, n = 575, J = 50, icc = NULL, alpha = 0.05, power = 0.8, type="main")
        # Cluster randomized trials with 3 arms
        #
        #      J   n   f      icc power alpha
        #     50 575 0.8 0.868855   0.8  0.05
        #
        # NOTE: n is the number of subjects per cluster.
        # URL: http://psychstat.org/crt3arm
        expected = 0.868855
        assert icc_results == pytest.approx(expected, abs=1e-05)

        alpha_results = power_tests.wp_crt3arm_test(
            f=0.8, n=575, J=50, icc=0.8, alpha=None, power=0.8, test_type="treatment"
        )["alpha"]
        # wp.crt3arm(f = 0.8, n = 575, J = 50, icc = 0.8, alpha = NULL, power = 0.8, type="treatment")
        # Cluster randomized trials with 3 arms
        #
        #      J   n   f icc power      alpha
        #     50 575 0.8 0.8   0.8 0.08915664
        #
        # NOTE: n is the number of subjects per cluster.
        # URL: http://psychstat.org/crt3arm
        expected = 0.08915664
        assert alpha_results == pytest.approx(expected, abs=1e-05)


if __name__ == "__main__":
    pytest.main()
