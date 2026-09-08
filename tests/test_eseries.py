"""Golden-value tests for the E-series snapping module.

These are the load-bearing numbers of the whole tool: if snapping is wrong,
every downstream simulation is silently wrong too.
"""

from __future__ import annotations

import math

import pytest

from cforge import eseries as es


class TestTables:
    def test_series_lengths(self) -> None:
        assert len(es.E6) == 6
        assert len(es.E12) == 12
        assert len(es.E24) == 24
        assert len(es.E48) == 48
        assert len(es.E96) == 96
        assert len(es.E192) == 192

    @pytest.mark.parametrize("name", list(es.SERIES))
    def test_tables_sorted_and_in_first_decade(self, name: str) -> None:
        table = es.SERIES[name]
        assert list(table) == sorted(table)
        assert len(set(table)) == len(table)
        assert all(1.0 <= m < 10.0 for m in table)

    def test_coarse_series_are_subsets_of_finer_ones(self) -> None:
        assert set(es.E6) <= set(es.E12)
        assert set(es.E12) <= set(es.E24)
        assert set(es.E48) <= set(es.E96)
        assert set(es.E96) <= set(es.E192)

    def test_unknown_series_names_the_valid_options(self) -> None:
        with pytest.raises(ValueError, match="E96"):
            es.snap(1000.0, "E7")


class TestSnapGoldenValues:
    @pytest.mark.parametrize(
        ("value", "series", "expected"),
        [
            (4870.0, "E24", 4700.0),
            (4870.0, "E96", 4870.0),
            (1.0e-8, "E12", 1.0e-8),
            (0.0033, "E6", 0.0033),
            (1.0, "E6", 1.0),
            (9.9, "E6", 10.0),
            (10.5e3, "E96", 10.5e3),
            (2.2e-9, "E6", 2.2e-9),
            (68.0, "E12", 68.0),
            (3300.0, "E24", 3300.0),
            (1.5e6, "E6", 1.5e6),
            (47.0e-6, "E6", 47.0e-6),
        ],
    )
    def test_known_values(self, value: float, series: str, expected: float) -> None:
        assert es.snap(value, series) == pytest.approx(expected, rel=1e-9)

    def test_snap_is_idempotent(self) -> None:
        for series in es.SERIES:
            for v in (1.0, 12.3, 4700.0, 1e-9, 8.2e5):
                once = es.snap(v, series)
                assert es.snap(once, series) == pytest.approx(once, rel=1e-12)

    @pytest.mark.parametrize(
        ("coarse", "fine"),
        [("E6", "E12"), ("E12", "E24"), ("E48", "E96"), ("E96", "E192")],
    )
    def test_finer_series_never_worse_within_a_family(self, coarse: str, fine: str) -> None:
        # Only meaningful within a nesting family: E24 is *not* a subset of E48,
        # so E96 can legitimately be worse than E12 for a value like 5.6.
        for v in (1234.0, 5.6e-9, 8.9e4, 3.21, 47.0):
            coarse_err = abs(math.log(es.snap(v, coarse) / v))
            fine_err = abs(math.log(es.snap(v, fine) / v))
            assert fine_err <= coarse_err + 1e-12

    def test_rejects_nonpositive_and_nonfinite(self) -> None:
        for bad in (0.0, -5.0, math.inf, math.nan):
            with pytest.raises(ValueError, match="positive"):
                es.snap(bad, "E24")


class TestDecadeBoundaries:
    @pytest.mark.parametrize("decade", range(-12, 7))
    def test_exact_decade_values_are_fixed_points(self, decade: int) -> None:
        v = 10.0**decade
        for series in es.SERIES:
            assert es.snap(v, series) == pytest.approx(v, rel=1e-9)

    def test_just_below_a_decade_rounds_up_across_the_boundary(self) -> None:
        # 9.9 is closer in log space to 10 than to E6's 6.8.
        assert es.snap(9.9e3, "E6") == pytest.approx(1.0e4, rel=1e-9)
        assert es.snap(9.9e-7, "E12") == pytest.approx(1.0e-6, rel=1e-9)

    def test_just_above_a_decade_rounds_down_across_the_boundary(self) -> None:
        assert es.snap(1.01e3, "E6") == pytest.approx(1.0e3, rel=1e-9)

    def test_snap_up_and_down_bracket_the_value(self) -> None:
        for series in ("E6", "E24", "E96"):
            for v in (1.0, 1.01, 4870.0, 9.99e5, 3.3e-9):
                lo = es.snap_down(v, series)
                hi = es.snap_up(v, series)
                assert lo <= v * (1 + 1e-12)
                assert hi >= v * (1 - 1e-12)
                assert lo <= hi

    def test_snap_up_down_are_exact_on_standard_values(self) -> None:
        assert es.snap_up(4700.0, "E24") == pytest.approx(4700.0)
        assert es.snap_down(4700.0, "E24") == pytest.approx(4700.0)

    def test_snap_up_crosses_the_decade(self) -> None:
        # 6.8 is the top of E6's first decade, so anything above it must go to 10.
        assert es.snap_up(7.0, "E6") == pytest.approx(10.0)
        assert es.snap_down(7.0, "E6") == pytest.approx(6.8)

    def test_neighbours_cross_decades(self) -> None:
        vals = es.neighbours(1.0e3, "E6", window=2)
        assert vals == pytest.approx([470.0, 680.0, 1000.0, 1500.0, 2200.0])


class TestDecadeValues:
    def test_range_is_inclusive_and_sorted(self) -> None:
        vals = es.decade_values("E6", 1.0e-9, 1.0e-8)
        assert vals == pytest.approx(
            [1.0e-9, 1.5e-9, 2.2e-9, 3.3e-9, 4.7e-9, 6.8e-9, 1.0e-8]
        )

    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            es.decade_values("E24", 10.0, 1.0)


class TestSnapRatio:
    def test_beats_independent_snapping_on_a_hard_case(self) -> None:
        # 1.586 is the Sallen-Key Butterworth gain ratio.  On E24 the
        # independent snap gives 16k/10k = 1.600 (+0.88%), while jointly
        # snapping finds 13k/8.2k = 1.5854 (-0.04%).
        r_hi, r_lo = 15860.0, 10000.0
        target = r_hi / r_lo

        indep = (es.snap(r_hi, "E24"), es.snap(r_lo, "E24"))
        joint = es.snap_ratio(r_hi, r_lo, "E24", window=3)

        indep_err = abs(es.ratio_error(indep, target))
        joint_err = abs(es.ratio_error(joint, target))
        assert indep_err > 0.8
        assert joint_err < 0.1
        assert joint_err < indep_err

    def test_never_worse_than_independent_snapping(self) -> None:
        cases = [
            (15860.0, 10000.0),
            (2200.0, 3300.0),
            (98700.0, 1234.0),
            (1.0e5, 3.7e4),
            (470.0, 470.0),
        ]
        for series in ("E6", "E12", "E24", "E96"):
            for r_hi, r_lo in cases:
                target = r_hi / r_lo
                indep = (es.snap(r_hi, series), es.snap(r_lo, series))
                joint = es.snap_ratio(r_hi, r_lo, series, window=3)
                assert abs(es.ratio_error(joint, target)) <= abs(
                    es.ratio_error(indep, target)
                ) + 1e-9

    def test_returns_standard_values(self) -> None:
        a, b = es.snap_ratio(15860.0, 10000.0, "E24")
        assert es.snap(a, "E24") == pytest.approx(a)
        assert es.snap(b, "E24") == pytest.approx(b)

    def test_exact_ratio_is_preserved_exactly(self) -> None:
        a, b = es.snap_ratio(2200.0, 1000.0, "E24")
        assert (a / b) == pytest.approx(2.2, rel=1e-12)


class TestSeriesCombo:
    def test_single_part_when_the_standard_value_is_close_enough(self) -> None:
        parts = es.series_combo(4700.0, "E24", max_parts=2, tolerance_percent=1.0)
        assert parts == pytest.approx([4700.0])

    def test_two_parts_beat_one_when_the_grid_is_coarse(self) -> None:
        target = 1234.0
        one = abs(100.0 * (es.snap(target, "E6") - target) / target)
        parts = es.series_combo(target, "E6", max_parts=2, tolerance_percent=0.5)
        assert len(parts) == 2
        best = min(
            abs(100.0 * (sum(parts) - target) / target),
            abs(100.0 * (1.0 / sum(1.0 / p for p in parts) - target) / target),
        )
        assert best < one

    def test_capacitor_mode_returns_realisable_parts(self) -> None:
        parts = es.series_combo(4.4e-9, "E6", max_parts=2, kind="capacitor")
        assert all(p > 0 for p in parts)
        assert sum(parts) == pytest.approx(4.4e-9, rel=0.05)

    def test_rejects_bad_arguments(self) -> None:
        with pytest.raises(ValueError, match="max_parts"):
            es.series_combo(1000.0, "E24", max_parts=0)
        with pytest.raises(ValueError, match="kind"):
            es.series_combo(1000.0, "E24", kind="inductor")


class TestSnapRcPair:
    @pytest.mark.parametrize("fc", [100.0, 1000.0, 2000.0, 10e3, 100e3])
    def test_tau_error_under_two_percent(self, fc: float) -> None:
        tau = 1.0 / (2.0 * math.pi * fc)
        r, c = es.snap_rc_pair(tau, c_series="E6", r_series="E96")
        assert abs(100.0 * (r * c - tau) / tau) < 2.0

    def test_returns_standard_values_from_each_series(self) -> None:
        tau = 1.0 / (2.0 * math.pi * 1000.0)
        r, c = es.snap_rc_pair(tau, c_series="E6", r_series="E96")
        assert es.snap(c, "E6") == pytest.approx(c, rel=1e-9)
        assert es.snap(r, "E96") == pytest.approx(r, rel=1e-9)

    def test_prefers_sane_impedances(self) -> None:
        tau = 1.0 / (2.0 * math.pi * 1000.0)
        r, c = es.snap_rc_pair(tau, r_range=(1e3, 1e5))
        assert 1e3 <= r <= 1e5

    def test_empty_capacitor_range_is_an_actionable_error(self) -> None:
        with pytest.raises(ValueError, match="widen c_range"):
            es.snap_rc_pair(1e-3, c_series="E6", c_range=(1.05e-9, 1.06e-9))
