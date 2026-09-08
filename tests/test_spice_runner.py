"""Tests for ngspice log parsing, the convergence ladder and model resolution.

The parsing tests run against canned logs captured from a real ngspice 42, so
they stay fast and do not need the binary.  A handful of tests marked ``slow``
invoke ngspice for real.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from cforge.spice import modellib
from cforge.spice import runner as runner_module
from cforge.spice.runner import (
    RETRY_LADDER,
    NgspiceNotFound,
    SpiceRun,
    _has_convergence_failure,
    _insert_before_end,
    measurements_from_log,
    ngspice_version,
    run,
)

FIXTURES = Path(__file__).parent / "fixtures"
HAVE_NGSPICE = shutil.which("ngspice") is not None
needs_ngspice = pytest.mark.skipif(not HAVE_NGSPICE, reason="ngspice is not on PATH")


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class TestMeasurementParsing:
    def test_parses_a_clean_log(self) -> None:
        values, failed = measurements_from_log(fixture("ngspice_ok.log"))
        assert failed == []
        assert values["fc_hz"] == pytest.approx(1000.908)
        assert values["gain_db_passband"] == pytest.approx(-8.685890e-03)
        assert values["atten_db_at_10fc"] == pytest.approx(-20.03484)

    def test_ignores_the_targ_and_trig_columns(self) -> None:
        values, _ = measurements_from_log(fixture("ngspice_ok.log"))
        assert values["vrip"] == pytest.approx(4.83291e-02)
        assert "targ" not in values
        assert "trig" not in values

    def test_parses_si_suffixes(self) -> None:
        values, _ = measurements_from_log(fixture("ngspice_ok.log"))
        assert values["rout_ohm"] == pytest.approx(4700.0)
        assert values["tsettle"] == pytest.approx(120e-6)

    def test_names_are_lowercased(self) -> None:
        values, _ = measurements_from_log("FC_HZ = 1.0e3\n")
        assert values == {"fc_hz": pytest.approx(1000.0)}

    def test_skips_diagnostic_lines_that_look_like_assignments(self) -> None:
        text = "Warning: gmin = 1e-12 stepped\nError: reltol = 0.001 too tight\nfc_hz = 2000\n"
        values, _ = measurements_from_log(text)
        assert values == {"fc_hz": pytest.approx(2000.0)}

    def test_empty_log_yields_nothing_and_does_not_raise(self) -> None:
        assert measurements_from_log("") == ({}, [])


class TestFailedMeasurements:
    def test_failed_measurement_is_recorded_not_crashed_on(self) -> None:
        values, failed = measurements_from_log(fixture("ngspice_failed_meas.log"))
        assert "settle_time_s" in failed
        assert "impossible" in failed
        assert "settle_time_s" not in values

    def test_surrounding_measurements_still_parse(self) -> None:
        values, _ = measurements_from_log(fixture("ngspice_failed_meas.log"))
        assert values["overshoot_pct"] == pytest.approx(3.421)
        assert values["fc_hz"] == pytest.approx(1000.908)

    def test_both_ngspice_failure_wordings_are_understood(self) -> None:
        # "name = failed" (older) and the echoed card (ngspice 42).
        values, failed = measurements_from_log(
            "a_meas = failed\n .meas tran b_meas when v(out)=9 failed!\nc_meas = 1.0\n"
        )
        assert set(failed) == {"a_meas", "b_meas"}
        assert values == {"c_meas": pytest.approx(1.0)}

    def test_a_failed_measurement_maps_to_none_via_spicerun(self) -> None:
        values, failed = measurements_from_log(fixture("ngspice_failed_meas.log"))
        run_result = SpiceRun(
            ok=True,
            measurements=values,
            stdout="",
            stderr="",
            converged=True,
            attempts=1,
            failed_measurements=failed,
        )
        assert run_result.get("settle_time_s") is None
        assert run_result.get("OVERSHOOT_PCT") == pytest.approx(3.421)
        assert "settle_time_s" in run_result.all_names()


class TestConvergenceDetection:
    def test_detects_non_convergence_in_a_canned_failure_log(self) -> None:
        assert _has_convergence_failure(fixture("ngspice_nonconvergent.log")) is True

    def test_clean_log_is_not_flagged(self) -> None:
        assert _has_convergence_failure(fixture("ngspice_ok.log")) is False

    @pytest.mark.parametrize(
        "marker",
        [
            "Warning: singular matrix: check node x",
            "Error: no convergence in dc analysis",
            "Error: Transient op failed, timestep too small",
            "doAnalyses: iteration limit reached",
        ],
    )
    def test_each_marker_is_recognised(self, marker: str) -> None:
        assert _has_convergence_failure(marker) is True

    def test_ladder_order_matches_the_documented_escalation(self) -> None:
        labels = [label for label, _ in RETRY_LADDER]
        assert labels == ["baseline", "rshunt", "gmin", "gear", "nodeset"]
        directives = dict(RETRY_LADDER)
        assert directives["rshunt"] == ".options rshunt=1e12"
        assert "gmin=1e-10" in directives["gmin"]
        assert "abstol=1e-10" in directives["gmin"]
        assert "reltol=1e-3" in directives["gmin"]
        assert directives["gear"] == ".options method=gear"


class TestNetlistSurgery:
    def test_directive_goes_before_the_end_card(self) -> None:
        out = _insert_before_end("* title\nR1 a 0 1k\n.end\n", ".options gmin=1e-10")
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        assert lines[-2] == ".options gmin=1e-10"
        assert lines[-1] == ".end"

    def test_netlist_without_end_still_gets_the_directive(self) -> None:
        out = _insert_before_end("R1 a 0 1k\n", ".options gmin=1e-10")
        assert ".options gmin=1e-10" in out


RC_NETLIST = """* cforge runner test: 1 kHz RC low-pass
V1 in 0 DC 0 AC 1
R1 in out 15.8k
C1 out 0 10n
.save v(out)
.ac dec 100 1 1e6
.meas ac fc_hz WHEN vdb(out)=-3.01
.meas ac gain_db_passband FIND vdb(out) AT=10
.end
"""

BAD_NETLIST = """* cforge runner test: deliberately singular
V1 a 0 DC 1
V2 a 0 DC 2
R1 a 0 1k
.op
.end
"""


@needs_ngspice
class TestLiveNgspice:
    def test_version_string_is_reported(self) -> None:
        assert "ngspice" in ngspice_version().lower()

    def test_runs_an_rc_lowpass_and_measures_fc(self, tmp_path: Path) -> None:
        result = run(RC_NETLIST, timeout_s=60, workdir=tmp_path)
        assert result.ok is True
        assert result.converged is True
        assert result.attempts == 1
        assert result.get("fc_hz") == pytest.approx(1007.0, rel=0.02)
        assert result.get("gain_db_passband") == pytest.approx(0.0, abs=0.1)

    def test_writes_the_netlist_and_log_into_the_workdir(self, tmp_path: Path) -> None:
        run(RC_NETLIST, timeout_s=60, workdir=tmp_path)
        assert (tmp_path / "attempt1" / "circuit.cir").is_file()
        assert (tmp_path / "attempt1" / "ngspice.log").is_file()

    def test_non_convergence_climbs_the_ladder_and_is_not_a_spec_failure(
        self, tmp_path: Path
    ) -> None:
        result = run(BAD_NETLIST, timeout_s=30, workdir=tmp_path)
        assert result.converged is False
        assert result.ok is False
        assert result.attempts == len(RETRY_LADDER)
        assert "not a requirement failure" in result.error
        assert result.options_applied[1] == ".options rshunt=1e12"


class TestTimeout:
    def test_timeout_returns_cleanly_rather_than_hanging(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Simulating a hung solver deterministically: a real long-running
        # netlist would make this test slow and flaky.
        def fake_run(*args: object, **kwargs: object) -> object:
            raise subprocess.TimeoutExpired(cmd="ngspice", timeout=1)

        monkeypatch.setattr(runner_module.subprocess, "run", fake_run)
        monkeypatch.setattr(runner_module, "ngspice_executable", lambda: "/usr/bin/ngspice")
        result = run(RC_NETLIST, timeout_s=1, workdir=tmp_path, max_attempts=1)
        assert result.ok is False
        assert result.converged is False
        assert result.measurements == {}
        assert "timeout" in result.error
        assert "1s" in result.error

    def test_missing_ngspice_raises_with_install_instructions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner_module.shutil, "which", lambda _name: None)
        with pytest.raises(NgspiceNotFound) as exc:
            run("* empty\n.end\n")
        assert "brew install ngspice" in str(exc.value)
        assert "apt install ngspice" in str(exc.value)


class TestModelLib:
    def test_generic_models_all_resolve(self) -> None:
        modellib.invalidate_cache()
        for name in ("1N4148", "2N3904", "2N7002", "OPAMP_GENERIC", "OPAMP_FAST"):
            text = modellib.resolve(name)
            assert name.lower() in text.lower()

    def test_resolution_is_case_insensitive(self) -> None:
        assert modellib.resolve("opamp_generic") == modellib.resolve("OPAMP_GENERIC")

    def test_subckt_block_is_captured_through_ends(self) -> None:
        text = modellib.resolve("OPAMP_GENERIC")
        lowered = text.lower()
        # Macromodels pull in internal .model cards first so the netlist is
        # self-contained (OPAMP_GENERIC needs DCLAMP).
        assert ".subckt opamp_generic" in lowered
        assert lowered.strip().endswith(".ends opamp_generic")
        assert ".model dclamp" in lowered

    def test_resolve_includes_internal_model_dependencies(self) -> None:
        text = modellib.resolve("OPAMP_GENERIC")
        assert text.lower().index(".model dclamp") < text.lower().index(".subckt")

    def test_model_continuation_lines_are_included(self) -> None:
        text = modellib.resolve("2N3904")
        assert "BF=416.4" in text
        assert text.count("\n") >= 2

    def test_missing_model_names_the_file_to_create(self) -> None:
        with pytest.raises(modellib.ModelNotFound) as exc:
            modellib.resolve("lmv321_nonexistent")
        message = str(exc.value)
        assert "lmv321_nonexistent.lib" in message
        assert "models/README.md" in message
        assert "Known models" in message

    def test_resolve_many_deduplicates_and_keeps_order(self) -> None:
        text = modellib.resolve_many(["1N4148", "1N4148", "2N3904"])
        assert text.lower().count(".model 1n4148") == 1
        assert text.index("1N4148") < text.index("2N3904")

    def test_user_library_overrides_generic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "vendor.lib").write_text(
            ".model 1N4148 D(IS=1e-99 N=9.99)\n", encoding="utf-8"
        )
        monkeypatch.setenv("CFORGE_MODEL_PATH", str(tmp_path))
        modellib.invalidate_cache()
        try:
            assert "N=9.99" in modellib.resolve("1N4148")
        finally:
            modellib.invalidate_cache()
