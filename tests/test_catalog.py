"""Every shipped pattern must load and validate, and selection must be deterministic."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from cforge.catalog import select as sel
from cforge.catalog.loader import (
    REQUIRED_FILES,
    Pattern,
    PatternError,
    default_pattern_root,
    load_all,
    load_pattern,
)
from cforge.models import Spec

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# The pattern each example spec must select when no --topology is forced.
EXPECTED_SELECTION: dict[str, str] = {
    "lp_1k.yaml": "rc_lowpass_1st",
    "sk_butterworth_2k.yaml": "sallen_key_lp2",
    "isrc_10ma.yaml": "isrc_opamp",
}


@pytest.fixture(scope="module")
def patterns() -> list[Pattern]:
    return load_all()


def load_example(name: str) -> Spec:
    return Spec.model_validate(yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8")))


def example_files() -> list[str]:
    return sorted(p.name for p in EXAMPLES.glob("*.yaml"))


class TestLoading:
    def test_catalog_is_not_empty(self, patterns: list[Pattern]) -> None:
        assert patterns

    def test_every_pattern_directory_loads(self, patterns: list[Pattern]) -> None:
        found = {p.id for p in patterns}
        on_disk = {
            d.name
            for d in default_pattern_root().iterdir()
            if d.is_dir() and not d.name.startswith((".", "_"))
        }
        assert found == on_disk

    def test_each_pattern_has_all_required_files(self, patterns: list[Pattern]) -> None:
        for pattern in patterns:
            for name in REQUIRED_FILES:
                assert (pattern.directory / name).is_file(), f"{pattern.id} lacks {name}"

    def test_metadata_is_populated(self, patterns: list[Pattern]) -> None:
        for pattern in patterns:
            assert pattern.meta.name
            assert pattern.meta.block
            assert pattern.meta.provides_meas, f"{pattern.id} declares no measurements"
            assert pattern.notes.strip(), f"{pattern.id} has an empty notes.md"

    def test_design_and_draw_entry_points_exist(self, patterns: list[Pattern]) -> None:
        for pattern in patterns:
            assert callable(pattern.design_module.solve)
            assert callable(pattern.draw_module.draw)

    def test_every_declared_gotcha_check_resolves(self, patterns: list[Pattern]) -> None:
        for pattern in patterns:
            for gotcha in pattern.meta.gotchas:
                if gotcha.check:
                    assert pattern.gotcha_check(gotcha.check) is not None, (
                        f"{pattern.id}: gotcha {gotcha.id} check {gotcha.check} is missing"
                    )

    def test_templates_render_with_synthesized_components(
        self, patterns: list[Pattern]
    ) -> None:
        from cforge.synth.solver import build_testbench, synthesize

        for name, pattern_id in EXPECTED_SELECTION.items():
            if not (EXAMPLES / name).is_file():
                continue
            spec = load_example(name)
            pattern = next(p for p in load_all() if p.id == pattern_id)
            result = synthesize(spec, pattern, refine=False)
            deck = build_testbench(spec, pattern, result.components)
            assert deck.strip().endswith(".end")
            assert ".save" in deck, f"{pattern_id}: .meas needs .save in ngspice batch mode"
            assert ".meas" in deck


class TestLoaderRejectsBadPatterns:
    def test_missing_file_is_reported_by_name(self, tmp_path: Path) -> None:
        source = default_pattern_root() / "rc_lowpass_1st"
        broken = tmp_path / "rc_lowpass_1st"
        shutil.copytree(source, broken)
        (broken / "draw.py").unlink()
        with pytest.raises(PatternError) as exc:
            load_pattern(broken)
        assert "draw.py" in str(exc.value)
        assert "incomplete" in str(exc.value)

    def test_unparseable_yaml_names_the_file(self, tmp_path: Path) -> None:
        source = default_pattern_root() / "rc_lowpass_1st"
        broken = tmp_path / "rc_lowpass_1st"
        shutil.copytree(source, broken)
        (broken / "pattern.yaml").write_text("id: [unclosed\n", encoding="utf-8")
        with pytest.raises(PatternError) as exc:
            load_pattern(broken)
        assert "pattern.yaml" in str(exc.value)

    def test_id_must_match_directory_name(self, tmp_path: Path) -> None:
        source = default_pattern_root() / "rc_lowpass_1st"
        broken = tmp_path / "rc_lowpass_1st"
        shutil.copytree(source, broken)
        text = (broken / "pattern.yaml").read_text(encoding="utf-8")
        (broken / "pattern.yaml").write_text(
            text.replace("id: rc_lowpass_1st", "id: something_else", 1), encoding="utf-8"
        )
        with pytest.raises(PatternError, match="does not match its directory name"):
            load_pattern(broken)

    def test_bad_jinja_reports_the_line(self, tmp_path: Path) -> None:
        source = default_pattern_root() / "rc_lowpass_1st"
        broken = tmp_path / "rc_lowpass_1st"
        shutil.copytree(source, broken)
        (broken / "tb.cir.j2").write_text("* bad\n{% for x in %}\n", encoding="utf-8")
        with pytest.raises(PatternError, match="Jinja2 syntax error"):
            load_pattern(broken)

    def test_design_without_solve_is_rejected(self, tmp_path: Path) -> None:
        source = default_pattern_root() / "rc_lowpass_1st"
        broken = tmp_path / "rc_lowpass_1st"
        shutil.copytree(source, broken)
        (broken / "design.py").write_text("VALUE = 1\n", encoding="utf-8")
        with pytest.raises(PatternError, match="must define solve"):
            load_pattern(broken)

    def test_missing_directory_is_actionable(self, tmp_path: Path) -> None:
        with pytest.raises(PatternError, match="does not exist"):
            load_pattern(tmp_path / "nope")


class TestSelection:
    @pytest.mark.parametrize("filename", example_files())
    def test_example_specs_select_the_expected_pattern(
        self, filename: str, patterns: list[Pattern]
    ) -> None:
        assert filename in EXPECTED_SELECTION, (
            f"{filename} has no expected selection recorded in this test"
        )
        spec = load_example(filename)
        assert sel.select(spec, patterns).id == EXPECTED_SELECTION[filename]

    def test_selection_is_deterministic(self, patterns: list[Pattern]) -> None:
        spec = load_example("lp_1k.yaml")
        chosen = {sel.select(spec, list(reversed(patterns))).id for _ in range(5)}
        assert chosen == {"rc_lowpass_1st"}

    def test_out_of_range_param_disqualifies_a_pattern(self, patterns: list[Pattern]) -> None:
        spec = load_example("lp_1k.yaml")
        spec = spec.model_copy(update={"params": {**spec.params, "fc_hz": 1.0e12}})
        with pytest.raises(sel.NoPatternFound) as exc:
            sel.select(spec, patterns)
        assert "outside range" in str(exc.value)

    def test_unknown_block_is_reported_with_a_fix(self, patterns: list[Pattern]) -> None:
        spec = load_example("lp_1k.yaml").model_copy(update={"block": "mixer"})
        with pytest.raises(sel.NoPatternFound, match="add a pattern"):
            sel.select(spec, patterns)

    def test_forced_topology_wins(self, patterns: list[Pattern]) -> None:
        spec = load_example("lp_1k.yaml").model_copy(update={"topology": "rc_lowpass_1st"})
        assert sel.select(spec, patterns).id == "rc_lowpass_1st"

    def test_forced_unknown_topology_lists_the_alternatives(
        self, patterns: list[Pattern]
    ) -> None:
        spec = load_example("lp_1k.yaml").model_copy(update={"topology": "nope"})
        with pytest.raises(sel.NoPatternFound) as exc:
            sel.select(spec, patterns)
        assert "Available:" in str(exc.value)
        assert "rc_lowpass_1st" in str(exc.value)

    def test_empty_catalog_is_an_actionable_error(self) -> None:
        with pytest.raises(sel.NoPatternFound, match="catalog is empty"):
            sel.select(load_example("lp_1k.yaml"), [])

    def test_explain_mentions_every_pattern(self, patterns: list[Pattern]) -> None:
        text = sel.explain(load_example("lp_1k.yaml"), patterns)
        for pattern in patterns:
            assert pattern.id in text


class TestCentredness:
    def test_centre_of_a_log_range_scores_zero(self) -> None:
        assert sel._centredness(1e3, 1.0, 1e6, log_scaled=True) == pytest.approx(0.0)

    def test_edges_score_one(self) -> None:
        assert sel._centredness(1.0, 1.0, 1e6, log_scaled=True) == pytest.approx(1.0)
        assert sel._centredness(1e6, 1.0, 1e6, log_scaled=True) == pytest.approx(1.0)

    def test_linear_and_log_scaling_differ(self) -> None:
        linear = sel._centredness(2000.0, 1.0, 1e6, log_scaled=False)
        logarithmic = sel._centredness(2000.0, 1.0, 1e6, log_scaled=True)
        assert linear > 0.99
        assert logarithmic < 0.15

    def test_degenerate_range_does_not_divide_by_zero(self) -> None:
        assert sel._centredness(2.0, 2.0, 2.0, log_scaled=False) == 0.0


class TestTagMatching:
    @pytest.mark.parametrize(
        ("tag", "value", "expected"),
        [
            ("unity", 1.0, True),
            ("unity", 2.0, False),
            ("unity_or_noninverting", 2.0, True),
            ("unity_or_noninverting", -2.0, False),
            ("inverting", -2.0, True),
            ("butterworth_or_bessel", "butterworth", True),
            ("butterworth_or_bessel", "chebyshev", False),
            ("any", 42.0, True),
        ],
    )
    def test_tags(self, tag: str, value: object, expected: bool) -> None:
        assert sel._tag_matches(tag, value) is expected
