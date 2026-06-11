"""Tests for README.md (Group 12, Task 12.13)."""
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_exists_and_has_required_sections():
    assert README.exists(), "README.md does not exist yet"
    text = README.read_text()
    # leads with the project title
    assert text.lstrip().startswith("# Deep Hedging")
    # embeds the money-shot figure from figures/
    assert "](figures/" in text and "![" in text
    # install + reproduce instructions
    assert "pip install -e ." in text
    assert "## Reproduce" in text
    assert "scripts/run_experiment.py" in text or "scripts/make_figures.py" in text
    # links onward to the full results report and the design spec
    assert "report.md" in text
    assert "docs/specs/2026-06-10-deep-hedging-design.md" in text
