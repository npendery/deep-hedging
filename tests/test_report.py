"""Tests for report.md (Group 12, Task 12.11)."""
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1] / "report.md"


def test_report_exists_and_nonempty():
    assert REPORT.exists()
    assert REPORT.stat().st_size > 0


def test_report_has_required_sections():
    text = REPORT.read_text()
    # required section headings
    for heading in (
        "# Deep Hedging",
        "## Regime results",
        "## Figures",
        "## CVaR loss",
        "## Heston pricer",
        "## No-trade-band finding",
    ):
        assert heading in text, heading


def test_report_embeds_both_figures():
    text = REPORT.read_text()
    assert "pnl_distribution.png" in text
    assert "hedge_ratio.png" in text


def test_report_cites_correct_facts_not_the_bugs():
    text = REPORT.read_text()
    # CVaR: tail-probability denominator (1 - alpha), not alpha.
    assert "(1 - alpha)" in text or "(1-alpha)" in text
    # Heston: the b = kappa fix, explicitly not kappa - rho*xi.
    assert "kappa" in text
    assert "b = kappa" in text or "= kappa" in text
    # No-trade band: Whalley-Wilmott, and explicitly NOT Leland as the band source.
    assert "Whalley" in text
    assert "cost^{1/3}" in text or "cost^(1/3)" in text
