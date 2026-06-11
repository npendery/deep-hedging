from deephedge.instruments import EuropeanOption


def test_european_option_fields_and_default_kind():
    opt = EuropeanOption(strike=100.0, maturity=0.5)
    assert opt.strike == 100.0
    assert opt.maturity == 0.5
    assert opt.kind == "call"


def test_european_option_put_kind():
    opt = EuropeanOption(strike=90.0, maturity=1.0, kind="put")
    assert opt.kind == "put"
