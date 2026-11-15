"""Price impact models on made-up numbers (no files, no network)."""

import numpy as np
import pandas as pd

from .price_impact import fit, leave_one_year_out, silage_effect, train


def toy(n=20, b=-0.5):
    """Made-up seasons where log price = 1 + b * log stocks-to-use exactly."""
    years = np.arange(2001, 2001 + n)
    stu = np.linspace(0.08, 0.20, n)[np.random.default_rng(0).permutation(n)]
    df = pd.DataFrame({"year": years, "log_stu": np.log(stu), "yield": 180.0, "prod": 15e9,
                       "end_stocks": stu * 13e9, "use": 13e9})
    df["log_price"] = 1 + b * df["log_stu"]
    df["log_price_prev"] = df["log_price"].shift(1).fillna(df["log_price"].iloc[0])
    df["dlog_price"] = df["log_price"].diff().fillna(0)
    df["dlog_prod"] = 0.0
    df["price_my"] = np.exp(df["log_price"])
    return df


def test_fit_recovers_a_line():
    x = np.linspace(-0.1, 0.1, 21)
    coef, sd = fit(x, 0.02 - 0.8 * x)
    assert np.allclose(coef, [0.02, -0.8]) and sd < 1e-9


def test_stocks_model_learns_and_predicts_held_out_years():
    df = toy()
    cv = leave_one_year_out(df, years=range(2001, 2021))
    stocks = cv[cv["model"] == "stocks"]
    assert stocks["abs_err_pct"].max() <= 0.2, stocks           # predictions are rounded to the cent


def test_leave_one_year_out_never_sees_the_test_year():
    df = toy()
    df.loc[df["year"] == 2010, ["log_price", "price_my"]] = [5.0, np.exp(5.0)]     # an outlier year
    cv = leave_one_year_out(df, years=range(2001, 2021))
    row = cv[(cv["model"] == "stocks") & (cv["year"] == 2010)].iloc[0]
    assert row["predicted"] < 0.5 * row["actual"]            # its own outlier did not pull its prediction


def test_more_silage_means_lower_stocks_and_higher_price():
    df = toy()
    coef = train(df, "stocks", range(2001, 2021))
    pct, dprod = silage_effect(df, "stocks", coef, 2010, 1_000_000)
    assert dprod == -180_000_000 and pct > 0


def run_all():
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS ", name)
            n += 1
    print(f"\n{n} tests passed")
    return n
