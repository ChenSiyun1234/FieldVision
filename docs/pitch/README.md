# Pitch numbers

The final slides are [`handoff/presentation/FieldVision.pdf`](../../handoff/presentation/FieldVision.pdf). The
result slides are also in [`handoff/figures/results/`](../../handoff/figures/results/) as images.

| File | What it holds |
|---|---|
| `pitch_numbers.py` | Every number quoted in the talk, with the logged run it comes from. `gap_checks.py` reads it, so a number changed here changes the checks |
| `limits.md` | What the results do not show, written for questions after the talk |

The clean tests (rules written first, each place run once): four of five within 5 points of USDA. Michigan
2025 14.4% against 14.7%, Lancaster County PA 2023 26.6% against 27.4%, New York 2025 48.7% against 53.2%,
Wisconsin 2025 16.6% against 21.6%; Pennsylvania 2025 missed, 15.2% against 32.8%. The full record is in
[`docs/RESULTS.md`](../RESULTS.md).
