# Pitch pack: the model module

Everything needed to put the model part into the 5-minute talk. Each figure is a 1920 x 1080 PNG
with its own title, so it can go on a slide as it is.

**Status.** Complete except for the results still running. A dashed "to come" box marks each of
them, and the list is at the bottom of this page.

## What is here

| File | What it shows | Use |
|---|---|---|
| `module_diagram.png` | The module in four boxes: what goes in, the evidence check, the classifier, the adding-up, and what is shown beside every total | Main slide 1 |
| `fig3_labelled_fields.png` | The organisers' labelled fields: small model 42 of 44, big model 37, always grain 32 | Main slide 2 |
| `fig5_counties.png` | County totals against the 2022 Census: where the first method failed, the second method, the Michigan test | Main slide 3 |
| `state_2025.png` | Maryland 2025 state acres beside the USDA figure, with what is shown beside it | Main slide 3 or 4, once it has a number |
| `fig1_signal.png` | What the small model keys on: moisture just before harvest | Backup, "how does it work?" |
| `fig2_new_country.png` | Small model against bigger ones in a country they never saw | Backup, "why not a bigger model?" |
| `fig4_date_dropping.png` | What dropping observation dates in training did for the big model | Backup, "what else did you try?" |
| `script.md` | What to say, line by line, with Chinese under each line; five likely questions with answers | For the speaker |
| `limits.md` | What we do not claim, and the source of every number | Read before changing any wording |
| `pitch_numbers.py`, `draw.py` | The numbers with their sources, and the script that draws all seven images | To redraw |

The model part has about 60 to 90 seconds, so three slides are enough: the diagram, the labelled
fields, and the adding-up. The other figures are for questions.

## Plain words used in the pack

- **Clearly seen, partly seen, not judged.** How well the satellite saw the harvest of a field:
  the last image before the cut and the first image after it are at most 12 days apart, 12 to 20
  days apart, or further apart (or no harvest was found).
- **Small model.** Logistic regression on six measurements of the crop at harvest. **Big model.**
  Boosted trees on 135 measurements.
- **Field-year.** One field in one season. The organisers' set is 44 field-years from 2021-2023.
- **First method, second method.** Two ways of adding field calls up to a county. The first set
  a local cut-off from a nearly all-grain county next door. The second counts silage only in
  two-week harvest windows where enough fields were clearly seen, and counts the other windows
  as grain.
- **Corn not seen at harvest.** The share of a county's corn area that falls in those other
  windows. It is shown next to every total.
- **ROC-AUC.** How well a model ranks silage above grain: 0.5 is guessing, 1.0 is perfect.

## Wording that must not change

- The county totals **failed** their Maryland test. Do not write "validated", "accurate" or
  "matches the Census" for any county or state number.
- The second method is **not confirmed** until the Michigan result is in the pack.
- Say **42 of 44**, not a percentage, and keep "always grain gets 32" next to it.
- The model was trained on European fields only. The 44 labelled fields were used for testing,
  never for training.

## To redraw

```
python docs/pitch/draw.py              # all seven images
python docs/pitch/draw.py counties     # one of: module signal new_country labelled_fields date_dropping counties state
```

Needs matplotlib. To fill a "to come" box, put the value and its source in `pitch_numbers.py`
and run the script again.

## Still to come

| Result | Where it will appear |
|---|---|
| Michigan 2022, two counties: the clean test of the second method | `fig5_counties.png`, script slot A, question 3, `limits.md` |
| Maryland 2025 state acres from the random sample | `state_2025.png`, script slot B, `limits.md` |
| Pennsylvania and Wisconsin 2025 | `state_2025.png` |
| One test of the big model trained with dates dropped, on the labelled fields | `fig3_labelled_fields.png`, `fig4_date_dropping.png`, `limits.md` |
| A model that also reads the pixels around the field | A line in `limits.md`, if it is logged in time |
