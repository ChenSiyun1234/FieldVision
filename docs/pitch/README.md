# Pitch pack: the model module

Everything needed to put the model part into the 5-minute talk. Each figure is a 1920 x 1080 PNG
with its own title, so it can go on a slide as it is.

**Status.** Michigan, Maryland 2025 and the last test on the labelled fields are in. Still to
come: Pennsylvania and Wisconsin 2025 (a dashed "to come" mark in `state_2025.png`).

## The three sentences the model part stands on

1. Field by field it works: trained only on European fields, the small model gets 42 of the
   organisers' 44 labelled field-years right; always saying grain gets 32.
2. Adding up is not solved: the first county method failed its test in Maryland; a plain count
   then passed a test on two Michigan counties.
3. The state figure reads low: Maryland 2025, 22,700 acres of silage against USDA's 40,000
   (33,135 once double-crop corn is scored too, which was done after the result was seen). We
   show that next to the number.

## What is here

| File | What it shows | Use |
|---|---|---|
| `module_diagram.png` | The module in four boxes: what goes in, the evidence check, the classifier, the adding-up, and what is shown beside every total | Main slide 1 |
| `fig3_labelled_fields.png` | The organisers' labelled fields: small model 42 of 44, big model 37 (39 with dates dropped), always grain 32 | Main slide 2 |
| `state_2025.png` | Maryland 2025 state acres beside the USDA figure: ours reads low | Main slide 3 |
| `fig5_counties.png` | County totals against the 2022 Census: the failure in Maryland, the pass in Michigan | Main slide 3 (with the state figure) or backup |
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
- **Plain count.** The share of the clearly seen corn area that the small model calls silage.
  County and state acres are that share times the corn acres from the field boundaries.
- **First method.** An earlier way of adding up, with a local cut-off taken from a nearly
  all-grain county next door. It failed its test and is not used.
- **ROC-AUC.** How well a model ranks silage above grain: 0.5 is guessing, 1.0 is perfect.

## Wording that must not change

- Do not write "validated", "accurate" or "matches USDA" for any county or state number. The
  first county method **failed**; the Maryland 2025 figure **reads low**.
- The Michigan result is "passed its test on two counties", not "works for counties".
- Say **42 of 44**, not a percentage, and keep "always grain gets 32" next to it.
- The model was trained on European fields only. The 44 labelled fields were used for testing,
  never for training.

## To redraw

```
python docs/pitch/draw.py              # all seven images
python docs/pitch/draw.py counties     # one of: module signal new_country labelled_fields date_dropping counties state
```

Needs matplotlib. To fill a "to come" mark, put the value and its source in `pitch_numbers.py`
and run the script again.

## Still to come

| Result | Where it will appear |
|---|---|
| Pennsylvania and Wisconsin 2025 | `state_2025.png`, one more line in `script.md`, `limits.md` |
| A model that also reads the pixels around the field | A line in `limits.md`, if it is logged in time |
