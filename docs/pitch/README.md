# Pitch pack: the model module

Everything needed to put the model part into the 5-minute talk. Each figure is a 1920 x 1080 PNG
with its own title, so it can go on a slide as it is.

**Status.** Every planned test is in: Michigan, Maryland 2025, the last test on the labelled
fields, Wisconsin 2023, Wisconsin 2025, Pennsylvania 2025, Lancaster County 2023, Michigan 2025
and New York 2025.

**Newer than the rest of this page:** the Wisconsin 2023 test failed (Marathon County read 4.3 %
where USDA has 32.1 to 49.9 %). Five places never looked at before were then run once each.
Wisconsin 2025 read 16.6 % where USDA has 21.6 %: 5.0 points low, right on the pass mark of 5
(16.62 % and 4.98 points; the 16.5 % logged first left out double-crop corn by mistake and was
corrected). We had written that it would read below 16.6 %, which misses by the same hair. A
warning check on two grain counties went off there (15.0 % for 4.5 %), which by the rule written
first marks the figure as unreliable, and the deck says so. Pennsylvania 2025 read 15.2 % where
USDA has 32.8 %: 17.6 points low, a clear miss, with 98 % of the corn seen and no warning, so
clouds are not the reason. Lancaster County 2023, Pennsylvania's largest dairy county, read 26.6 %
where USDA's 2022 Census has 27.4 %: 0.8 points low, a pass (the 2023 survey says 44.5 % but
counts half the county's corn acres; the Census was fixed as the reference before the run).
Michigan 2025, a state sample, read 14.4 % where USDA has 14.7 %: 0.3 points low, a pass, with
98 % of the corn seen and no warning. We had predicted in writing that it would read low; that
was wrong, and by the reading written with the prediction the low readings go with the place, not
the year. New York 2025, where more than half the corn is silage, read 48.7 % where USDA has
53.2 %: 4.5 points low, a pass, with 88 % of the corn seen and no warning; we had predicted a
fail, and were wrong again. So four of the five clean tests are within 5 points of USDA
(Michigan, New York, Lancaster, and Wisconsin on its mark and flagged); Pennsylvania as a whole
reads less than half, and we do not yet know why. The slide deck is up to date; the figures, `script.md` and `limits.md` of the
first pack were written before all of these.

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

## The slide deck

The deck itself (a ten-slide story for the judges, then ten backup slides, 16:9, speaker notes
in English and Chinese on every slide) is built from the same numbers. Earlier versions (16
slides, ten, twelve) are in this folder's history. The PowerPoint and PDF files are not in git;
they are made by:

```
python docs/pitch/deck_figs.py --pictures <folder with strip_*.png> --maps <folder with map_*.png>
python docs/pitch/build_deck.py --out <folder>/FieldVision_model_story.pptx
pwsh docs/pitch/export_deck.ps1 -Pptx <folder>/FieldVision_model_story.pptx
```

`deck_figs.py` draws the figure bodies into `docs/pitch/deck/` (the slide carries the title, so
these have none) and cuts the field pictures and county maps out of the sheets made by the
extraction work. `build_deck.py` needs python-pptx; titles, cards and notes are editable text.
`export_deck.ps1` asks PowerPoint for the PDF. To add a place to the results slides, add a row to
`PLACES` in `pitch_numbers.py`.

**Who the story is for.** Ten judges: five from USDA (the question is theirs), two from NVIDIA, two
from Niantic Spatial, one from UC Davis. Five minutes, no questions. The rubric is USDA's and its
words are on the slides: presentation 40 % ("demonstrating the solution and its value", "explain
their process and how it could be developed further"); technical quality 20 % (precision, recall,
false positive rate, response time); innovation 20 % ("a new data source or harvest signal");
feasibility 20 % ("potential to be developed further").

**The story, ten slides.** 1 USDA's own question, with a real field. 2 The catch and the idea: no
US labels, so learn from European farm registries (the new data source). 3 The signal, in real
pictures: silage is cut green and wet (the harvest signal). 4 How it works in three plain steps,
and the rule that a hidden harvest is not guessed. 5 The test on USDA's 44 labelled fields, with
the rubric's four measures. 6 The scoreboard: fourteen places against USDA, grouped by
state: within 5 points in Michigan (the whole state in 2025, 14.4 % for USDA's 14.7 %), New York
(48.7 % for 53.2 %) and Wisconsin (16.6 % for 21.6 %, on its pass mark and flagged); right in
Lancaster County (26.6 % for 27.4 %); Pennsylvania as a whole missed (15.2 % for 32.8 %);
Maryland, where the method was tuned, 4 low as a state. 7 A second satellite, Landsat, in the three counties where clouds hid
the harvest: all three figures move towards USDA's; in a fourth county that was already well seen
the figure changes little and slips one point past USDA's range (the second new data source).
8 What went wrong on the way and what it taught us (the process); the last lesson is the measured
limit: four of five clean tests within 5 points; Pennsylvania as a whole reads less than half. 9 Where the model sits in the
team's system. 10 What USDA gets today (field calls, county totals where silage is rare, a plain
warning elsewhere), what comes next (American labelled fields), and the closing line. After a divider
come the slides that go deeper; every slide stands on its own, so any of them can be lifted into
another deck. One of them was a story slide until Wisconsin 2025 came in: starting from USDA's
last figure and updating it with the pictures moved six of the eight counties seen while tuning
closer to the truth, but made four of the five state figures worse. The notes of slide 1 end with a
one-minute spoken version of the whole story.

## Still to come

| Result | Where it will appear |
|---|---|
| Why Pennsylvania as a whole reads low when the other clean tests are within 5 points | The scoreboard and lesson 4 of the deck, `limits.md` |
| A model that also reads the pixels around the field | A line in `limits.md`, if it is logged in time |
