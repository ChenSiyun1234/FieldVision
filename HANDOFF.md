# FieldVision model module: hand-over

Written on the evening of 14 November for the teammates who continue from here. Read this page first; everything else is linked from it.

## In one minute

- **What it does.** Takes corn field outlines and Sentinel-2 time series and says, field by field, silage or grain; then adds the fields up to county and state acres.
- **The model.** Logistic regression on six features, trained on 8,000 labelled fields in Austria, France and the Netherlands. No US labels were used for training.
- **Field level works.** On the organisers' 44 labelled Beltsville field-years, scored once: 42 right (11 of 12 silage found, 1 false silage).
- **County and state totals are not solved.** Two Michigan counties we had never looked at came out right (2.1% for 1.8%, 30.0% for 24.2%). Maryland 2025, the kind of figure the organisers score, reads low (4.75% for 8.9%). Of two Wisconsin counties, the grain county came out right (Racine 6.4%, official 2.4 to 6.4%) and the heaviest dairy county was missed badly (Marathon 4.3% for 49.9%).
- **A second satellite fills cloud gaps.** Adding Landsat 8 and 9 to the Sentinel-2 pictures brought the corn area seen in Marathon from 29% to 83% and in Kent (Maryland) from 9% to 94%, and both readings moved towards USDA's: Marathon 4.3% to 13.8% (official 49.9%), Kent 12.9% to 7.3% (official 5.5%). Run it with `Toolbox(landsat=True)`; the extractor is `fieldvision.ls`.
- **In dairy states the satellite reads low, and this is now measured.** Two states never looked at, each run once on a random sample with both satellites and the rule written first: Wisconsin 2025 16.6% for USDA's 21.6% (5.0 points low, right on our pass mark, and flagged unreliable by our own warning check); Pennsylvania 2025 15.2% for 32.8% (17.6 low, a clear miss, with 98% of the corn seen). But not everywhere: Lancaster 2023, Pennsylvania's largest dairy county, also never looked at, read 26.6% for the Census 27.4%. We do not yet know what separates the two; the next step is American labelled fields, or training against USDA's county totals.
- **For state totals, do not shift the probabilities by a prior.** Starting from USDA's last figure and updating it field by field helped on the eight counties we tuned on (2.6 points off against 2.7 for the last figure alone) but made all three state checks worse (Maryland 9.5 to 6.8 for 8.9, Wisconsin 19.8 to 12.9 for 21.6, Pennsylvania 31.4 to 18.7 for 32.8). USDA's last figure alone is within 2 points in all three. What the model adds today is the call for each field with its evidence, and totals where silage is rare.
- **Last update 02:45 on 15 November.** Michigan and New York 2025 are being extracted as two more clean tests of the same module. The open leads are listed below.

## Where everything is

| What | Where |
|---|---|
| Code | this repository, branch `main` |
| Slides | [`handoff/presentation/`](handoff/presentation/) (PowerPoint and PDF, told for the judges: eleven story slides, then detail slides to pick from; speaker notes on each, and a one-minute spoken version in the notes of slide 1. Slide 5 is the Beltsville test with precision, recall and false positive rate, slide 6 every county and state against USDA, slide 7 what the second satellite (Landsat) did in the two cloudy counties, slide 8 the update of USDA's last figure, slide 9 the lessons, slide 10 where the model sits in the team's system. Every slide stands on its own, so lift the ones that fit your story); the pieces and scripts are in `docs/pitch/` |
| Colab notebooks | [`colab/`](colab/); to look at pictures of any field use `colab/view_field_images.ipynb`; the extraction notebooks as they were run are in `colab/extraction/` (the two for the sequence and pixel experiment in `colab/bundles/`). Open one with `https://colab.research.google.com/github/ChenSiyun1234/FieldVision/blob/main/colab/<path>.ipynb` |
| Data (field outlines and time series per county) | `handoff/FieldVision_handoff_data.zip`; unzip it and the counties are in `3_data/counties/<fips>/<year>/` |
| The two Wisconsin counties of 2023 (Racine, and the third of Marathon that was drawn) | `handoff/FieldVision_counties_wisconsin_2023.zip`; unzip at the repository root and they land in `data/counties/55101/2023/` and `data/counties/55073/2023/`. Marathon is the county the model missed, so it is the one to look at |
| Training data (labelled fields in Austria, France, the Netherlands, with their time series) | `handoff/FieldVision_training_data_europe.zip`; unzip at the repository root and the files land in `data/labels/` and `data/timeseries/`, the paths the training command expects |
| Figures and field pictures | [`handoff/figures/`](handoff/figures/) |
| Every result, with when it was logged | [`docs/RESULTS.md`](docs/RESULTS.md); the full record is [`handoff/results/ACCESS_LOG.md`](handoff/results/ACCESS_LOG.md) |
| How the module works | [`docs/module.md`](docs/module.md), [`docs/model.md`](docs/model.md), [`docs/features.md`](docs/features.md) |
| How to call it from an agent | [`docs/toolbox.md`](docs/toolbox.md) |

## Try it in two minutes

In the team workspace the copy at `/workspace/FieldVision/` already runs Beltsville (`24033`) and Frederick (`24021`):

```bash
python3 /workspace/FieldVision/scripts/toolbox_demo.py 24033 2022
```

For the other counties, from a fresh clone of this repository:

```bash
unzip handoff/FieldVision_handoff_data.zip -d handoff_data && mkdir -p data && cp -r handoff_data/3_data/counties data/
pip install -e . && python scripts/toolbox_demo.py 26057 2022
```
`24033` is Beltsville (the labelled fields), `24021` Frederick, `24035` Queen Anne's, `24029` Kent, `24013` Carroll, `26023` Branch and `26057` Gratiot in Michigan; all 2022. The Maryland 2025 sample is the folder `24_2025_sample`.

From Python:

```python
from fieldvision.toolbox import Toolbox
tb = Toolbox()                               # reads data/counties, registry, results
fields = tb.classify("24021", 2022, "module_v2")   # one row per field: label, probability, evidence
county = tb.aggregate("24021", 2022)               # acres, shares, what could not be seen
```

Policies: `standard` is the one the agent already uses (unchanged); `module_v2` is the final module (evidence tiers, the Europe-trained logistic, the county step); `in_site` is for Beltsville only.

## Look at the fields yourself

`colab/view_field_images.ipynb` draws, for any field, six true-colour Sentinel-2 pictures across the season with the outline, and the greenness curve below. Give it a county folder from the workspace and pick fields by id, at random, or by filter (called silage or grain, clearly seen or not, harvest date). Ready-made examples are in `4_figures/strips/`.

## The numbers

| Test | Compared with | Our figure | Seen before? |
|---|---|---|---|
| Beltsville, 44 labelled field-years | organisers' labels | 42 of 44 (sequence model also 42; boosted trees 37; with dates dropped in training 39; always-grain 32) | no, scored once per model |
| A country the model never saw (Austria, France, Netherlands in turn) | registry labels | mean ROC-AUC 0.765 (boosted trees 0.764; sequence model 0.817, three seeds) | held out |
| Branch, Michigan 2022 | Census 1.8% silage | 2.1% | no |
| Gratiot, Michigan 2022 | Census 24.2% | 30.0% | no |
| Racine, Wisconsin 2023 | survey 2.4% (official figures 2.4 to 6.4%) | 6.4% | no |
| Marathon, Wisconsin 2023, a third of the county's corn | survey 49.9% (official figures 32.1 to 49.9%) | 4.3%; half the fields could not be judged | no |
| Lancaster, Pennsylvania 2023, 10,533 fields (64% of the county's corn), two satellites | Census 2022 27.4% (the 2023 survey, 44.5%, counts half the corn acres) | 26.6% | no |
| Wisconsin 2025, random sample of 8,199 fields, two satellites | NASS 21.6% | 16.6%, range 13.8 to 19.5; flagged unreliable by our warning check | no |
| Pennsylvania 2025, random sample of 3,931 fields, two satellites | NASS 32.8% | 15.2%, range 12.9 to 17.5 (9.5% without double-crop corn) | no |
| Maryland 2025, random sample of 3,611 fields | NASS 8.9% (40,000 acres) | 4.75%, range 3.4 to 6.3 (22,700 acres); with double-crop corn scored afterwards 6.7% (33,135 acres) | no |
| Queen Anne's, Kent, Carroll, Frederick 2022 | Census 1.3 / 5.5 / 8.9 / 19.6% | 4.0 / 12.9 / 3.6 / 13.2% by plain counting; 2.0 / 6.2 / 5.1 / 15.1% added up by harvest window | yes, used while tuning |

## What we learned while tuning, in order

1. **Summing probabilities over a county reads far too high** where silage is rare (22% for a county with 1.3%). Fields whose harvest fell between two looks far apart are the cause: hide four to five weeks of pictures from a labelled grain field and it is called silage 28 to 39% of the time. So only fields seen clearly on both sides of the cut are counted (12 days or less between the two looks).
2. **A correction step with a reference county failed its test** (Kent 21% for 5.5%, Carroll 0% for 8.9%) and was dropped.
3. **Clouds decide what can be seen.** On Maryland's Eastern Shore in 2022 more than half of the corn was cut inside one cloudy fortnight in which no field was seen clearly. Adding up by two-week harvest window, and reporting the corn that could not be seen, brought the worst county miss from 7.4 to 3.8 points on the four Maryland counties.
4. **Early is not silage.** Kent's early, still-green fields have the crop history of grain farms (soybeans in 97%, hay in 6%), not of forage systems. They are most likely grain cut early. Pictures cannot separate them from silage, before or after the cut.
5. **Michigan, never looked at, passed**; Maryland 2025 read low.
6. **A northern dairy county was missed.** In Marathon (Wisconsin, 2023) September had about one clear look per field (August had four), and the silage was cut inside that gap. The fields seen clearly on both sides of the cut are the late harvest, which is grain, so counting them reads 4% where the county has 50%. The crop had also yellowed before it was cut, so "silage is cut green" shows less in the north. The cause is still being worked on; do not use the satellite figure alone for a heavy dairy county.
7. **A second satellite brought the hidden fields back.** Landsat had clear pictures inside both cloudy spells (1 October 2023 over Marathon, 13, 21 and 29 September 2022 over Kent). With them the share of corn seen rose to 83% and 94%, and both readings moved towards USDA's. In Marathon the rest of the gap is the classifier: silage in the north is cut after the crop has yellowed, and a model trained in Europe reads that as grain. That needs labelled northern fields; no index we tried (greenness before the cut, bare soil after it) separates Marathon from Racine.
8. **The pictures improve on USDA's last figure.** See the second-to-last point of the summary at the top; the table is in `docs/RESULTS.md`. A land-use prior in place of the last official figure is worse, because it misses dairy counties that grow little hay (Gratiot).
9. **A timing feature that does not move with the season did not help.** Lancaster's own fields read 15.3% inside the Pennsylvania 2025 sample (488 fields) against 26.6% in the 2023 county run, and the 2025 green peak came 16 days earlier. We tried three replacements for the days-from-peak feature, with the rule for choosing written first: all equal in Europe, none closed the year-to-year gap by a third. The module was left as it is.

## Open leads, in the order we would try them

1. **The harvest detector sometimes takes a later drop.** Field `csb-241825014024676` (Frederick, in `4_figures/strips/`) falls in greenness in late August, recovers, and falls again in October; the October fall was taken as the harvest and the field was called grain. If this is common in dairy counties it explains part of the low reading.
2. **Double-crop corn is not scored.** Corn sown after a small-grain forage is its own class in the boundaries. Maryland 2025: 22,700 silage acres seen plus 17,056 acres of this class is 39,756, against 40,000 from NASS. The sample already holds the time series for these fields.
3. **Two models, one per kind of evidence.** The boosted trees trained with dates dropped may hold up better where the harvest is not clearly seen.
4. **Radar for cloudy weeks.** A Sentinel-1 pilot for Kent is built (`colab/`, package `fieldvision.s1`); it would give a harvest date for fields the optical pictures missed.
5. **A land-use prior.** The share of hay and alfalfa in the cropland lines up with the silage share. Checked against the Census with whole states held out it is 5.9 points off per county and 4.2 per state; on the 2025 state figures, compared once, 3.7 points (weighted by acres). It reads Maryland at 10.6% (NASS 8.9%) but misses the irrigated dairy states of the West badly. It tells how much silage to expect in a place, not which field; show it beside the satellite figure, never instead of it. See [`docs/landuse-prior.md`](docs/landuse-prior.md).

## What not to claim

- That county or state totals are validated. Of the clean tests, Michigan, Racine and Lancaster came out right, Wisconsin 2025 sat on the pass mark, Maryland 2025 read low, and Marathon and Pennsylvania 2025 were missed.
- That the harvest-window figures for Maryland 2022 are a test. They were found on those counties.
- That radar or image models are part of the result. Radar is a pilot and was not run (over Marathon it has the same hole as the optical pictures); the pixel models were built and not run to a result. The sequence model ties the logistic at Beltsville (42 of 44) and calls far less false silage where pictures are missing, but it reads the counties no better overall, so the logistic stays the classifier.
- More than four scorings of the Beltsville labels (logistic, boosted trees, boosted trees with dates dropped, sequence model). Each is in the log with its pass mark written first.
