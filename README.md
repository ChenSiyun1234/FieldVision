# FieldVision

Hack the Harvest 2026, USDA NASS challenge: tell silage corn from grain corn field by field, using only public satellite data.

![Fields called grain or silage from Sentinel-2 pictures and greenness curves](handoff/figures/deck/strips_sheet.png)

The team won the Grand Prize. I built the classification model. It reads each corn field's Sentinel-2 time series (and Landsat 8 and 9 where clouds leave gaps) and calls the field silage, grain, or "not seen clearly". The calls add up to county and state silage acreage.

- **Training.** Only 44 labelled US field-years exist, so I trained on 8,000 labelled fields from Austria, France and the Netherlands and kept the US labels for testing.
- **Model.** Logistic regression on six crop-timing features: greenness and moisture before the cut, how fast the field dries, days from peak to cut, and what the stubble looks like afterwards. I compared boosted trees and sequence models and chose the simplest one by holding out a whole country.
- **Evidence.** A field counts toward a total only if two clear looks around the cut are at most 12 days apart.
- **Field level.** 42 of the 44 labelled USDA fields right, scored once; 11 of 12 silage fields found, one false alarm.
- **Totals.** Four of five places I had never opened came within 5 points of USDA's figure, for example Michigan 2025 at 14.4% against 14.7%. Pennsylvania did not: 15.2% against 32.8%.
- **Second satellite.** Landsat took the corn area seen clearly in Kent County, Maryland from 9% to 94%.
- **Testing.** I wrote each pass mark before opening a place's data and ran each test once.

The model is a Python package, `fieldvision.toolbox.Toolbox`, that the team's agent calls. Start with [HANDOFF.md](HANDOFF.md); the slides are in `handoff/presentation/`.

```
src/        data access, features, models, evaluation
app/        map demo
docs/       results, model notes, pitch
colab/      extraction notebooks
```
