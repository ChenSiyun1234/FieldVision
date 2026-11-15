# Known gaps: what the data says

Model estimates, not NASS statistics. Built from results on disk; nothing re-trained or tuned.

## 1. Thin ground truth

42 of 44 is 84.9–98.7% right at 95%; always-grain is 58.2–83.7%. The ranges do not overlap, so the model beats the baseline, but 44 field-years from one farm cannot say how it does elsewhere. No other US field labels are public; county shares are checked against the Census instead.

## 2. Corn Belt

In the Corn Belt the model lands within 3.8 points of NASS 2025 (WI +3.8, IA +2.9, NE -1.3); USDA's figure is inside our sampling range in WI, NE. Where silage is rare a few points is still a large relative miss, so low-silage states need a higher bar before a field is called silage. PA is 24.2 points low, so the big miss is in the East, not the Corn Belt. Each state rests on 228–233 clearly seen fields. Further north it breaks: Marathon, Wisconsin 2023 reads 13.8% with Landsat against USDA's 49.9%. Northern silage is cut after the crop yellows, so a model trained in Europe reads it as grain; this needs labelled northern fields.

## 3. Year gap

Pennsylvania's survey share jumped to 33.7% in 2023 from about 30.2% in 2022, so if Lancaster kept its 2022 ratio to the state its 2023 share would be about 30.5%. That explains 3.1 of the 7.8 points between us and the 2022 Census; 4.7 points are left, at the edge of our sampling range (30.7–39.8%). The Census trend alone (-1.4 points a year) would point the other way, so the year gap is real but does not close the difference.

## 4. Unjudged fields

In Lancaster the share could be anywhere from 22.6% to 58.3% depending on the unjudged fields. Filling them with the model's own lean gives 40.1%. Most unjudged fields were cut inside a long gap between clear images (wide_harvest_interval); Landsat 8/9 is the fix the team took: a second satellite whose passes fill the gaps. Kent: corn seen 9% to 94%, silage 12.9% to 7.3% (USDA 5.5). Queen Anne's: corn seen 14% to 70%, silage 4.0% to 2.4% (USDA 1.3). Marathon: corn seen 29% to 83%, silage 4.3% to 13.8% (USDA 49.9). The app uses Landsat wherever a county has it on disk; Frederick and Lancaster do not yet. These are checks on counties seen while tuning.

## 5. Price story

Our four-state silage error nets out to 301,000 acres, about 0.31 points of US corn, which the price models turn into 0.3–2.9% on price. State errors run from -24.2 to +3.8 points, so the sign is not settled. On clean leave-one-year-out tests no model beats last year's price by much (15.9% vs 14.4% average error). So the price page stays a labelled what-if; it is not a forecast and our silage map should not feed one yet.

## 6. Blind test

4 of 6 blind comparisons land within 5 points of USDA. The misses: Frederick County, MD -6.4, PA -24.2. Frederick's critic range (24.3–32.1%, from the 2012 and 2017 Census) would only have warned on the true 19.6% as well, so the history check cannot see a real change coming. Starting from USDA's last figure and letting the fields update it, 6 of 8 counties move closer to the truth, mean miss 2.6 points (the satellite alone 9.7); these counties were seen while tuning, so it is not a clean test. Clean tests, written down before opening: Wisconsin 2025, Pennsylvania 2025, Lancaster, Pa. 2023, Michigan 2025, New York 2025.

## 7. Scale

Satellite pulls are the bottleneck; the batch script lines up the top silage counties for the pod.

| Place | Season | Ours % | USDA % | Error | USDA source | Hidden from pipeline |
|---|---|---|---|---|---|---|
| Frederick County, MD | 2022 | 13.2 | 19.6 | -6.4 | 2022 Census | yes: the critic only reads Census years before 2022 (2012, 2017) |
| Maryland | 2025 | 4.7 | 8.9 | -4.2 | NASS 2025 survey | yes: the state run used 2022 Census reference blocks |
| PA | 2025 | 8.6 | 32.8 | -24.2 | NASS 2025 survey | yes: priors come from earlier years |
| WI | 2025 | 25.4 | 21.6 | +3.8 | NASS 2025 survey | yes: priors come from earlier years |
| IA | 2025 | 4.8 | 1.9 | +2.9 | NASS 2025 survey | yes: priors come from earlier years |
| NE | 2025 | 0.8 | 2.1 | -1.3 | NASS 2025 survey | yes: priors come from earlier years |
| Lancaster County, PA | 2023 | 35.2 | 27.4 | +7.8 | 2022 Census (no 2023 county figure) | no: nearest year, and the critic saw it |
