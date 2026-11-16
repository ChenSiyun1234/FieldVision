# watch: follow a county through the season in 5-day steps

Added Sunday 15 Nov, about 3 AM. Model estimates, not NASS statistics.

Sentinel-2 passes every 5 days. At each step the code uses only images taken up to that date and reports
a range: at least (silage found so far) and at most (that plus all corn not judged yet).

    python -m fieldvision_agents.watch.watch 24021 2022 --data <counties dir> --out results/watch --reference "Census 2022" 19.6
    python -m fieldvision_agents.watch.watch 24021 2026 --data <counties dir> --poll 2026-11-01

Rules fixed in advance: re-classify when 5% or more of fields got a new clear look; call the agent again
when the "at least" figure moved by a point or more.

## Replay of 2022 (the time series on disk, cut off at each step; nothing fetched)

| | Frederick MD (2,233 fields) | Wyoming NY (820 fields) |
|---|---|---|
| Steps | 25 | 21 |
| Steps re-classified / skipped (no new clear imagery) | 23 / 2 | 14 / 7 |
| Steps where the agent would run again | 7 | 4 |
| Compute for all classifications | 132.6 s | 17.9 s |
| Silage found reaches its final level | 10 Oct (15.0%, final 15.3%) | 4 Nov (37.8%, final 38.6%) |
| Range within 20 points from | 30 Oct | never |
| Range within 10 points from | 4 Nov | never |
| Final range | 15.3% to 22.6% | 38.6% to 66.6% |
| Census 2022 | 19.6% | 66.7% |

- In Frederick the silage count is settled by 10 Oct and the range is within 10 points by 4 Nov, about two
  months before the January summary. The Census figure sits inside the final range.
- In Wyoming cloud keeps the range wide to the end: a 25-day gap in September, then no clear look from
  5 to 25 Oct. A cut is only counted once a later clear look confirms it.
- Live check on 15 Nov 2026: the public catalog lists 36 Sentinel-2 scenes over Frederick since 1 Nov, on
  5 days (latest 13 Nov). Fetching and extracting new scenes uses the existing extractor; that step was
  not run here.

Chart: replay.png in the output folder. One band, one axis, the Census line for comparison only.
