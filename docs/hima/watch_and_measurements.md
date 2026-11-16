# Watch loop and measured numbers

Hima Verma, Hack the Harvest 2026. Model estimates, not NASS statistics.
Code: `src/fieldvision_agents/watch/`. Outputs and charts: `handoff/results/hima/watch/`.

## Following a county through the season
Each 5-day step uses only images taken up to that date. "At least" is silage found so far as a share of
all corn acres; "at most" adds all corn not judged yet. Rules fixed first: re-classify when 5% or more of
fields got a new clear look; the agent runs again only when "at least" moved a point or more.

| | Frederick MD (2,233 fields) | Wyoming NY (820 fields) |
|---|---|---|
| Steps | 25 | 21 |
| Re-classified / skipped | 23 / 2 | 14 / 7 |
| Agent would run again | 7 | 4 |
| Compute, all classifications | 132.6 s | 17.9 s |
| Silage found reaches its final level | 10 Oct (15.0%, final 15.3%) | 4 Nov (37.8%, final 38.6%) |
| Range within 10 points from | 4 Nov | not reached |
| Final range | 15.3% to 22.6% | 38.6% to 66.6% |
| Census 2022 | 19.6% | 66.7% |

Frederick's silage count is settled by 10 October and the range is within 10 points by 4 November,
about two months before the January report, with the Census figure inside the final range.
`poll` lists new Sentinel-2 scenes over a county from the public catalog: on 15 Nov 2026 it found 36
scenes over Frederick since 1 Nov, on 5 days.

## Agent runs with inspect_gaps
| Run | County | What happened | Tokens | Time |
|---|---|---|---|---|
| 36121-2022-3ea37a | Wyoming NY | critic failed 50.7% against the county's history; agent called inspect_gaps and abstained, naming the gap (10 Sep to 5 Oct), 28.0% not judged, and 66.6% if all unjudged fields were silage (Census 66.7%) | 72,800 | 64.5 s |
| 36121-2022-9345f2 | Wyoming NY | first run with inspect_gaps and 6 vision fields; reason names the gap and the range 64.5% to 65.1% | 70,572 | 177 s |
| 24021-2022-f63f7b | Frederick MD | no cloud gap flagged, 7.3% not judged | 102,246 | 100.4 s |

## Speed
- Classification on CPU: about 1.3 s for 820 fields, 5.8 s for 2,233.
- Vision: about 6 s per Super VL call (about 570 reasoning tokens) plus about 14 s to fetch two chips.
- Nemotron runs on NVIDIA's hosted API; replies are cached, so a rerun with the same input costs nothing.
