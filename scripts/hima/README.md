# scripts/hima

Scoring scripts for the gap path and the watch loop (Hima Verma). The code they call is in
`src/fieldvision_agents/gap/` and `src/fieldvision_agents/watch/`; their outputs are in `handoff/results/hima/`.
Each rule was written down before its first run and scored once.

These ran in the hackathon's Keeboh workspace, so paths point at `/workspace/...`. Edit the paths at the top
of a script to rerun it elsewhere.

| Script | Produces |
|---|---|
| `scoring/gap_states.py` | `gap/states_2025_gap_rule.csv`: gap fields counted by the model's call, 9 states |
| `scoring/gap_states2.py` | `gap/gap_rules_vs_usda.csv`: counting rules and the prior shift against USDA |
| `scoring/gap_states3.py` | `gap/interval_rule_vs_usda.csv`: the timing probability p = pi*S / (pi*S + (1-pi)*G) |
| `scoring/vision_barc.py` | `gap/barc/vision_barc.json`: Super VL scored on the 44 labelled Beltsville fields |
| `scoring/late_green.py` | `gap/late_green_by_place.csv`, `gap/late_green_barc.csv`: green in November |
| `scoring/separator.py` | `gap/separator_by_place.csv`: share of corn not judged against the miss |
| `scoring/example.py` | the worked S, G and p example for Wyoming NY and Frederick MD |
| `scoring/rechart.py` | `watch/*/replay.png` |
| `scoring/show.py`, `audit.py`, `audit2.py`, `vstat.py`, `reasons.py` | small readers for gap reports, agent audit logs, vision answers and reasons |

The county download notebook that produced the Wyoming County, NY data is in `hima_data/`.
