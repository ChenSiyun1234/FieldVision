"""Entry point for county selection, field collection, and the three time series.

    python -m src.extract_csb
    python -m src.run_all --step select
    python -m src.run_all --step collect --region all
    python -m src.run_all --step s2 --region barc
"""

from __future__ import annotations

import argparse
import os

from .config import GDAL_CLOUD_DEFAULTS, TEST_COUNTIES

os.environ.update(GDAL_CLOUD_DEFAULTS)

from .fields import region_id


def _regions(argument: str) -> list[str]:
    if argument == "barc":
        return ["barc"]
    if argument == "all":
        return ["barc", *[region_id(state, fips) for state, fips in TEST_COUNTIES]]
    return [argument]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BARC and transfer-county time series for 2023.")
    parser.add_argument("--step", required=True, choices=["select", "extract", "collect", "s2", "s1", "weather"])
    parser.add_argument("--region", help="barc, all, or a county folder such as pa_42027.")
    parser.add_argument("--test", action="store_true", help="Process 4 fields. County tests use one tile.")
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Load every scene in one pass. Skips parallel chunks, the MGRS filter, and the SCL date pass.",
    )
    args = parser.parse_args()
    if args.step == "select":
        from .select_counties import main as select_main

        select_main()
        return
    if args.step == "extract":
        from .extract_csb import main as extract_main

        extract_main()
        return
    if not args.region:
        raise SystemExit("--region is required for this step. Use barc, all, or a county id such as pa_42027.")
    if args.step == "collect":
        from .collect_fields import main as collect_main

        collect_main(args.region, test=args.test)
        return
    if args.step == "s2":
        from .s2_timeseries import main as step_main
    elif args.step == "s1":
        from .s1_timeseries import main as step_main
    else:
        from .weather import main as step_main
    for region in _regions(args.region):
        if args.step in {"s2", "s1"}:
            step_main(region, test=args.test, optimize=not args.no_optimize)
        else:
            step_main(region, test=args.test)


if __name__ == "__main__":
    main()
