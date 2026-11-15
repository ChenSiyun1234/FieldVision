"""Full agent run with the critic and the recommender: python3 FieldVision/recommender/run_agent_demo.py 24021 2022"""
import json
import os
import sys

sys.path[:0] = ["/workspace", "/workspace/FieldVision/.deps", "/workspace/FieldVision/src",
                "/workspace/FieldVision/rag-pipeline", "/workspace/FieldVision/recommender"]
os.environ.setdefault("QUICKSTATS_CSV", "/workspace/02_QuickStats/corn_grain_silage_acres_2012_2025.csv")

from fieldvision.toolbox import Toolbox  # noqa: E402
from critic_toolbox import CriticToolbox  # noqa: E402
from silage_match import SilageMatchToolbox  # noqa: E402
from agents.react.supervisor import run_county, NemotronClient  # noqa: E402

if __name__ == "__main__":
    fips = sys.argv[1] if len(sys.argv) > 1 else "24021"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2022
    tools = SilageMatchToolbox(CriticToolbox(Toolbox()))
    result = run_county(fips, year, tools, NemotronClient(), log_dir="/workspace/logs")
    print(json.dumps(result, indent=1, default=str))
