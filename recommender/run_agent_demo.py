"""Full agent run with the critic and the recommender: python recommender/run_agent_demo.py 24021 2022

Needs `pip install -e ".[agents]"` from the repository root and NVIDIA_API_KEY in .env.
The same run is `fieldvision run 24021 2022` from the app's command line (recommend_sources is on by default).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "rag-pipeline"), str(REPO / "recommender")]

from fieldvision_agents import settings  # noqa: E402,F401  (loads .env, sets QUICKSTATS_CSV paths)
from fieldvision.toolbox import Toolbox  # noqa: E402
from critic_toolbox import CriticToolbox  # noqa: E402
from silage_match import SilageMatchToolbox  # noqa: E402
from fieldvision_agents.react.supervisor import run_county, NemotronClient  # noqa: E402

if __name__ == "__main__":
    fips = sys.argv[1] if len(sys.argv) > 1 else "24021"
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2022
    tools = SilageMatchToolbox(CriticToolbox(Toolbox(data_root=str(settings.COUNTY_DATA))))
    result = run_county(fips, year, tools, NemotronClient())
    print(json.dumps(result, indent=1, default=str))
