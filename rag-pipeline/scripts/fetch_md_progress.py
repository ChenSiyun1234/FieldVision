"""
Download Maryland weekly Crop Progress & Condition PDFs (NASS Northeastern Regional Field Office),
August to November 2021-2023, into dataset/maryland/ as MD_crop_progress_YYYY-MM-DD.pdf.

  python rag-pipeline/scripts/fetch_md_progress.py
  python rag-pipeline/scripts/fetch_md_progress.py --years 2022

Links taken from the NASS Maryland Crop Progress and Condition index page on 2026-11-14.
NASS moved these files under /sites/default/files/, and release days are not always Mondays,
so the links are listed exactly instead of guessed.
"""
import argparse
import re
import time
import urllib.request
from pathlib import Path

BASE = "https://www.nass.usda.gov/sites/default/files/Statistics_by_State/Maryland/Publications/Crop_Progress_&_Condition/"
FILES = [
    "2021/MDCW_0802.pdf",
    "2021/MDCW_0809.pdf",
    "2021/MDCW_0816.pdf",
    "2021/MDCW_0822.pdf",
    "2021/MDCW_0830.pdf",
    "2021/MDCW_0907.pdf",
    "2021/MDCW_0913.pdf",
    "2021/MDCW_0920.pdf",
    "2021/MDCW_0927.pdf",
    "2021/MDCW_1004.pdf",
    "2021/MDCW_1012.pdf",
    "2021/MDCW_1018.pdf",
    "2021/MDCW_1025.pdf",
    "2021/MDCW_1101.pdf",
    "2021/MDCW_1108.pdf",
    "2021/MDCW_1115.pdf",
    "2021/MDCW_1122.pdf",
    "2021/MDCW_1129.pdf",
    "2022/MDCW_0801.pdf",
    "2022/MDCW_0808.pdf",
    "2022/MDCW_0815.pdf",
    "2022/MDCW_0822.pdf",
    "2022/MDCW_0829.pdf",
    "2022/MDCW_0906.pdf",
    "2022/MDCW_0912.pdf",
    "2022/MDCW_0919.pdf",
    "2022/MDCW_0926.pdf",
    "2022/MDCW_1003.pdf",
    "2022/MDCW_1009.pdf",
    "2022/MDCW_1017.pdf",
    "2022/MDCW_1024.pdf",
    "2022/MDCW_1030.pdf",
    "2022/MDCW_1106.pdf",
    "2022/MDCW_1114.pdf",
    "2022/MDCW_1121.pdf",
    "2022/MDCW_1127.pdf",
    "2023/MDCW_0807.pdf",
    "2023/MDCW_0814.pdf",
    "2023/MDCW_821.pdf",
    "2023/MDCW_828.pdf",
    "2023/MDCW_0905.pdf",
    "2023/MDCW_0911.pdf",
    "2023/MDCW_0918.pdf",
    "2023/MDCW_0925.pdf",
    "2023/MDCW_1002.pdf",
    "2023/MDCW_1010.pdf",
    "2023/MDCW_1016.pdf",
    "2023/MDCW_1023.pdf",
    "2023/MDCW_1030.pdf",
    "2023/MDCW_1106.pdf",
    "2023/MDCW_1113.pdf",
    "2023/MDCW_1120.pdf",
    "2023/MDCW_1127.pdf",
]


def target_name(rel: str) -> str:
    year, fname = rel.split("/")
    digits = re.search(r"(\d{3,4})", fname).group(1).zfill(4)
    return f"MD_crop_progress_{year}-{digits[:2]}-{digits[2:]}.pdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "dataset" / "maryland"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    counts = {y: 0 for y in args.years}
    for rel in FILES:
        year = int(rel[:4])
        if year not in counts:
            continue
        target = out / target_name(rel)
        if not target.exists():
            try:
                req = urllib.request.Request(BASE + rel, headers={"User-Agent": "Mozilla/5.0 (hack-the-harvest-rag)"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                if data[:4] != b"%PDF":
                    print(f"not a PDF: {rel}")
                    continue
                target.write_bytes(data)
                time.sleep(0.3)
            except Exception as exc:
                print(f"failed {rel}: {exc}")
                continue
        counts[year] += 1
    for y, n in counts.items():
        print(f"{y}: {n} weekly reports in {out}")


if __name__ == "__main__":
    main()
