"""
Download NASS weekly Crop Progress reports (national, 18 states), Aug to Nov 2021-2023,
as plain text, renamed for the RAG: US_crop_progress_2023_w38.txt

  python rag-pipeline/scripts/fetch_crop_progress.py

Links collected from esmis.nal.usda.gov (publication 8336h188j) on 2026-11-14.
Maryland is NOT one of the 18 states in these reports; they give national and
other-state harvest pace vs the 5-year average.
"""
import argparse
import re
import time
import urllib.request
from pathlib import Path

URLS = [
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/k0699547x/8g84nk258/prog3121.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/s4656d25r/cc08jc79f/prog3221.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/m900pr962/zw1303988/prog3321.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/qj72q579n/2801qf533/prog3421.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/br86c237g/zk51wg005/prog3521.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/c534gm50x/79408w400/prog3621.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/g445dc740/5712n522c/prog3721.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/h702r5844/08613m26p/prog3821.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/vx022d38r/gt54mm18g/prog3921.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/tq57pq71h/9306tx73k/prog4021.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/vx022d71s/v405t856g/prog4121.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/m326n1833/w9506126d/prog4221.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/z316r1819/7w62g8506/prog4321.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/bg258c944/sn00bz467/prog4421.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/x920gx81z/4b29c731m/prog4521.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/79408x98m/jh344t003/prog4621.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/jh344t178/p8419n89p/prog4721.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/bg258d49h/1n79j475z/prog4821.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/t435hk252/sf269c89z/prog3122.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/cn69n9494/0c484r510/prog3222.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/pg15cn04c/d791tp10g/prog3322.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/5t34ts423/p8419w08x/prog3422.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/vt151s09q/h989s958f/prog3522.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/00001739t/wh2480312/prog3622.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/st74dz44n/pv63h7438/prog3722.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/w6635b211/hm50w0763/prog3822.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/fq979382f/rb68zm72z/prog3922.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/df65wh90c/m039md649/prog4022.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/z890t358m/z890t359w/prog4122.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/zs25zj733/2b88rp203/prog4222.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/9s162f77g/zs25zj81s/prog4322.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/cv43q526q/pc28bs97b/prog4422.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/q524kz25d/fb495j67b/prog4522.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/hq37wz05w/1r66kb46m/prog4622.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/rn302b37f/dz012026z/prog4722.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/pr76gd79t/5h73r561w/prog4822.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/f7624v982/gb19gp070/prog3123.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/gb19gp25x/69700h69k/prog3223.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/0r968m29b/9s162p280/prog3323.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/hx11zz54j/7h14c6818/prog3423.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/rf560r376/n009xj86f/prog3523.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/1257cb32w/s7527v216/prog3623.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/qz20v964k/c534h539d/prog3723.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/h702rq94p/1g05gw82z/prog3823.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/k930df51r/5h73rd26r/prog3923.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/3j334m76p/mk61t2712/prog4023.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/np194v04b/zg64w510v/prog4123.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/xd07jc25m/73667p05m/prog4223.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/7w62gv00h/m326nm00v/prog4323.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/p841b7017/4t64j711p/prog4423.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/0v839k77n/sq87dc48j/prog4523.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/7p88f363p/x633gm09p/prog4623.txt",
    "https://esmis.nal.usda.gov/sites/default/release-files/8336h188j/1z40nc415/6108wx78b/prog4723.txt",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "dataset" / "national"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = 0
    for url in URLS:
        week, yy = re.search(r"prog(\d\d)(\d\d)\.txt$", url).groups()
        target = out / f"US_crop_progress_20{yy}_w{week}.txt"
        if target.exists():
            ok += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "hack-the-harvest-rag/1.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                target.write_bytes(r.read())
            ok += 1
        except Exception as exc:
            print(f"failed {url.split('/')[-1]}: {exc}")
        time.sleep(0.3)
    print(f"{ok}/{len(URLS)} reports in {out}")


if __name__ == "__main__":
    main()
