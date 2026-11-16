import os, zipfile, fnmatch
from pathlib import Path
W = Path("/workspace")
OUT = W / "hima_data" / "export"
OUT.mkdir(exist_ok=True)
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", "rag_index", ".deps", "cache", "export"}
SKIP_FILES = {".env", "mde_afo_raw.json"}

def add_tree(z, src, arc, skip_dirs=(), only=None):
    n = 0
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and str(rel / d) not in skip_dirs and d not in skip_dirs]
        for f in files:
            if f in SKIP_FILES or f.endswith((".pyc", ".part")):
                continue
            p = Path(root) / f
            if only and not any(fnmatch.fnmatch(str(p.relative_to(src)), pat) for pat in only):
                continue
            z.write(p, str(Path(arc) / p.relative_to(src))); n += 1
    return n

code = OUT / "hima_branch_code.zip"
with zipfile.ZipFile(code, "w", zipfile.ZIP_DEFLATED) as z:
    a = add_tree(z, W / "FieldVision" / "rag-pipeline", "rag-pipeline")
    b = add_tree(z, W / "FieldVision" / "recommender", "recommender", skip_dirs={"out"})
    c = add_tree(z, W / "hima_data", "hima_data",
                 only=["README.md", "DATA_LOG.md", "run_*.log", "hima_data_download.ipynb", "hima_data_download.py", "scratch/*.py"])
print("code zip:", code, round(code.stat().st_size / 1e6, 1), "MB; files:", a, b, c)

data = OUT / "hima_data_outputs.zip"
with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as z:
    d = add_tree(z, W / "FieldVision" / "recommender" / "out", "recommender/out")
    e = add_tree(z, W / "hima_data" / "processed", "hima_data/processed", skip_dirs={"parts"})
    f = add_tree(z, W / "hima_data" / "fieldvision_export", "hima_data/fieldvision_export")
    g = add_tree(z, W / "hima_data" / "results", "hima_data/results")
    h = add_tree(z, W / "hima_data" / "model_test_results", "hima_data/model_test_results")
print("data zip:", data, round(data.stat().st_size / 1e6, 1), "MB; files:", d, e, f, g, h)
with zipfile.ZipFile(code) as z:
    names = z.namelist()
print("top:", sorted({n.split('/')[0] + '/' + (n.split('/')[1] if n.count('/') > 1 else '') for n in names})[:40])
