"""Copy the extraction notebooks and their READMEs into colab/extraction/ for GitHub.

Only notebooks and READMEs are copied (no data, code zips or fields). Machine paths in
the READMEs become placeholders: <FV_DATA> for the data/work root, <REPO> for the
repository checkout, <HOME> for the user folder. Fails if any machine path is left.
"""
import argparse
import re
import shutil
from pathlib import Path

SKIP = {"_templates", "stream_px", "stream_train"}   # templates hold data; stream_* belong to another job


def placeholders(text, repo):
    text = text.replace(str(repo), "<REPO>").replace(str(repo).replace("\\", "/"), "<REPO>")
    for pattern, new in [(r"[A-Za-z]:\\Users\\[^\\\s`'\"]+", "<HOME>"), (r"[A-Za-z]:/Users/[^/\s`'\"]+", "<HOME>"),
                         (r"[Cc]:\\t\\fv", "<FV_DATA>"), (r"[Cc]:/t/fv", "<FV_DATA>"), (r"/c/t/fv", "<FV_DATA>")]:
        text = re.sub(pattern, lambda m: new, text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True, help="folder with one subfolder per Colab bundle")
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    dest = args.repo / "colab" / "extraction"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    copied = []
    for folder in sorted(p for p in args.src.iterdir() if p.is_dir() and p.name not in SKIP):
        files = [*folder.glob("*.ipynb"), *folder.glob("README.md")]
        if not files:
            continue
        (dest / folder.name).mkdir()
        for path in files:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".md":
                text = placeholders(text, args.repo.resolve())
            (dest / folder.name / path.name).write_text(text, encoding="utf-8")
        copied.append(folder.name)
    guide = args.src / "README.md"
    if guide.exists():
        (dest / "BRING_BACK.md").write_text(placeholders(guide.read_text(encoding="utf-8"), args.repo.resolve()),
                                            encoding="utf-8")
    leftovers = []
    for path in dest.rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            # A drive letter standing alone (not the "s:" of "https:" or of an escaped "\\n").
            for bad in (r"(?<![A-Za-z])[A-Za-z]:\\\\?(?![nrt\"\\])", r"(?<![A-Za-z])[A-Za-z]:/(?!/)", r"/c/t/",
                        r"\\Users\\", "20053"):
                if re.search(bad, text):
                    leftovers.append((str(path.relative_to(dest)), bad))
    print("copied", len(copied), "bundles:", copied)
    if leftovers:
        raise SystemExit(f"machine paths left: {leftovers}")


if __name__ == "__main__":
    main()
