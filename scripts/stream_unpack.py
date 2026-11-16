"""Verify and safely unpack multipart stream_px chip results with bounded I/O memory."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
from tempfile import TemporaryDirectory
import zipfile

CHUNK = 1024 * 1024
RUN_NAMES = {"at_2022", "at_2023", "at_2024", "fr_2022", "nl_2025",
             "barc_2021", "barc_2022", "barc_2023"}
MEMBERS = {"chips.npz", "timeseries.parquet", "manifest.json", "colab_environment.txt"}


def read_checksums(path):
    entries = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)\n?", line)
            if not match:
                raise ValueError(f"Malformed checksum line in {path.name}")
            entries.append((match[2], match[1]))
    if len({name for name, _ in entries}) != len(entries):
        raise ValueError(f"Duplicate checksum names in {path.name}")
    return entries


def unpack_one(checksums, output):
    name = checksums.name.removeprefix("stream_px_").removesuffix("_SHA256SUMS.txt")
    if name not in RUN_NAMES:
        raise ValueError(f"Unknown result run {name!r}")
    archive_name = f"stream_px_{name}_colab_chips.zip"
    entries = read_checksums(checksums)
    if len(entries) < 2 or entries[-1][0] != archive_name:
        raise ValueError(f"{checksums.name}: expected parts followed by the full ZIP checksum")
    parts = entries[:-1]
    for index, (part, _) in enumerate(parts):
        if part != f"{archive_name}.{index:03d}":
            raise ValueError(f"{checksums.name}: part numbers must start at 000 and be contiguous")
    actual_parts = {p.name for p in checksums.parent.glob(archive_name + ".*")}
    if actual_parts != {part for part, _ in parts}:
        raise ValueError(f"{name}: missing or unexpected multipart files")
    target = output
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError(f"Unsafe output run directory: {target}")
    with TemporaryDirectory(prefix=f".unpack-{name}-", dir=output) as temporary:
        stage = Path(temporary)
        archive_path = stage / "verified.zip"
        whole = hashlib.sha256()
        with archive_path.open("wb") as destination:
            for part_name, expected in parts:
                part = checksums.parent / part_name
                if part.is_symlink() or not part.is_file() or not 0 < part.stat().st_size <= 50 * 1024 * 1024:
                    raise ValueError(f"Invalid part: {part_name}")
                digest = hashlib.sha256()
                with part.open("rb") as source:
                    for chunk in iter(lambda: source.read(CHUNK), b""):
                        digest.update(chunk)
                        whole.update(chunk)
                        destination.write(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f"SHA256 mismatch: {part_name}")
        if whole.hexdigest() != entries[-1][1]:
            raise ValueError(f"SHA256 mismatch: {archive_name}")
        extracted = stage / "result"
        extracted.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) != len(MEMBERS) or {m.filename for m in members} != MEMBERS:
                raise ValueError(f"{archive_name}: unexpected, missing, duplicate, or unsafe members")
            for member in members:
                if member.is_dir() or member.flag_bits & 1 or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"Unsafe ZIP member: {member.filename}")
            corrupt = archive.testzip()
            if corrupt:
                raise ValueError(f"ZIP CRC failed: {corrupt}")
            for member in members:
                with archive.open(member) as source, (extracted / member.filename).open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=CHUNK)
        manifest = json.loads((extracted / "manifest.json").read_text(encoding="utf-8"))
        counts = {key: manifest.get(key) for key in ("status", "input_fields", "fields_with_chips",
                  "n_obs", "median_chips_per_field", "scene_failures", "block_count")}
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise ValueError(f"Unsafe output run directory: {target}")
        if target.exists():
            for member in MEMBERS:
                (extracted / member).replace(target / member)
        else:
            extracted.rename(target)
    print(name, json.dumps(counts, sort_keys=True))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        files = sorted(args.parts_dir.glob("stream_px_*_SHA256SUMS.txt"))
        if len(files) != 1:
            raise ValueError("Parts directory must contain exactly one stream_px checksum file")
        args.out.mkdir(parents=True, exist_ok=True)
        output = args.out.resolve()
        for path in files:
            if path.is_symlink() or not path.is_file():
                raise ValueError("Checksum files must be regular files")
            unpack_one(path, output)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
