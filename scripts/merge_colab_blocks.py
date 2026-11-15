"""Merge block results computed on Colab back into the laptop run folder.

Input: the parts <run>_colab_blocks.zip.NNN and <run>_SHA256SUMS.txt from the notebook.
Checks every checksum, then that the Colab run used the same code and the same run
signature (same fields bytes, same arguments, same code) as the laptop run. A block
already complete on the laptop is kept; any other block is replaced by Colab's copy
(parquet first, then its checkpoint json, each by atomic rename). Stop the laptop
extraction for this run before merging. Afterwards rerun the run's own command with
the same code to rebuild timeseries.parquet, fields_coverage.csv and the manifest; with
every block finished it reads nothing from the network.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.name
    sums = (args.parts_dir / f"{run}_SHA256SUMS.txt").read_text().split("\n")
    expected = {name: digest for digest, name in (line.split("  ", 1) for line in sums if line.strip())}
    parts = sorted(n for n in expected if n.startswith(f"{run}_colab_blocks.zip."))
    data = b""
    for name in parts:
        payload = (args.parts_dir / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected[name]:
            raise SystemExit(f"checksum mismatch: {name} (download it again)")
        data += payload
    if hashlib.sha256(data).hexdigest() != expected[f"{run}_colab_blocks.zip"]:
        raise SystemExit("checksum of the joined zip does not match")
    print(f"{len(parts)} part(s), {len(data) / 1e6:.1f} MB, checksums OK")

    archive = zipfile.ZipFile(io.BytesIO(data))
    colab = json.loads(archive.read("colab_manifest.json"))
    laptop = json.loads((args.run_dir / "manifest.json").read_text())
    for key in ("source_sha256", "signature"):
        if colab[key] != laptop[key]:
            raise SystemExit(f"{key} differs: Colab {colab[key][:12]} vs laptop {laptop[key][:12]}; not merging")
    print("same code and same run signature as the laptop run:", laptop["signature"][:12])
    print(archive.read("colab_environment.txt").decode().strip().splitlines()[-1])

    blocks = sorted({Path(n).stem for n in archive.namelist() if n.startswith("blocks/")})
    copied, kept = [], []
    for block in blocks:
        local_json = args.run_dir / "blocks" / f"{block}.json"
        if local_json.exists() and json.loads(local_json.read_text()).get("status") == "complete":
            kept.append(block)
            continue
        status = json.loads(archive.read(f"blocks/{block}.json"))
        copied.append((block, status["status"], len(status.get("scene_errors", {}))))
        if args.dry_run:
            continue
        for suffix in (".parquet", ".json"):         # data before checkpoint, as the extractor does
            target = args.run_dir / "blocks" / f"{block}{suffix}"
            temporary = target.with_suffix(suffix + ".colab_tmp")
            temporary.write_bytes(archive.read(f"blocks/{block}{suffix}"))
            os.replace(temporary, target)
    if not args.dry_run and laptop.get("status") == "colab_pending" and not laptop.get("started_at"):
        # Stub folder (no laptop extraction): take the run's start time from Colab so
        # the finalising run and the READY line report a real start. Not in the signature.
        laptop["started_at"] = colab.get("started_at")
        temporary = args.run_dir / "manifest.json.colab_tmp"
        temporary.write_text(json.dumps(laptop, indent=2), encoding="utf-8")
        os.replace(temporary, args.run_dir / "manifest.json")
        print("stub manifest: started_at set from the Colab run:", laptop["started_at"])
    print(f"{'would copy' if args.dry_run else 'copied'} {len(copied)} blocks from Colab; kept {len(kept)} already complete here")
    for block, state, errors in copied:
        if state != "complete":
            print(f"  {block}: {state} with {errors} scene errors (the finalising rerun retries them)")
    states = {}
    for path in (args.run_dir / "blocks").glob("*.json"):
        states[json.loads(path.read_text()).get("status")] = states.get(json.loads(path.read_text()).get("status"), 0) + 1
    print("laptop run now has:", states, "of", laptop["block_count"], "blocks")


if __name__ == "__main__":
    main()
