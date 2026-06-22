"""Emit a CWL input job for fed_eval.cwl from a prepared shard directory and a feature mask.

Usage: python workflow/make_job.py workflow/shards/<dataset> mask.npy > job.yml
Then:  cwltool workflow/cwl/fed_eval.cwl job.yml
"""
import json
import sys
from pathlib import Path


def main() -> int:
    shard_dir = Path(sys.argv[1])
    mask = sys.argv[2] if len(sys.argv) > 2 else "mask.npy"
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    lines = ["mask:", f"  class: File", f"  path: {Path(mask).resolve()}",
             "k: 5", "s: 1.0", "shards:"]
    ids = []
    for c in manifest["clients"]:
        lines.append("  - class: File")
        lines.append(f"    path: {(shard_dir / c['file']).resolve()}")
        ids.append(c["client_id"])
    lines.append("client_ids: [" + ", ".join(str(i) for i in ids) + "]")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
