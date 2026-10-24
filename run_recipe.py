"""Run one recorded training recipe on a user-provided prepared dataset."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    plan = json.loads((ROOT / "experiment_plan.json").read_text(encoding="utf-8"))
    candidates = {item["name"]: item for item in plan["candidates"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--candidate", choices=tuple(candidates))
    parser.add_argument("--data", type=Path, default=ROOT / "data/yolo/data.yaml")
    parser.add_argument("--name", help="New run name; recorded artifacts are never overwritten")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.list:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    if not args.candidate:
        parser.error("Choose --candidate; use --list to inspect the recorded recipes")
    cfg = candidates[args.candidate]
    model = cfg["model"]
    if model.startswith("models/") and not (ROOT / model).is_file():
        parser.error("Download the required parent/reference checkpoint from Release into models/")
    if not args.data.is_file():
        parser.error("Prepare the original dataset before training")
    command = [sys.executable, str(ROOT / "train.py"), "--data", str(args.data.resolve()),
               "--model", model, "--name", args.name or ("reproduce_" + args.candidate),
               "--device", args.device, "--workers", str(args.workers)]
    for flag in ("epochs", "imgsz", "batch", "lr0", "patience"):
        command += ["--" + flag, str(cfg[flag])]
    command += ["--seed", str(cfg.get("seed", 42))]
    subprocess.run(command, cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
