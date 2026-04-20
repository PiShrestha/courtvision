"""guarded download recipes for a basketball-specific YOLOv8 checkpoint.

this script does NOT automatically download anything. `YOLO(path.pt)`
deserializes pickle, which is an RCE vector if the weights are from an
untrusted source, so we make the user explicitly pick a provider and
acknowledge the security posture before anything touches disk.

usage:
    python experiments/demo/download_basketball_model.py --list
    python experiments/demo/download_basketball_model.py --provider roboflow \\
        --api-key <YOUR_KEY> --target experiments/demo/weights/basketball.pt
    python experiments/demo/download_basketball_model.py --provider huggingface \\
        --repo-id <user/repo> --revision <git_sha> \\
        --target experiments/demo/weights/basketball.pt
    python experiments/demo/download_basketball_model.py --provider local \\
        --source /path/to/your/verified/best.pt \\
        --target experiments/demo/weights/basketball.pt

providers:
    roboflow      uses the roboflow python sdk with *your* api key. the
                  recommended "basketball-players" workspace requires you
                  to sign in and pick a version. we never hard-code
                  credentials.
    huggingface   uses huggingface_hub with a pinned git revision so an
                  upstream force-push can't swap weights from under you.
    local         copy from a path you've already vetted locally. no
                  network. this is the safest path once you have a copy.

after download, verify and register:

    sha256sum experiments/demo/weights/basketball.pt
    python -c "from ultralytics import YOLO; m = YOLO('experiments/demo/weights/basketball.pt'); print(m.names)"

then plug into the v2 runner:

    python experiments/demo/run_demo_v2.py \\
        --video 1v1-mk.mov --start 268 --end 338 \\
        --hoop experiments/demo/hoop_configs/1v1-mk.json \\
        --custom-model experiments/demo/weights/basketball.pt \\
        --out experiments/demo/outputs_v2_live/mk_custom
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


PROVIDERS = ("roboflow", "huggingface", "local")

# expected canonical classes the rest of the demo wires up.
EXPECTED_CLASSES = {"player", "ball", "hoop"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true",
                    help="print the known providers and exit")
    ap.add_argument("--provider", choices=PROVIDERS,
                    help="which source to fetch the weights from")
    ap.add_argument("--target", default="experiments/demo/weights/basketball.pt",
                    help="where to write the downloaded .pt (default: weights/basketball.pt)")
    # roboflow
    ap.add_argument("--api-key", default=None,
                    help="roboflow api key (required for --provider roboflow)")
    ap.add_argument("--workspace", default="roboflow-universe-projects",
                    help="roboflow workspace slug")
    ap.add_argument("--project", default="basketball-players-fy4c2",
                    help="roboflow project slug")
    ap.add_argument("--version", type=int, default=None,
                    help="roboflow dataset version number (required for roboflow)")
    # huggingface
    ap.add_argument("--repo-id", default=None,
                    help="hugging face repo id, e.g. user/basketball-yolov8")
    ap.add_argument("--revision", default=None,
                    help="pinned git revision (sha or tag) on huggingface — "
                         "required so a force-push can't swap weights")
    ap.add_argument("--filename", default="best.pt",
                    help="name of the .pt inside the hf repo")
    # local
    ap.add_argument("--source", default=None,
                    help="local source path (for --provider local)")
    # post-download sanity
    ap.add_argument("--skip-sanity", action="store_true",
                    help="skip the YOLO(path).names check after download")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        _list_providers()
        return 0
    if args.provider is None:
        print("error: --provider is required (or pass --list).", file=sys.stderr)
        return 2

    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print(f"refusing to overwrite existing {target}", file=sys.stderr)
        print("remove it first if you really mean to replace it.", file=sys.stderr)
        return 1

    if args.provider == "roboflow":
        rc = _download_roboflow(args, target)
    elif args.provider == "huggingface":
        rc = _download_huggingface(args, target)
    elif args.provider == "local":
        rc = _copy_local(args, target)
    else:
        rc = 2
    if rc != 0:
        return rc

    # print the sha256 so the user can record it for future integrity checks.
    _print_sha256(target)
    if not args.skip_sanity:
        return _sanity_check(target)
    return 0


# ---- providers ------------------------------------------------------------


def _download_roboflow(args, target: Path) -> int:
    if not args.api_key:
        print("--api-key is required for --provider roboflow.", file=sys.stderr)
        return 2
    if args.version is None:
        print("--version is required for --provider roboflow.", file=sys.stderr)
        return 2
    try:
        from roboflow import Roboflow        # type: ignore
    except ImportError:
        print("roboflow sdk not installed. run: pip install roboflow", file=sys.stderr)
        return 2
    print(f"* fetching {args.workspace}/{args.project} v{args.version}")
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov8")
    # roboflow puts the weights in dataset.location/weights/best.pt after
    # export, depending on the project. user confirms the actual path.
    candidates = [Path(dataset.location) / "weights" / "best.pt",
                  Path(dataset.location) / "best.pt"]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        print("roboflow export did not include a best.pt — list and copy manually:",
              file=sys.stderr)
        print(f"  {dataset.location}", file=sys.stderr)
        return 1
    shutil.copy2(src, target)
    print(f"saved -> {target}")
    return 0


def _download_huggingface(args, target: Path) -> int:
    if not args.repo_id:
        print("--repo-id is required for --provider huggingface.", file=sys.stderr)
        return 2
    if not args.revision:
        print("--revision is required so upstream can't silently swap weights.",
              file=sys.stderr)
        return 2
    try:
        from huggingface_hub import hf_hub_download    # type: ignore
    except ImportError:
        print("huggingface_hub not installed. run: pip install huggingface_hub",
              file=sys.stderr)
        return 2
    print(f"* fetching {args.repo_id}@{args.revision}:{args.filename}")
    local = hf_hub_download(repo_id=args.repo_id, filename=args.filename,
                             revision=args.revision)
    shutil.copy2(local, target)
    print(f"saved -> {target}")
    return 0


def _copy_local(args, target: Path) -> int:
    if not args.source:
        print("--source is required for --provider local.", file=sys.stderr)
        return 2
    src = Path(args.source)
    if not src.exists():
        print(f"source not found: {src}", file=sys.stderr)
        return 1
    shutil.copy2(src, target)
    print(f"copied {src} -> {target}")
    return 0


# ---- helpers --------------------------------------------------------------


def _list_providers() -> None:
    print("available providers:")
    print("  roboflow     — roboflow universe via the roboflow sdk")
    print("                 requires: --api-key, --workspace, --project, --version")
    print("                 default workspace/project: roboflow-universe-projects/"
          "basketball-players-fy4c2")
    print("  huggingface  — huggingface hub with pinned revision")
    print("                 requires: --repo-id, --revision (sha or tag), --filename")
    print("  local        — copy from a path you've already verified")
    print("                 requires: --source /path/to/best.pt")
    print()
    print("canonical class set this pipeline expects (via alias table in")
    print("custom_tracker.DEFAULT_ALIASES):")
    print("  player  (aliases: person, player, players)")
    print("  ball    (aliases: ball, basketball, sports-ball)")
    print("  hoop    (aliases: hoop, rim, basketball-hoop, net)")


def _print_sha256(path: Path) -> None:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    print(f"sha256({path.name}) = {h.hexdigest()}")
    print("record this value if you want to pin to this exact weights blob.")


def _sanity_check(path: Path) -> int:
    try:
        from ultralytics import YOLO          # type: ignore
    except ImportError:
        print("ultralytics not installed — skipping sanity check", file=sys.stderr)
        return 0
    print(f"* loading {path} for class-name sanity check …")
    try:
        model = YOLO(str(path))
    except Exception as exc:
        print(f"YOLO failed to load the weights: {exc}", file=sys.stderr)
        return 1
    names = getattr(model, "names", None) or {}
    print(f"  classes: {dict(names)}")
    found = {str(v).lower() for v in names.values()}
    missing = EXPECTED_CLASSES - {n for n in found
                                    if any(n == e or e in n for e in EXPECTED_CLASSES)}
    if "hoop" not in found and "rim" not in found and not any("hoop" in n for n in found):
        print("warning: no obvious hoop/rim class in this checkpoint.",
              file=sys.stderr)
    if missing:
        print(f"note: expected classes not clearly present: {missing}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
