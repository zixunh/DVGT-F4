"""Prepare pinned DINOv3 code, official anchors and optional DVGT-2 checkpoint.

Run from an environment with the dvgt2 project extra installed. No model download
occurs during normal model construction. No training dataset is downloaded here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Download official DVGT-2 weights (6.84 GB)",
    )
    parser.add_argument("--verify-checksum", action="store_true")
    args = parser.parse_args()
    lock = json.loads((ROOT / "upstream.lock.json").read_text())
    dino = ROOT / "third_party/dinov3"
    if not dino.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "--filter=blob:none",
                "https://github.com/facebookresearch/dinov3.git",
                str(dino),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(dino), "checkout", "--detach", lock["dinov3_commit"]],
            check=True,
        )
    else:
        revision = subprocess.check_output(
            ["git", "-C", str(dino), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != lock["dinov3_commit"]:
            raise RuntimeError(
                f"DINOv3 revision mismatch: {revision}; expected {lock['dinov3_commit']}. Preserve local changes before switching revisions."
            )
    from huggingface_hub import hf_hub_download

    directory = ROOT / "checkpoints"
    directory.mkdir(exist_ok=True)
    anchors = ROOT / "data_annotation/anchors/train_total_20"
    anchors.mkdir(parents=True, exist_ok=True)
    names = ("future_anchors_20_rdf.npz", "past_anchors_20_rdf.npz")
    if not all((anchors / n).is_file() for n in names):
        archive = hf_hub_download(
            "RainyNight/DVGT_demo_dataset",
            "dvgt_2_anchors.zip",
            repo_type="dataset",
            local_dir=directory,
        )
        with zipfile.ZipFile(archive) as z:
            for entry in z.namelist():
                name = Path(entry).name
                if name in names:
                    (anchors / name).write_bytes(z.read(entry))
        if not all((anchors / n).is_file() for n in names):
            raise RuntimeError("Official anchor archive did not contain expected files")
    if args.weights:
        checkpoint = Path(
            hf_hub_download(
                lock["checkpoint_repo"],
                lock["checkpoint_file"],
                revision=lock["checkpoint_revision"],
                local_dir=directory,
            )
        )
        if checkpoint.stat().st_size != lock["checkpoint_bytes"]:
            raise RuntimeError("Official checkpoint size mismatch")
    else:
        checkpoint = directory / lock["checkpoint_file"]
    if args.verify_checksum:
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != lock["checkpoint_sha256"]:
            raise RuntimeError("Official checkpoint SHA256 mismatch")
        print("Verified checkpoint SHA256:", digest.hexdigest())
    print("DVGT-F4 dependencies prepared. Checkpoint:", checkpoint)


if __name__ == "__main__":
    main()
