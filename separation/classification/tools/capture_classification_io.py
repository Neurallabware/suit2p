"""Capture standalone classification inputs/outputs from a real suite2p run.

This script monkeypatches suite2p classification at runtime, runs one
registration+detection+classification execution on real data, and writes
serialized entry/exit artifacts for standalone replay.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import inspect
import json
from pathlib import Path
import pickle
from typing import Any, Dict

from separation.classification.serialization import serialize_object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/mnt/nas02/Dataset/suite2p/demo"),
        help="Directory containing real demo tiff data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/nas02/Dataset/suite2p/output/classification_capture"),
        help="Root directory for source pipeline output.",
    )
    parser.add_argument(
        "--file-name",
        type=str,
        default="file_00002_00001.tif",
        help="Single demo file used for fast real-data capture.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Where test_input.pkl/test_output.pkl/capture_manifest.json are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    run_dir = args.output_root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)

    run_module = importlib.import_module("suite2p.run_s2p")
    params_module = importlib.import_module("suite2p.parameters")
    classification_pkg = importlib.import_module("suite2p.classification")
    classification_mod = importlib.import_module("suite2p.classification.classify")

    run_s2p = run_module.run_s2p
    default_db = params_module.default_db
    default_settings = params_module.default_settings

    original_pkg_classify = classification_pkg.classify
    original_mod_classify = classification_mod.classify
    signature = inspect.signature(original_pkg_classify)
    captured: Dict[str, Any] = {}

    def wrapped_classify(*classify_args, **classify_kwargs):
        bound = signature.bind_partial(*classify_args, **classify_kwargs)
        bound.apply_defaults()
        captured["input"] = serialize_object(dict(bound.arguments))
        out = original_pkg_classify(*classify_args, **classify_kwargs)
        captured["output"] = serialize_object(out)
        return out

    classification_pkg.classify = wrapped_classify
    classification_mod.classify = wrapped_classify
    try:
        db = default_db()
        db.update(
            {
                "data_path": [str(args.data_path)],
                "save_path0": str(run_dir),
                "fast_disk": str(run_dir),
                "input_format": "tif",
                "file_list": [args.file_name],
                "keep_movie_raw": True,
            }
        )

        settings = default_settings()
        settings["torch_device"] = "cpu"
        settings["run"]["do_registration"] = 1
        settings["run"]["do_detection"] = True
        settings["run"]["do_deconvolution"] = False
        settings["run"]["do_regmetrics"] = False
        settings["io"]["delete_bin"] = False

        run_s2p(settings=settings, db=db)
    finally:
        classification_pkg.classify = original_pkg_classify
        classification_mod.classify = original_mod_classify

    if "input" not in captured or "output" not in captured:
        raise RuntimeError("Classification capture failed: classify was not invoked.")

    input_path = args.artifact_dir / "test_input.pkl"
    output_path = args.artifact_dir / "test_output.pkl"
    manifest_path = args.artifact_dir / "capture_manifest.json"

    with input_path.open("wb") as f:
        pickle.dump(captured["input"], f, protocol=pickle.HIGHEST_PROTOCOL)
    with output_path.open("wb") as f:
        pickle.dump(captured["output"], f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data_path),
        "file_name": args.file_name,
        "run_dir": str(run_dir),
        "device": "cpu",
        "db": db,
        "classification_settings": settings["classification"],
        "detection_settings": settings["detection"],
        "extraction_settings": settings["extraction"],
        "settings_run": settings["run"],
        "artifacts": {
            "test_input": str(input_path),
            "test_output": str(output_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Saved capture input:  {input_path}")
    print(f"Saved capture output: {output_path}")
    print(f"Saved manifest:       {manifest_path}")
    print("Capture complete.")


if __name__ == "__main__":
    main()
