"""Capture standalone registration inputs/outputs from a real suite2p run.

This script monkeypatches suite2p registration at runtime, runs one
registration-only pipeline execution on real data, and writes serialized
entry/exit artifacts for standalone replay.
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

from separation.registration.serialization import serialize_object # only used for packing the registration_wrapper input/output into pickle-safe structures, not for any recursive or complex serialization logic that would require a more robust approach.


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
        default=Path("/mnt/nas02/Dataset/suite2p/output/registration_capture"),
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
    registration_pkg = importlib.import_module("suite2p.registration")

    run_s2p = run_module.run_s2p
    default_db = params_module.default_db
    default_settings = params_module.default_settings

    original_wrapper = registration_pkg.registration_wrapper
    signature = inspect.signature(original_wrapper)
    captured: Dict[str, Any] = {}

    def wrapped_registration_wrapper(*wrapper_args, **wrapper_kwargs):
        bound = signature.bind_partial(*wrapper_args, **wrapper_kwargs)
        bound.apply_defaults()
        captured["input"] = serialize_object(dict(bound.arguments))
        out = original_wrapper(*wrapper_args, **wrapper_kwargs)
        captured["output"] = serialize_object(out)
        return out

    registration_pkg.registration_wrapper = wrapped_registration_wrapper
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
        settings["run"]["do_detection"] = False
        settings["run"]["do_deconvolution"] = False
        settings["run"]["do_regmetrics"] = False
        settings["io"]["delete_bin"] = False

        run_s2p(settings=settings, db=db)
    finally:
        registration_pkg.registration_wrapper = original_wrapper

    if "input" not in captured or "output" not in captured:
        raise RuntimeError(
            "Registration capture failed: monkeypatched registration_wrapper was not invoked."
        )

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
        "registration_settings": settings["registration"],
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
