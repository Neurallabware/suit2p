"""
Batch suite2p processing for downsampled 2P calcium imaging data.

Processes downsampled TIF files from /mnt/nas02/WMR/pico-2p/20260114_pico_20x/
through the full suite2p pipeline (registration, detection, extraction,
deconvolution, classification) and generates post-processing figures.

Usage:
    python run_suite2p_batch.py                          # first 5 sessions
    python run_suite2p_batch.py --sessions mouse_01 mouse_02
    python run_suite2p_batch.py --all                     # all 31 sessions
    python run_suite2p_batch.py --dry-run                 # list sessions only
    python run_suite2p_batch.py --figures-only             # regenerate figures only
"""

import os
import sys
import time
import struct
import zipfile
import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.io import savemat
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suite2p import run_s2p
from suite2p.parameters import default_settings, default_db

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_PATH = "/mnt/nas02/WMR/pico-2p/20260114_pico_20x"
DATE_TAG = datetime.now().strftime("%Y%m%d")

ALL_SESSIONS = (
    [f"mouse_{i:02d}" for i in range(1, 11)]
    + [f"mouse02_{i:02d}" for i in range(1, 13)]
    + [f"mouse03_{i:02d}" for i in range(1, 11) if i != 4]
)


# ── Session Discovery ─────────────────────────────────────────────────────────
def build_session_list(session_names=None):
    """Build list of session dicts with validated paths."""
    names = session_names if session_names else ALL_SESSIONS
    sessions = []
    for name in names:
        session_dir = os.path.join(BASE_PATH, name)
        tif_name = f"2p_{name}_ds.tif"
        tif_path = os.path.join(session_dir, tif_name)
        if not os.path.isdir(session_dir):
            logger.warning(f"Session dir not found, skipping: {session_dir}")
            continue
        if not os.path.isfile(tif_path):
            logger.warning(f"Downsampled TIF not found, skipping: {tif_path}")
            continue
        sessions.append({
            "name": name,
            "dir": session_dir,
            "tif_name": tif_name,
            "tif_path": tif_path,
        })
    return sessions


# ── Suite2p Configuration ──────────────────────────────────────────────────────
def make_db(session):
    """Create the db dict for a session."""
    save_folder = f"suite2p_yz_{DATE_TAG}"
    return {
        "data_path": [session["dir"]],
        "file_list": [session["tif_name"]],
        "save_path0": session["dir"],
        "save_folder": save_folder,
        "fast_disk": os.path.join(session["dir"], save_folder),
        "input_format": "tif",
        "nplanes": 1,
        "nchannels": 1,
        "keep_movie_raw": True,
    }


def make_settings():
    """Create settings dict with hyperparameters for this dataset."""
    settings = default_settings()

    # Core imaging parameters
    settings["fs"] = 7.5               # frame rate (Hz)
    settings["tau"] = 1.0              # GCaMP decay constant (s)
    settings["diameter"] = [10, 10]    # neuron diameter in px (after 4x downsample)

    # Pipeline steps
    settings["run"]["do_registration"] = 1
    settings["run"]["do_regmetrics"] = False  # only 900 frames, below 1500 practical min
    settings["run"]["do_detection"] = True
    settings["run"]["do_deconvolution"] = True

    # IO
    settings["io"]["save_mat"] = True
    settings["io"]["delete_bin"] = False

    # Registration
    settings["registration"]["reg_tif"] = True       # save corrected movie
    settings["registration"]["nonrigid"] = True
    settings["registration"]["nimg_init"] = 300       # 1/3 of 900 frames

    # Detection
    settings["detection"]["denoise"] = True           # PCA denoising (noisy data)
    settings["detection"]["soma_crop"] = True          # soma-only
    settings["detection"]["threshold_scaling"] = 0.9   # tuned to match ~16 manual ROIs on mouse_01

    # Classification
    settings["classification"]["use_builtin_classifier"] = True
    settings["classification"]["preclassify"] = 0.0    # keep all ROIs

    return settings


# ── Registered Movie Stitching ─────────────────────────────────────────────────
def stitch_registered_tif(plane_path, session_name):
    """Stitch individual reg_tif frames into a single multi-frame TIF."""
    reg_tif_dir = os.path.join(plane_path, "reg_tif")
    if not os.path.isdir(reg_tif_dir):
        logger.warning(f"  No reg_tif/ directory found for {session_name}")
        return None

    frame_files = sorted([
        f for f in os.listdir(reg_tif_dir)
        if f.endswith(".tif") and f.startswith("file")
    ])
    if not frame_files:
        logger.warning(f"  No TIF frames in reg_tif/ for {session_name}")
        return None

    out_path = os.path.join(plane_path, f"registered_{session_name}.tif")
    with tifffile.TiffWriter(out_path, bigtiff=True) as writer:
        for ff in frame_files:
            img = tifffile.imread(os.path.join(reg_tif_dir, ff))
            # Each file may contain multiple frames
            if img.ndim == 2:
                writer.write(img, contiguous=True)
            else:
                for frame in img:
                    writer.write(frame, contiguous=True)

    logger.info(f"  Stitched registered movie: {out_path}")
    return out_path


# ── ImageJ ROI Parsing ─────────────────────────────────────────────────────────
def parse_imagej_roi(data):
    """Parse a single ImageJ .roi file. Returns dict with type, bounds."""
    if data[:4] != b'Iout':
        return None
    roi_type = data[6]
    top = struct.unpack('>h', data[8:10])[0]
    left = struct.unpack('>h', data[10:12])[0]
    bottom = struct.unpack('>h', data[12:14])[0]
    right = struct.unpack('>h', data[14:16])[0]
    type_map = {0: 'polygon', 1: 'rect', 2: 'oval', 7: 'freehand', 10: 'point'}
    return {
        "type": type_map.get(roi_type, f"unknown({roi_type})"),
        "top": top, "left": left, "bottom": bottom, "right": right,
        "cy": (top + bottom) / 2.0, "cx": (left + right) / 2.0,
        "h": bottom - top, "w": right - left,
    }


def load_imagej_roiset(zip_path):
    """Load all ROIs from an ImageJ RoiSet.zip file."""
    rois = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith('.roi'):
                continue
            roi = parse_imagej_roi(zf.read(name))
            if roi is not None:
                roi["name"] = name
                rois.append(roi)
    return rois


def rois_to_masks(rois, Ly, Lx):
    """Convert ImageJ ROIs to binary mask stack (n_rois, Ly, Lx)."""
    masks = np.zeros((len(rois), Ly, Lx), dtype=bool)
    for i, roi in enumerate(rois):
        if roi["type"] == "oval":
            cy, cx = roi["cy"], roi["cx"]
            ry, rx = roi["h"] / 2.0, roi["w"] / 2.0
            yy, xx = np.ogrid[:Ly, :Lx]
            masks[i] = ((yy - cy)**2 / max(ry**2, 1e-6) +
                         (xx - cx)**2 / max(rx**2, 1e-6)) <= 1.0
        else:  # rect or fallback
            t, l, b, r = roi["top"], roi["left"], roi["bottom"], roi["right"]
            t, l = max(0, t), max(0, l)
            b, r = min(Ly, b), min(Lx, r)
            masks[i, t:b, l:r] = True
    return masks


# ── Ground Truth Output ────────────────────────────────────────────────────────
def save_ground_truth(plane_path, session_name):
    """Save suite2p detected components in naomi-style GT format.

    Outputs to plane_path/ground_truth/:
      - seg_masks.tif: boolean mask stack (n_cells, Ly, Lx)
      - seg_max_proj.tif: meanImg as float32
      - ground_truth.pkl: dict with masks, centers, traces
      - ground_truth_summary.png: 3-panel plot
    """
    gt_dir = os.path.join(plane_path, "ground_truth")
    os.makedirs(gt_dir, exist_ok=True)

    ops = np.load(os.path.join(plane_path, "ops.npy"), allow_pickle=True).item()
    stat = np.load(os.path.join(plane_path, "stat.npy"), allow_pickle=True)
    iscell = np.load(os.path.join(plane_path, "iscell.npy"), allow_pickle=True)
    F = np.load(os.path.join(plane_path, "F.npy"), allow_pickle=True)
    Fneu = np.load(os.path.join(plane_path, "Fneu.npy"), allow_pickle=True)

    Ly, Lx = ops["Ly"], ops["Lx"]
    cell_ids = np.where(iscell[:, 0].astype(bool))[0]
    n_cells = len(cell_ids)
    mean_img = ops.get("meanImg", np.zeros((Ly, Lx)))

    # Build masks from suite2p stat
    masks = np.zeros((n_cells, Ly, Lx), dtype=bool)
    centers = np.zeros((n_cells, 2))
    for i, idx in enumerate(cell_ids):
        s = stat[idx]
        ypix, xpix = s["ypix"], s["xpix"]
        valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
        masks[i, ypix[valid], xpix[valid]] = True
        centers[i] = s["med"]

    # 1. seg_masks.tif
    tifffile.imwrite(os.path.join(gt_dir, "seg_masks.tif"),
                     masks.astype(np.uint8) * 255)

    # 2. seg_max_proj.tif
    tifffile.imwrite(os.path.join(gt_dir, "seg_max_proj.tif"),
                     mean_img.astype(np.float32))

    # 3. ground_truth.pkl
    # Compute dF/F traces for cells
    dff = np.zeros_like(F[cell_ids])
    for i, idx in enumerate(cell_ids):
        trace = F[idx] - 0.7 * Fneu[idx]
        f0 = np.percentile(trace, 10)
        dff[i] = (trace - f0) / max(f0, 1e-6)

    gt_data = {
        "seg_masks": masks,
        "seg_traces": dff,
        "seg_neuron_ids": cell_ids,
        "n_neurons": n_cells,
        "roi_centers": centers,
        "source": "suite2p",
    }
    with open(os.path.join(gt_dir, "ground_truth.pkl"), "wb") as f:
        pickle.dump(gt_data, f)

    # 4. ground_truth_summary.png — 3-panel naomi-style
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.hsv(np.linspace(0, 1, n_cells, endpoint=False))

    # Panel 1: Colored neuron masks on black
    mask_rgb = np.zeros((Ly, Lx, 3))
    for i in range(n_cells):
        for c in range(3):
            mask_rgb[:, :, c] += masks[i].astype(float) * colors[i, c]
    axes[0].imshow(np.clip(mask_rgb, 0, 1))
    axes[0].set_title(f"{n_cells} neuron masks")
    axes[0].axis("off")

    # Panel 2: Max projection with colored circles
    axes[1].imshow(mean_img, cmap="gray", vmin=np.percentile(mean_img, 1),
                   vmax=np.percentile(mean_img, 99.5))
    for i in range(n_cells):
        s = stat[cell_ids[i]]
        radius = np.sqrt(s["npix"] / np.pi)
        circle = plt.Circle((centers[i, 1], centers[i, 0]), radius,
                            fill=False, edgecolor=colors[i], linewidth=1.5)
        axes[1].add_patch(circle)
    axes[1].set_title("Max projection")
    axes[1].axis("off")

    # Panel 3: Sample dF/F traces
    fs = ops.get("fs", 7.5)
    t = np.arange(F.shape[1]) / fs
    n_show = min(10, n_cells)
    for j in range(n_show):
        axes[2].plot(t, dff[j] + j * 2, linewidth=0.7, color=colors[j])
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("dF/F (offset)")
    axes[2].set_title(f"Sample dF/F ({n_show}/{n_cells})")

    fig.suptitle(f"Ground Truth Segmentation — {session_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(gt_dir, "ground_truth_summary.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  Ground truth saved to {gt_dir} ({n_cells} cells)")
    return gt_dir


# ── Post-Processing Figures ────────────────────────────────────────────────────
def get_roi_contours(stat_i, Ly, Lx):
    """Extract boundary pixels of an ROI."""
    mask = np.zeros((Ly, Lx), dtype=bool)
    ypix = stat_i["ypix"]
    xpix = stat_i["xpix"]
    valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
    mask[ypix[valid], xpix[valid]] = True
    interior = binary_erosion(mask)
    contour = mask & ~interior
    return np.argwhere(contour)  # (N, 2) array of (y, x)


def get_roi_filled_mask(stat_i, Ly, Lx):
    """Get filled mask of an ROI."""
    mask = np.zeros((Ly, Lx), dtype=bool)
    ypix = stat_i["ypix"]
    xpix = stat_i["xpix"]
    valid = (ypix >= 0) & (ypix < Ly) & (xpix >= 0) & (xpix < Lx)
    mask[ypix[valid], xpix[valid]] = True
    return mask


def generate_figures(plane_path, session_name):
    """Generate post-processing figures for a processed session."""
    fig_dir = os.path.join(plane_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ops = np.load(os.path.join(plane_path, "ops.npy"), allow_pickle=True).item()
    stat = np.load(os.path.join(plane_path, "stat.npy"), allow_pickle=True)
    iscell = np.load(os.path.join(plane_path, "iscell.npy"), allow_pickle=True)

    Ly = ops["Ly"]
    Lx = ops["Lx"]
    yrange = ops.get("yrange", [0, Ly])
    xrange = ops.get("xrange", [0, Lx])
    y0, x0 = yrange[0], xrange[0]
    Ly_crop = yrange[1] - yrange[0]
    Lx_crop = xrange[1] - xrange[0]

    # Use meanImg (full FOV) as the base grayscale image
    mean_img = ops.get("meanImg", np.zeros((Ly, Lx)))
    max_proj = ops.get("max_proj", None)

    # 1. Correlation image (gray)
    if "Vcorr" in ops:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(ops["Vcorr"], cmap="gray")
        ax.set_title(f"{session_name} — Correlation Image")
        ax.axis("off")
        fig.savefig(os.path.join(fig_dir, "corr_image.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 2. PNR / max projection image (gray, high contrast)
    if max_proj is not None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        vmin, vmax = np.percentile(max_proj, 0.5), np.percentile(max_proj, 99.9)
        ax.imshow(max_proj, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{session_name} — Max Projection")
        ax.axis("off")
        fig.savefig(os.path.join(fig_dir, "max_proj.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Max projection + ALL ROIs in red (before filtering)
    if max_proj is not None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        vmin, vmax = np.percentile(max_proj, 0.5), np.percentile(max_proj, 99.9)
        ax.imshow(max_proj, cmap="gray", vmin=vmin, vmax=vmax)
        for i in range(len(stat)):
            contour = get_roi_contours(stat[i], Ly, Lx)
            if len(contour) > 0:
                cy, cx = contour[:, 0] - y0, contour[:, 1] - x0
                valid = (cy >= 0) & (cy < Ly_crop) & (cx >= 0) & (cx < Lx_crop)
                if valid.any():
                    ax.scatter(cx[valid], cy[valid], s=0.5, c="red",
                               alpha=0.8, linewidths=0)
        ax.set_title(f"{session_name} — All ROIs ({len(stat)} detected)", color="red")
        ax.axis("off")
        fig.savefig(os.path.join(fig_dir, "max_proj_all_rois.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. Max projection + classified cells only in red (after filtering)
    if max_proj is not None:
        cell_mask = iscell[:, 0].astype(bool)
        n_cells = int(cell_mask.sum())
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        vmin, vmax = np.percentile(max_proj, 0.5), np.percentile(max_proj, 99.9)
        ax.imshow(max_proj, cmap="gray", vmin=vmin, vmax=vmax)
        for i in np.where(cell_mask)[0]:
            contour = get_roi_contours(stat[i], Ly, Lx)
            if len(contour) > 0:
                cy, cx = contour[:, 0] - y0, contour[:, 1] - x0
                valid = (cy >= 0) & (cy < Ly_crop) & (cx >= 0) & (cx < Lx_crop)
                if valid.any():
                    ax.scatter(cx[valid], cy[valid], s=0.5, c="red",
                               alpha=0.8, linewidths=0)
        ax.set_title(f"{session_name} — Cells Only ({n_cells} cells)", color="red")
        ax.axis("off")
        fig.savefig(os.path.join(fig_dir, "max_proj_cells_only.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 5. Motion field plot
    reg_path = os.path.join(plane_path, "reg_outputs.npy")
    if os.path.exists(reg_path):
        reg = np.load(reg_path, allow_pickle=True).item()
        if "yoff" in reg and "xoff" in reg:
            fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            axes[0].plot(reg["yoff"], linewidth=0.5)
            axes[0].set_ylabel("Y shift (px)")
            axes[0].set_title(f"{session_name} — Registration Shifts")
            axes[1].plot(reg["xoff"], linewidth=0.5)
            axes[1].set_ylabel("X shift (px)")
            axes[1].set_xlabel("Frame")
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "motion_field.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

    logger.info(f"  Figures saved to {fig_dir}")


# ── Main Processing ────────────────────────────────────────────────────────────
def run_single_session(session):
    """Run suite2p + post-processing for one session."""
    logger.info(f"=== Processing {session['name']} ===")
    logger.info(f"  Input: {session['tif_path']}")

    db = make_db(session)
    settings = make_settings()

    t0 = time.time()
    db_paths = run_s2p(db=db, settings=settings)
    elapsed = time.time() - t0
    logger.info(f"  suite2p completed in {elapsed:.1f}s")

    # Post-processing
    plane_path = os.path.join(session["dir"], f"suite2p_yz_{DATE_TAG}", "plane0")
    if os.path.isdir(plane_path):
        stitch_registered_tif(plane_path, session["name"])
        generate_figures(plane_path, session["name"])
        save_ground_truth(plane_path, session["name"])
    else:
        logger.warning(f"  plane0 output not found at {plane_path}")

    return db_paths


def run_postprocessing(session_name):
    """Run only post-processing (figures, stitching, GT) for an existing session."""
    session_dir = os.path.join(BASE_PATH, session_name)
    plane_path = os.path.join(session_dir, f"suite2p_yz_{DATE_TAG}", "plane0")

    if not os.path.isdir(plane_path):
        logger.error(f"No results at {plane_path}")
        return

    stitch_registered_tif(plane_path, session_name)
    generate_figures(plane_path, session_name)
    save_ground_truth(plane_path, session_name)


def main():
    parser = argparse.ArgumentParser(
        description="Batch suite2p processing for downsampled 2P data")
    parser.add_argument("--sessions", nargs="*", default=None,
                        help="Session names to process (e.g., mouse_01 mouse_02)")
    parser.add_argument("--all", action="store_true",
                        help="Process all 31 sessions")
    parser.add_argument("--dry-run", action="store_true",
                        help="List sessions that would be processed, then exit")
    parser.add_argument("--figures-only", action="store_true",
                        help="Regenerate figures only (skip suite2p)")
    parser.add_argument("--date-tag", default=None,
                        help="Override date tag (default: today YYYYMMDD)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    global DATE_TAG
    if args.date_tag:
        DATE_TAG = args.date_tag

    # Determine which sessions to process
    if args.all:
        session_names = None  # all sessions
    elif args.sessions:
        session_names = args.sessions
    else:
        session_names = ALL_SESSIONS[:5]

    sessions = build_session_list(session_names)
    logger.info(f"Found {len(sessions)} sessions to process")

    if args.dry_run:
        for s in sessions:
            print(f"  {s['name']:20s}  {s['tif_path']}")
        return

    if args.figures_only:
        for s in sessions:
            logger.info(f"Regenerating figures for {s['name']}")
            run_postprocessing(s["name"])
        return

    # Full processing
    t_total = time.time()
    results = {}
    for i, session in enumerate(sessions):
        logger.info(f"\n[{i+1}/{len(sessions)}] Starting {session['name']}")
        try:
            db_paths = run_single_session(session)
            results[session["name"]] = "OK"
            logger.info(f"  Completed {session['name']}")
        except Exception as e:
            results[session["name"]] = f"FAILED: {e}"
            logger.error(f"  FAILED {session['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    elapsed_total = time.time() - t_total
    logger.info(f"\n{'='*60}")
    logger.info(f"Batch complete: {elapsed_total:.1f}s total")
    for name, status in results.items():
        logger.info(f"  {name:20s}  {status}")


if __name__ == "__main__":
    main()
