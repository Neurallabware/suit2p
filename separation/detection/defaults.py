"""Local defaults for standalone detection."""

from copy import deepcopy


_DETECTION_DEFAULTS = {
    "algorithm": "sparsery",
    "denoise": False,
    "block_size": (64, 64),
    "nbins": 5000,
    "bin_size": None,
    "highpass_time": 100,
    "threshold_scaling": 1.0,
    "npix_norm_min": 0.0,
    "npix_norm_max": 100,
    "max_overlap": 0.75,
    "soma_crop": True,
    "chan2_threshold": 0.25,
    "cellpose_chan2": False,
    "sparsery_settings": {
        "highpass_neuropil": 25,
        "max_ROIs": 5000,
        "spatial_scale": 0,
        "active_percentile": 0.0,
    },
    "sourcery_settings": {
        "connected": True,
        "max_iterations": 20,
        "smooth_masks": False,
    },
    "cellpose_settings": {
        "cellpose_model": "cpsam",
        "img": "max_proj / meanImg",
        "highpass_spatial": 0,
        "flow_threshold": 0.4,
        "cellprob_threshold": 0.0,
        "params": None,
        "params_chan2": None,
    },
}

_EXTRACTION_DEFAULTS = {
    "snr_threshold": 0.0,
    "batch_size": 500,
    "neuropil_extract": True,
    "neuropil_coefficient": 0.7,
    "inner_neuropil_radius": 2,
    "min_neuropil_pixels": 350,
    "lam_percentile": 50.0,
    "allow_overlap": False,
    "circular_neuropil": False,
}


def default_settings():
    """Return standalone detection defaults in nested form."""
    return {
        "detection": deepcopy(_DETECTION_DEFAULTS),
        "extraction": deepcopy(_EXTRACTION_DEFAULTS),
    }
