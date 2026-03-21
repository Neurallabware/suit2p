"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import sys

# When vendored under SOMMv2, map the canonical top-level name to this package
# so internal imports like "from suite2p.xxx import ..." continue to work.
sys.modules.setdefault("suite2p", sys.modules[__name__])

from suite2p.version import version, version_str
from suite2p.parameters import default_settings, user_settings, SETTINGS_FOLDER, default_db
from suite2p.pipeline_s2p import pipeline
from suite2p.run_s2p import run_s2p, run_plane
from suite2p.detection import detection_wrapper
from suite2p.classification import classify
from suite2p.extraction import extraction_wrapper
from suite2p.registration import registration_wrapper

name = "suite2p"
