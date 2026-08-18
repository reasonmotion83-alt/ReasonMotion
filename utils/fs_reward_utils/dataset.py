"""
This module has been merged into utils/finefs.py, which is now the single source of truth.
This file is kept only for backward compatibility with old imports; please import directly from utils.finefs.
"""
# Re-export everything from the unified module
from utils.finefs import (   # noqa: F401
    FineFS,
    build_mask,
    expand_motion_name,
    random_rotate_y,
    MOTION_NAME_MAP,
    ROTATION_MAP,
    EDGES,
    BONE_LINKS,
)
