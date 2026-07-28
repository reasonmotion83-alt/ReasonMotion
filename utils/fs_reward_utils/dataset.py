"""
此模組已合併至 utils/finefs.py，統一維護。
保留此檔案只為了向下相容舊 import，請直接從 utils.finefs 匯入。
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
