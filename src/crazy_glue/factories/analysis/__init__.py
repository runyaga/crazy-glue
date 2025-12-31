"""
Analysis Room Package - Modular components for the System Architect.

This package contains:
- room_editor: RoomConfigEditor class for YAML operations
- commands: Command handlers for room management
"""

from crazy_glue.factories.analysis.room_editor import MANAGED_ROOMS_FILE
from crazy_glue.factories.analysis.room_editor import RoomConfigEditor

__all__ = ["RoomConfigEditor", "MANAGED_ROOMS_FILE"]
