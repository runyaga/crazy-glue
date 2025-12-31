"""
Analysis Package - Modular components for the System Architect room.

This package contains:
- context: AnalysisContext dataclass
- parser: Command parsing
- handlers: Command handlers with registry
- formatters: Output formatting
- validators: Input validation
- room_editor: RoomConfigEditor class for YAML operations
- tools/: Operation implementations
"""

from crazy_glue.analysis.context import AnalysisContext
from crazy_glue.analysis.handlers import HANDLERS
from crazy_glue.analysis.parser import ParsedCommand
from crazy_glue.analysis.parser import parse_command
from crazy_glue.analysis.room_editor import MANAGED_ROOMS_FILE
from crazy_glue.analysis.room_editor import RoomConfigEditor

__all__ = [
    "AnalysisContext",
    "HANDLERS",
    "ParsedCommand",
    "parse_command",
    "RoomConfigEditor",
    "MANAGED_ROOMS_FILE",
]
