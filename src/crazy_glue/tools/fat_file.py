"""
Tool: fat_file

Find the largest file in a directory tree.
"""

from pathlib import Path

import pydantic


class FatFileResult(pydantic.BaseModel):
    """Structured result for fat_file tool."""

    success: bool = pydantic.Field(description="Whether operation succeeded")
    message: str = pydantic.Field(description="Human-readable result message")
    data: dict | None = pydantic.Field(
        default=None,
        description="File path, size in bytes, and human-readable size",
    )
    error: str | None = pydantic.Field(
        default=None,
        description="Error message if operation failed",
    )


def _human_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


async def fat_file(path: str = ".") -> FatFileResult:
    """
    Find the largest file in the directory hierarchy.

    Args:
        path: Root path to start searching (default: current directory)

    Returns:
        FatFileResult: The largest file path and its size.
    """
    try:
        target = Path(path).resolve()
        if not target.exists():
            return FatFileResult(
                success=False,
                message=f"Path '{path}' does not exist",
                error="PathNotFound",
            )

        if not target.is_dir():
            return FatFileResult(
                success=False,
                message=f"Path '{path}' is not a directory",
                error="NotADirectory",
            )

        largest_file = None
        largest_size = -1

        for file_path in target.rglob("*"):
            if not file_path.is_file():
                continue
            try:
                size = file_path.stat().st_size
                if size > largest_size:
                    largest_size = size
                    largest_file = file_path
            except (PermissionError, OSError):
                continue

        if largest_file is None:
            return FatFileResult(
                success=True,
                message="No accessible files found",
                data={"path": str(target), "largest": None, "size_bytes": 0},
            )

        return FatFileResult(
            success=True,
            message=f"'{largest_file.name}': {_human_size(largest_size)}",
            data={
                "path": str(target),
                "largest": str(largest_file),
                "size_bytes": largest_size,
                "size_human": _human_size(largest_size),
            },
        )

    except Exception as e:
        return FatFileResult(
            success=False,
            message=f"Operation failed: {e}",
            error=str(e),
        )
