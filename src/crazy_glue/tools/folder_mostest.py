"""
Tool: folder_mostest

Find the folder with the most children (files + subdirectories) in a hierarchy.
"""

from pathlib import Path

import pydantic


class FolderMostestResult(pydantic.BaseModel):
    """Structured result for folder_mostest tool."""

    success: bool = pydantic.Field(description="Whether operation succeeded")
    message: str = pydantic.Field(description="Human-readable result message")
    data: dict | None = pydantic.Field(
        default=None,
        description="Folder path, child count, and breakdown",
    )
    error: str | None = pydantic.Field(
        default=None,
        description="Error message if operation failed",
    )


async def folder_mostest(path: str = ".") -> FolderMostestResult:
    """
    Find the folder with the most direct children in the hierarchy.

    Args:
        path: Root path to start searching (default: current directory)

    Returns:
        FolderMostestResult: The folder with most children and its count.
    """
    try:
        target = Path(path).resolve()
        if not target.exists():
            return FolderMostestResult(
                success=False,
                message=f"Path '{path}' does not exist",
                error="PathNotFound",
            )

        if not target.is_dir():
            return FolderMostestResult(
                success=False,
                message=f"Path '{path}' is not a directory",
                error="NotADirectory",
            )

        max_folder = None
        max_count = -1

        # Walk the directory tree
        for folder in target.rglob("*"):
            if not folder.is_dir():
                continue
            try:
                children = list(folder.iterdir())
                count = len(children)
                if count > max_count:
                    max_count = count
                    max_folder = folder
            except PermissionError:
                continue

        # Also check the root
        try:
            root_children = list(target.iterdir())
            if len(root_children) > max_count:
                max_count = len(root_children)
                max_folder = target
        except PermissionError:
            pass

        if max_folder is None:
            return FolderMostestResult(
                success=True,
                message="No accessible folders found",
                data={"path": str(target), "winner": None, "count": 0},
            )

        return FolderMostestResult(
            success=True,
            message=f"'{max_folder}' has the most children: {max_count}",
            data={
                "path": str(target),
                "winner": str(max_folder),
                "count": max_count,
            },
        )

    except Exception as e:
        return FolderMostestResult(
            success=False,
            message=f"Operation failed: {e}",
            error=str(e),
        )
