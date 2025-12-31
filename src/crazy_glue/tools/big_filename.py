"""
Tool: big_filename

Find the file or folder with the longest filename in a directory.
"""

from pathlib import Path

import pydantic


class BigFilenameResult(pydantic.BaseModel):
    found: bool
    path: str | None
    filename: str | None
    is_dir: bool | None
    error: str | None = None


async def big_filename(path: str = ".") -> BigFilenameResult:
    """Find the file or folder with the longest filename in a directory."""
    try:
        target = Path(path).resolve(strict=True)
        all_entries = [
            p for p in target.rglob("*") if p.is_file() or p.is_dir()
        ]
        if not all_entries:
            return BigFilenameResult(
                found=False, path=None, filename=None, is_dir=None
            )
        latest = max(all_entries, key=lambda p: len(p.name))
        return BigFilenameResult(
            found=True,
            path=str(latest.resolve()),
            filename=latest.name,
            is_dir=latest.is_dir(),
        )
    except Exception as exc:
        return BigFilenameResult(
            found=False, path=None, filename=None, is_dir=None, error=str(exc)
        )
