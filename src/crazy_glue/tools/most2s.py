"""
Tool: most2s

Recursively find the filename with the most '2's, returning structured result or indicating a draw.
"""

import pydantic
from pathlib import Path
from typing import Optional


class Most2sResult(pydantic.BaseModel):
    """Result of searching for the filename with the most '2' characters.

    Attributes:
        directory: The root directory that was searched.
        best_file: Full path to the file that contains the maximum number of '2's; ``None`` if no such file exists (draw).
        twos_count: Number of '2's in the ``best_file``; ``None`` if a draw.
        draw: ``True`` if no filename contains any '2', ``False`` otherwise.
        error: Error message if an exception occurred; otherwise ``None``.
    """

    directory: str
    best_file: Optional[str]
    twos_count: Optional[int]
    draw: bool
    error: Optional[str] = None


async def most2s(path: str = ".") -> Most2sResult:
    """Recursively search the given directory for the file whose name contains the most '2' characters.

    The function returns the full path to this file and the count of '2's.  If no file contains a '2', the
    result is marked as a draw.  Any exception encountered during the search is captured and returned in
    the model's ``error`` field.

    Parameters
    ----------
    path: str
        The directory to search, relative or absolute. Defaults to the current working directory.

    Returns
    -------
    Most2sResult
        Structured information about the search, including the winning filename, number of '2's, draw
        status, and any error message.
    """
    try:
        start_path = Path(path).resolve()
        if not start_path.exists():
            raise FileNotFoundError(f"Path '{path}' does not exist.")

        max_twos: int = -1
        best_file: Optional[str] = None

        for file_path in start_path.rglob("*"):
            if file_path.is_file():
                count = str(file_path.name).count("2")
                if count > max_twos:
                    max_twos = count
                    best_file = str(file_path)

        if max_twos <= 0:
            # No filename contains a '2' – draw
            return Most2sResult(
                directory=str(start_path),
                best_file=None,
                twos_count=None,
                draw=True,
                error=None,
            )
        else:
            return Most2sResult(
                directory=str(start_path),
                best_file=best_file,
                twos_count=max_twos,
                draw=False,
                error=None,
            )
    except Exception as exc:  # pragma: no cover - capture any runtime error
        return Most2sResult(
            directory=path,
            best_file=None,
            twos_count=None,
            draw=False,
            error=str(exc),
        )