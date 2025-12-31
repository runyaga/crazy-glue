"""Validation utilities for the System Architect room."""

from __future__ import annotations

import keyword
import re
import subprocess
from pathlib import Path

# Python reserved keywords that can't be used as identifiers
PYTHON_KEYWORDS = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
})


def sanitize_identifier(name: str) -> tuple[str, str | None]:
    """Convert a user-provided name into a valid Python identifier.

    Args:
        name: User-provided tool/function name

    Returns:
        Tuple of (sanitized_name, error_message).
        If error_message is not None, sanitization failed.
    """
    if not name or not name.strip():
        return "", "Name cannot be empty"

    # Start with lowercase, strip whitespace
    result = name.strip().lower()

    # Replace common separators with underscores
    result = re.sub(r"[-\s.]+", "_", result)

    # Remove any character that isn't alphanumeric or underscore
    result = re.sub(r"[^a-z0-9_]", "", result)

    # Collapse multiple underscores
    result = re.sub(r"_+", "_", result)

    # Strip leading/trailing underscores
    result = result.strip("_")

    # If starts with digit, prefix with underscore
    if result and result[0].isdigit():
        result = f"_{result}"

    # Check if empty after sanitization
    if not result:
        return "", f"Name '{name}' contains no valid identifier characters"

    # Check for Python keywords
    if keyword.iskeyword(result) or result in PYTHON_KEYWORDS:
        result = f"{result}_tool"

    # Validate it's a valid identifier
    if not result.isidentifier():
        return "", f"Could not create valid identifier from '{name}'"

    return result, None


def validate_config(project_root: Path) -> list[str]:
    """Run soliplex config validation and return errors."""
    installation_yaml = project_root / "installation.yaml"

    if not installation_yaml.exists():
        return []

    try:
        result = subprocess.run(
            ["soliplex-cli", "check-config", str(installation_yaml)],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(project_root),
        )

        errors = []
        if result.returncode != 0:
            # Parse output for validation errors
            for line in result.stdout.split("\n"):
                line_lower = line.lower()
                is_validation_err = "validation error" in line_lower
                is_other_err = "error" in line_lower and ":" in line
                if is_validation_err or is_other_err:
                    errors.append(line.strip())

            if not errors and result.stderr:
                snippet = result.stderr[:200]
                errors.append(f"Config validation failed: {snippet}")

        return errors
    except FileNotFoundError:
        return ["soliplex-cli not found in PATH"]
    except subprocess.TimeoutExpired:
        return ["Config validation timed out"]
    except Exception as e:
        return [f"Config validation error: {e}"]


def validate_python_syntax(code: str, filename: str = "<generated>") -> str | None:
    """Validate Python syntax.

    Returns:
        Error message if invalid, None if valid.
    """
    import ast

    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"Syntax error: {e}"


def validate_python_compiles(code: str, filename: str = "<generated>") -> str | None:
    """Validate Python code compiles.

    Returns:
        Error message if invalid, None if valid.
    """
    try:
        compile(code, filename, "exec")
        return None
    except Exception as e:
        return f"Compilation error: {e}"
