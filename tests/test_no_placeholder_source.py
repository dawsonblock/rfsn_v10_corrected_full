"""Guard test: no Python source file may contain literal placeholder text.

This test catches generated-code debris — files that contain prose like
"FULL COMPLETE CODE WITH FUNCTIONAL INTERFACE HERE" instead of real Python.
Such files compile as SyntaxErrors and must never reach the package.

If this test fails, locate the offending file and either implement the
feature or replace the file with a valid disabled stub.
"""
from __future__ import annotations

from pathlib import Path

# Patterns that indicate a file is a placeholder and not real Python.
# These are split to avoid the test matching itself.
_P = "FULL"
FORBIDDEN_PATTERNS = [
    _P + " COMPLETE CODE",
    _P + " CODE FOR",
    "FUNCTIONAL INTERFACE HERE",
    "PLACEHOLDER ONLY",
    "PLACEHOLDER CODE",
    "INSERT CODE HERE",
    "TODO GENERATED",
    "TODO: IMPLEMENT",
]

# This file is the source-guard test itself — skip it to avoid self-match.
_THIS_FILE = "test_no_placeholder_source.py"

# Directories to skip (build artifacts, dependencies, etc.)
SKIP_DIRS = {
    ".venv", "venv", "dist", "build", ".eggs", "__pycache__", "node_modules",
}


def test_no_generated_placeholder_text_in_python_sources():
    """Every .py file must be valid Python, not placeholder text."""
    root = Path(__file__).resolve().parents[1]
    offenders = []

    for path in sorted(root.rglob("*.py")):
        # Skip excluded directories and this file itself
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == _THIS_FILE:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                # Report relative path for readability
                rel = path.relative_to(root)
                # Find the first matching line for context
                for lineno, line in enumerate(text.splitlines(), 1):
                    if pattern in line:
                        offenders.append(
                            f"{rel}:{lineno}: contains {pattern!r}"
                        )
                        break

    assert not offenders, (
        f"Found {len(offenders)} source file(s) with placeholder text "
        f"(replace with valid Python or a disabled stub):\n"
        + "\n".join(f"  {o}" for o in offenders)
    )
