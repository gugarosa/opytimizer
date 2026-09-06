# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import ast
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Format hook-selected Python files without discarding the required docstring layout.

    Black owns code formatting and repository settings. The final pass restores the explicit blank lines
    required around docstrings, which Black otherwise collapses for single-sentence descriptions.

    Returns:
        The Black exit code or zero after formatting succeeds.

    """

    paths = [Path(name) for name in sys.argv[1:]]
    if any(not path.is_file() for path in paths):
        raise ValueError("`files` must identify existing Python files.")

    result = subprocess.run([sys.executable, "-m", "black", *sys.argv[1:]], check=False)
    if result.returncode:
        return result.returncode

    for path in paths:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        starts = [0]
        for line in lines:
            starts.append(starts[-1] + len(line))

        edits = []
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if ast.get_docstring(node) is None:
                continue
            literal = node.body[0]
            start_column = len(lines[literal.lineno - 1].encode("utf-8")[: literal.col_offset].decode("utf-8"))
            end_column = len(lines[literal.end_lineno - 1].encode("utf-8")[: literal.end_col_offset].decode("utf-8"))
            start = starts[literal.lineno - 1] + start_column
            end = starts[literal.end_lineno - 1] + end_column
            token = text[start:end]
            quote = token[-3:]
            opening = token.index(quote) + len(quote)
            content = token[opening : -len(quote)].rstrip()
            indent = " " * literal.col_offset
            replacement = token[:opening] + content + "\n\n" + indent + quote
            edits.append((start, end, replacement))

            if len(node.body) > 1:
                after = starts[literal.end_lineno]
                following = literal.end_lineno
                while following < len(lines) and not lines[following].strip():
                    following += 1
                edits.append((after, starts[following], "\n"))

        for start, end, replacement in sorted(edits, reverse=True):
            text = text[:start] + replacement + text[end:]

        ast.parse(text, filename=str(path))
        if path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8", newline="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
