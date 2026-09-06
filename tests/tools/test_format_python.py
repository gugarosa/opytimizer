# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _formatter():
    path = Path(__file__).resolve().parents[2] / "tools" / "format_python.py"
    spec = importlib.util.spec_from_file_location("project_formatter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_main_preserves_docstring_content_and_reaches_a_stable_layout(monkeypatch, tmp_path):
    formatter = _formatter()
    path = tmp_path / "sample.py"
    source = (
        'r"""Describe a L\u00e9vy path with \\alpha."""\n\n'
        'class Example:\n    """Hold a value."""\n    value = 1\n\n'
        '    def read(self):\n        """Return the value."""\n        return self.value\n'
    )
    path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(formatter.sys, "argv", ["format_python.py", str(path)])
    monkeypatch.setattr(formatter.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))

    assert formatter.main() == 0
    formatted = path.read_text(encoding="utf-8")
    assert '\n\n    """\n\n    value = 1' in formatted

    def canonical(tree):
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                doc = ast.get_docstring(node)
                if doc is not None:
                    node.body[0].value.value = doc.strip()

        return ast.dump(tree)

    assert canonical(ast.parse(source)) == canonical(ast.parse(formatted))
    assert formatter.main() == 0
    assert path.read_text(encoding="utf-8") == formatted


def test_main_propagates_black_failure_without_rewriting_files(monkeypatch, tmp_path):
    formatter = _formatter()
    path = tmp_path / "sample.py"
    source = '"""Keep the original text."""\n'
    path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(formatter.sys, "argv", ["format_python.py", str(path)])
    monkeypatch.setattr(formatter.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=123))

    assert formatter.main() == 123
    assert path.read_text(encoding="utf-8") == source
