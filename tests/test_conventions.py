# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.

import ast
import tokenize
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _python_files():
    for directory in ("opytimizer", "tests", "examples", "tools"):
        yield from sorted((ROOT / directory).rglob("*.py"))
    yield from sorted((ROOT / "docs").glob("*.py"))


def _sources():
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        yield path.relative_to(ROOT).as_posix(), text, ast.parse(text, feature_version=(3, 11))


def test_python_files_keep_license_headers():
    violations = []
    for path in _python_files():
        if path.read_text(encoding="utf-8").splitlines()[:2] != [
            "# Copyright (c) 2019-2026 Opytimizer contributors.",
            "# Licensed under the Apache License, Version 2.0.",
        ]:
            violations.append(path.relative_to(ROOT).as_posix())

    assert violations == []


def test_python_files_use_absolute_modern_imports():
    legacy = {
        "Callable",
        "Collection",
        "Container",
        "Dict",
        "Generator",
        "Iterable",
        "Iterator",
        "List",
        "Mapping",
        "MutableMapping",
        "MutableSequence",
        "Optional",
        "Sequence",
        "Set",
        "Tuple",
        "Union",
    }
    violations = []
    for path, text, tree in _sources():
        top_level = set(tree.body)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in top_level:
                violations.append((path, node.lineno, "nested import"))
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    violations.append((path, node.lineno, "relative import"))
                if node.module == "typing":
                    for alias in node.names:
                        if alias.name in legacy:
                            violations.append((path, node.lineno, alias.name))

    assert violations == []


def test_python_files_keep_runtime_validation_and_comments_explicit():
    violations = []
    for path, text, tree in _sources():
        library = path.startswith("opytimizer/")
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                violations.append((path, node.lineno, "bare except"))
            if library and isinstance(node, ast.Assert):
                violations.append((path, node.lineno, "runtime assert"))
            if library and isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                violations.append((path, node.lineno, "library print"))
            if path.startswith("tests/") and isinstance(node, ast.Assert) and node.msg is not None:
                violations.append((path, node.lineno, "assertion message"))

        comments = [
            token
            for token in tokenize.generate_tokens(StringIO(text).readline)
            if token.type == tokenize.COMMENT and token.start[0] > 2
        ]
        previous = 0
        consecutive = 0
        for token in comments:
            comment = token.string.removeprefix("#").strip()
            if comment.startswith(("noqa", "type: ignore", "pragma:", "fmt:", "isort:")):
                continue
            consecutive = consecutive + 1 if token.start[0] == previous + 1 else 1
            previous = token.start[0]
            if consecutive > 3:
                violations.append((path, token.start[0], "long comment block"))
            if comment.endswith("."):
                violations.append((path, token.start[0], "comment period"))
            if comment and not comment.strip("-=_*"):
                violations.append((path, token.start[0], "comment banner"))

    assert violations == []


def test_python_files_use_consistent_docstrings():
    violations = []
    hooks = {"compile", "evaluate", "update", "clip_by_bound", "forward", "compose"}
    for path, text, tree in _sources():
        lines = text.splitlines()
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = getattr(node, "name", "<module>")
            doc = ast.get_docstring(node)
            if doc is not None:
                doc = doc.strip()
            is_test = path.startswith("tests/")
            private = name.startswith("_") and not name.endswith("__")
            parent = parents.get(node)
            nested = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not isinstance(
                parent, (ast.Module, ast.ClassDef)
            )
            override = (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and isinstance(parent, ast.ClassDef)
                and bool(parent.bases)
                and (name in hooks or name.startswith("on_"))
            )
            if is_test and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
                if node.args.vararg is not None:
                    arguments.append(node.args.vararg)
                if node.args.kwarg is not None:
                    arguments.append(node.args.kwarg)
                if node.returns is not None or any(argument.annotation is not None for argument in arguments):
                    violations.append((path, node.lineno, "annotated test helper"))
            if is_test or private or nested or override:
                if doc is not None:
                    violations.append((path, getattr(node, "lineno", 1), "unnecessary docstring"))
                continue
            if not isinstance(node, ast.Module) and doc is None:
                violations.append((path, node.lineno, "missing public docstring"))
                continue
            if doc is None:
                continue

            literal = node.body[0]
            first = doc.split("\n\n", 1)[0]
            if len(first.splitlines()) != 1:
                violations.append((path, literal.lineno, "wrapped summary"))
            if isinstance(node, ast.ClassDef) and len(doc.splitlines()) != 1:
                data_class = any(isinstance(base, ast.Name) and base.id == "BaseModel" for base in node.bases) or any(
                    isinstance(decorator, ast.Name)
                    and decorator.id == "dataclass"
                    or isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "dataclass"
                    for decorator in node.decorator_list
                )
                if not data_class:
                    violations.append((path, literal.lineno, "regular class has extended documentation"))
            if literal.lineno == literal.end_lineno or lines[literal.end_lineno - 2].strip():
                violations.append((path, literal.end_lineno, "missing blank before docstring close"))
            elif literal.end_lineno >= 3 and not lines[literal.end_lineno - 3].strip():
                violations.append((path, literal.end_lineno, "extra blank before docstring close"))
            if len(node.body) > 1:
                if literal.end_lineno < len(lines) and lines[literal.end_lineno].strip():
                    violations.append((path, literal.end_lineno, "missing blank after docstring"))
                elif literal.end_lineno + 1 < len(lines) and not lines[literal.end_lineno + 1].strip():
                    violations.append((path, literal.end_lineno, "extra blank after docstring"))

            section = None
            for number, line in enumerate(doc.splitlines(), start=literal.lineno):
                stripped = line.strip()
                if line and not line.startswith(" ") and line.endswith(":"):
                    section = stripped
                    continue
                if section in {"Args:", "Returns:", "Raises:", "Attributes:"} and stripped:
                    indent = len(line) - len(line.lstrip())
                    if indent != 4:
                        violations.append((path, number, "wrapped docstring entry"))
                    if ";" in line or "defaults to " in line.lower():
                        violations.append((path, number, "docstring entry tail"))
                    if len(line) + literal.col_offset > 120:
                        violations.append((path, number, "long docstring entry"))

    assert violations == []


def test_library_errors_identify_the_offender():
    violations = []
    for path, text, tree in _sources():
        if not path.startswith("opytimizer/"):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call) or not node.exc.args:
                continue
            message = node.exc.args[0]
            if isinstance(message, ast.Constant) and isinstance(message.value, str):
                first = last = message.value
            elif isinstance(message, ast.JoinedStr):
                first = message.values[0].value if isinstance(message.values[0], ast.Constant) else ""
                last = message.values[-1].value if isinstance(message.values[-1], ast.Constant) else ""
            else:
                continue
            if not first.startswith("`") or not last.endswith("."):
                violations.append((path, node.lineno))

    assert violations == []
