# Opytimizer conventions

These conventions adopt the cpmux/phitrain code-style rules without importing cpmux's application architecture.
Code defines behavior. Preserve supported numerical behavior, public APIs, and useful scientific references.

## Project adaptations

- Opytimizer remains Apache-2.0 licensed. Do not copy cpmux's MIT notice or its ownership statement.
- Use the requested modern syntax while preserving the declared Python 3.11 support floor.
  Union syntax and builtin generics work on both Python 3.11 and Python 3.12+.
- Absolute local imports use `opytimizer`, not `cpmux`. Existing public package exports remain supported.
- This is a numerical library, not a Rich/Typer/Textual application. Examples may print intentional results.
- The library currently has no diagnostic logging calls.
  Do not introduce an unused logging wrapper or a cpmux dependency.
  If shared diagnostics become necessary, use a project-local `get_logger(__name__)`.
  Leave handler setup to callers.
- Test functions remain plain pytest functions without docstrings or annotations.
  Their assertions check behavior and are not runtime input validation.

## Headers and imports

Every Python file starts with:

```python
# Copyright (c) 2019-2026 Opytimizer contributors.
# Licensed under the Apache License, Version 2.0.
```

Use top-level absolute imports, grouped as standard library, third-party, then local imports.
Separate these groups with a blank line. Use builtin generics and `X | None`, never `Optional[X]`.
Import ABCs such as `Callable`, `Iterable`, `Mapping`, and `Sequence` from `collections.abc`.
Keep typing-specific features such as `Any`, `Literal`, `Annotated`, and `SupportsIndex` in `typing`. (R2)

## Docstrings

Public functions, regular classes, and explicit public constructors use Google-style docstrings.
Keep a single-sentence summary. A regular class has only its one-line summary and documents constructor
arguments on `__init__`. Use one line for each `Args`, `Returns`, or `Raises` entry.
Do not put semicolons or `defaults to <value>` tails in those entries. (R3, R13)

Keep one blank line before the closing triple quotes and one blank line after them before code or fields:

```python
def example(value: int) -> int:
    """Return the supplied value.

    Args:
        value: Value to return.

    Returns:
        The unchanged value.

    """

    return value
```

Private helpers have no docstrings. Concrete overrides dispatched by the optimizer, space, or callback
lifecycle also have none. Public base hook declarations retain their contracts.
Public callable protocols such as objective `__call__` methods remain documented.

Do not discard useful documentation to satisfy placement rules. Keep lifecycle and ownership details in
public I/O prose or guides. Keep configuration details and scientific references in constructor notes or
module documentation when a regular class is reduced to its summary.

Pydantic models and dataclasses without an explicit constructor document every field in an `Attributes`
section, with one `name: what it holds.` line per field. Do not convert existing mutable array containers
to dataclasses merely to use that format.

## Errors and diagnostics

Validate input with `if` and a specific raised exception, never a runtime `assert`.
Bare `except:` is forbidden.
Raised messages identify a backticked offender and end with a period:

```python
raise ValueError(f"`n_agents` must be positive, but got {n_agents}.")
```

Use `is None` and `is True` prose where relevant. Preserve the original exception as a cause when adding
context to an expected failure. Do not add broad catches or success-shaped fallbacks. (R1)

Library code never uses `print`. Diagnostic warnings and errors follow the same offender/period form,
such as ``f"`name=value` cannot be used."``. Informational and debug messages remain plain.
Do not recreate noisy lifecycle logging or configure application console handlers in library modules. (R14)

## Readability and reuse

- Comments explain why, not what. Prefer none or a single line, with a three-line maximum.
  Do not use banner separators or trailing periods. Copyright/license notices and tool pragmas are exceptions. (R8)
- Preserve useful equation and publication references when removing narration or relocating docstrings.
- In functions of at least twelve lines, use a single blank line at each logical phase transition.
  Do not separate every assignment mechanically or compress unrelated phases together. (R11)
- Inline first. Introduce a helper, constant, or parameter only when a second real call site justifies it.
  Keep framework hooks and meaningful existing numerical phases rather than flattening algorithms into monoliths. (R16)
- Use double-quoted strings. Keep readable prose within 120 characters. (R9)

## Tests and tools

Tests mirror the package structure. Name tests after the function/class and the observed behavior.
Use plain test functions and bare assertions without appended failure-message strings. (R15)

Use the existing Black, isort, Flake8, pytest, and Sphinx toolchain.
Black/isort/Flake8 use a 120-character line length. Python 3.11 parsing remains a compatibility gate.
Documentation uses Napoleon and executable Sphinx examples. No additional runtime dependency is required.

Run formatting through `uv run pre-commit run --all-files`.
The Black hook restores the prescribed docstring padding after Black formats code, without suppressing files or regions.
Bare Black collapses single-sentence docstrings and therefore is not the complete project formatting command.
