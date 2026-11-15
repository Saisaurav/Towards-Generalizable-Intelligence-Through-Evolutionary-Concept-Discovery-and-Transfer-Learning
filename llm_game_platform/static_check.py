"""Static (AST) checks for generated environment source. Defence in depth only;
the real containment is the sandbox worker (separate process, limits, no
filesystem/network imports)."""

from __future__ import annotations

import ast

from .env_interface import REQUIRED_METHODS

ALLOWED_IMPORTS = {
    "math", "random", "copy", "collections", "itertools", "functools", "dataclasses",
    "enum", "typing", "heapq", "bisect", "statistics", "operator", "fractions", "abc",
}
FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "memoryview", "dir", "exit", "quit",
    "help", "classmethod_", "__builtins__",
}
MAX_SOURCE_BYTES = 60_000


def static_check(source: str) -> list[str]:
    """Return a list of human-readable problems (empty list = passed)."""
    problems: list[str] = []
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return [f"source is larger than {MAX_SOURCE_BYTES} bytes"]
    try:
        tree = ast.parse(source, filename="<generated_env>")
    except SyntaxError as error:
        return [f"SyntaxError on line {error.lineno}: {error.msg}"]

    has_spec = False
    env_class: ast.ClassDef | None = None
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"line {line}: import of '{alias.name}' is not allowed (allowed: {', '.join(sorted(ALLOWED_IMPORTS))})")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"line {line}: import from '{node.module}' is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {line}: use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__init__":
            problems.append(f"line {line}: access to dunder attribute '{node.attr}' is not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {line}: global/nonlocal statements are not allowed")
        elif isinstance(node, (ast.AsyncFunctionDef, ast.Await)):
            problems.append(f"line {line}: async code is not allowed")
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SPEC" for t in node.targets):
            has_spec = True
        if isinstance(node, ast.ClassDef) and node.name == "Environment":
            env_class = node
    if not has_spec:
        problems.append("module-level 'SPEC = {...}' assignment is missing")
    if env_class is None:
        problems.append("class 'Environment' is missing")
    else:
        defined = {n.name for n in env_class.body if isinstance(n, ast.FunctionDef)}
        for name in REQUIRED_METHODS:
            if name not in defined:
                problems.append(f"Environment.{name}() is missing")
    return problems
