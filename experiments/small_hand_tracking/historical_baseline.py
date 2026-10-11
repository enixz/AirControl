"""Replay exact pre-fix methods from a verified local source snapshot."""
import ast
from contextlib import contextmanager, ExitStack
import importlib
import json
from pathlib import Path
from unittest.mock import patch

from .replay import ROOT
from .run_replay import sha

METHODS = (
    ("app/services/base_hand_tracker.py", "services.base_hand_tracker", "BaseHandTracker", "_assign_smoother_keys"),
    ("app/modes/draw_mode.py", "modes.draw_mode", "DrawMode", "handle"),
    ("app/modes/mouse_mode.py", "modes.mouse_mode", "MouseMode", "handle"),
)


def extract(tree, cls_name, method_name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)


def inspect_snapshot(directory):
    directory = Path(directory).resolve()
    baseline = json.loads((directory / "baseline.json").read_text(encoding="utf-8"))
    allowed = {Path(p) for p, *_ in METHODS}
    for key, digest in baseline["source_sha256"].items():
        path = Path(key)
        if path.parts[0] != "app":
            continue
        assert sha(directory / "source" / path) == digest, f"Snapshot altered: {key}"
        if path not in allowed:
            assert sha(ROOT / path) == digest, f"Unrelated application change: {key}"
    methods = []
    for relative, module, cls, name in METHODS:
        source = directory / "source" / relative
        old = ast.parse(source.read_text(encoding="utf-8-sig"))
        new = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))
        old_method, new_method = extract(old, cls, name), extract(new, cls, name)
        # No other code difference is allowed outside the selected methods.
        old_body, new_body = old_method.body, new_method.body
        old_method.body = new_method.body = [ast.Pass()]
        assert ast.dump(old) == ast.dump(new), f"Unrelated changes outside {cls}.{name}"
        old_method.body, new_method.body = old_body, new_body
        methods.append((source, module, cls, name, old_method))
    return methods


@contextmanager
def before_fixes(directory):
    methods = inspect_snapshot(directory)
    with ExitStack() as stack:
        for source, module_name, cls_name, method_name, node in methods:
            module = importlib.import_module(module_name)
            namespace = {}
            # Keep the live module globals so the replay's source-clock binding
            # also governs historical methods, rather than capturing wall time.
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), vars(module), namespace)
            stack.enter_context(patch.object(getattr(module, cls_name), method_name, namespace[method_name]))
        yield
