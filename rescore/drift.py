"""Field-by-field comparison of a re-scored result against the one the study committed.

The committed results in the research repository predate later changes to the scoring code,
so a re-score is expected to differ. This module says how: a last-digit float difference
from a different platform or library build (`float_noise`), a value that really moved
(`changed`), a field that exists on one side only (`added`, `removed`), or anything under
`manual_review`, whose row samples are drawn afresh and are not a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

KINDS = ("float_noise", "changed", "added", "removed", "manual_review")
MANUAL_REVIEW = "/manual_review"


@dataclass(frozen=True)
class Diff:
    path: str
    kind: str
    committed: object
    rescored: object


def compare(committed: dict, rescored: dict, rel_tol: float = 1e-12,
            abs_tol: float = 1e-15) -> list[Diff]:
    """Returns every difference, depth first, keys in sorted order.

    A list whose length differs is one diff at the list's path, carrying the two lengths.
    """
    out: list[Diff] = []
    _walk(committed, rescored, "", out, rel_tol, abs_tol)
    return out


def _is_manual_review(path: str) -> bool:
    return path == MANUAL_REVIEW or path.startswith((MANUAL_REVIEW + "/", MANUAL_REVIEW + "["))


def _kind(path: str, kind: str) -> str:
    return "manual_review" if _is_manual_review(path) else kind


def _is_number(x: object) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def _walk(a, b, path: str, out: list[Diff], rel_tol: float, abs_tol: float) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(a.keys() | b.keys()):
            sub = f"{path}/{key}"
            if key not in b:
                out.append(Diff(sub, _kind(sub, "removed"), a[key], None))
            elif key not in a:
                out.append(Diff(sub, _kind(sub, "added"), None, b[key]))
            else:
                _walk(a[key], b[key], sub, out, rel_tol, abs_tol)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(Diff(path, _kind(path, "changed"), len(a), len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _walk(x, y, f"{path}[{i}]", out, rel_tol, abs_tol)
        return
    if _is_number(a) and _is_number(b):
        if a == b or (math.isnan(a) and math.isnan(b)):
            return
        close = math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        out.append(Diff(path, _kind(path, "float_noise" if close else "changed"), a, b))
        return
    if type(a) is not type(b) or a != b:
        out.append(Diff(path, _kind(path, "changed"), a, b))
