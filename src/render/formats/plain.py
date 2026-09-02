"""Words only: one paragraph per turn, nothing else. For grep, diff, and piping."""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List

from render import RenderParams, Turn


def render(turns: List[Turn], stamp: Dict[str, Any], params: RenderParams) -> str:
    paras = []
    for t in turns:
        text = " ".join(w.text for w in t.words)
        paras.append(textwrap.fill(text, width=params.width, break_on_hyphens=False, break_long_words=False))
    return "\n\n".join(paras) + "\n"
