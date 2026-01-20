"""
Connected components clustering on the similarity graph.
"""
from __future__ import annotations

from typing import Dict, List, Sequence


def connected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    visited = set()
    comps: List[List[int]] = []

    for node in graph:
        if node in visited:
            continue
        stack = [node]
        comp = []
        visited.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nei in graph.get(cur, []):
                if nei not in visited:
                    visited.add(nei)
                    stack.append(nei)
        comps.append(comp)
    return comps
