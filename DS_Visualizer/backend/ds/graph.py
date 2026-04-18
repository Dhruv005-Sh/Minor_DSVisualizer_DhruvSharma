"""
Graph — undirected, weighted.
Nodes are INTEGERS only (1, 2, 3 …).
Edges have weights for Kruskal's algorithm.
Algorithms: BFS, DFS, Kruskal MST.
All logic in Python; JS only visualizes.
"""
from __future__ import annotations
from collections import deque
from typing import Optional
import random


class Graph:
    def __init__(self):
        # Nodes are integers. Start with 5 default nodes.
        self._nodes: list[int]                   = [1, 2, 3, 4, 5]
        # Edges: (node_a, node_b, weight)
        self._edges: list[tuple[int, int, int]]  = [
            (1, 2, random.randint(1, 20)),
            (1, 3, random.randint(1, 20)),
            (2, 4, random.randint(1, 20)),
            (3, 4, random.randint(1, 20)),
            (4, 5, random.randint(1, 20)),
        ]
        self._next_id: int = 6   # auto-increment for new nodes

    # ── Add node (auto-numbered integer) ─────────────────
    def add_node(self, value: Optional[int] = None) -> dict:
        if value is not None:
            nid = int(value)
        else:
            nid = self._next_id
        if nid in self._nodes:
            return {"ok": False, "error": f"Node {nid} already exists"}
        self._nodes.append(nid)
        self._next_id = max(self._next_id, nid + 1)
        return {"ok": True, "op": "add_node", "node": nid, "state": self._state()}

    # ── Remove node + all connected edges ────────────────
    def remove_node(self, value: int) -> dict:
        nid = int(value)
        if nid not in self._nodes:
            return {"ok": False, "error": f"Node {nid} not found"}
        self._nodes = [n for n in self._nodes if n != nid]
        self._edges = [(a, b, w) for a, b, w in self._edges if a != nid and b != nid]
        return {"ok": True, "op": "remove_node", "node": nid, "state": self._state()}

    # ── Add edge between two integer node IDs with weight ─────────────
    def add_edge(self, node_a: int, node_b: int, weight: Optional[int] = None) -> dict:
        a, b = int(node_a), int(node_b)
        if a not in self._nodes:
            return {"ok": False, "error": f"Node {a} does not exist"}
        if b not in self._nodes:
            return {"ok": False, "error": f"Node {b} does not exist"}
        if a == b:
            return {"ok": False, "error": "Self-loops are not allowed"}
        if any((x == a and y == b) or (x == b and y == a) for x, y, w in self._edges):
            return {"ok": False, "error": f"Edge {a}—{b} already exists"}
        w = int(weight) if weight is not None else random.randint(1, 20)
        self._edges.append((a, b, w))
        return {"ok": True, "op": "add_edge", "edge": [a, b, w], "state": self._state()}

    # ── Remove edge ───────────────────────────────────────
    def remove_edge(self, node_a: int, node_b: int) -> dict:
        a, b   = int(node_a), int(node_b)
        before = len(self._edges)
        self._edges = [
            (x, y, w) for x, y, w in self._edges
            if not ((x == a and y == b) or (x == b and y == a))
        ]
        if len(self._edges) == before:
            return {"ok": False, "error": f"Edge {a}—{b} not found"}
        return {"ok": True, "op": "remove_edge", "edge": [a, b], "state": self._state()}

    # ── BFS — returns frames list for JS animation ────────
    def bfs(self, start: Optional[int] = None) -> dict:
        if not self._nodes:
            return {"ok": False, "error": "Graph is empty"}
        sid = int(start) if start is not None else self._nodes[0]
        if sid not in self._nodes:
            return {"ok": False, "error": f"Start node {sid} not found"}

        adj: dict[int, list[int]] = {n: [] for n in self._nodes}
        for a, b, w in self._edges:
            adj[a].append(b)
            adj[b].append(a)

        visited: list[int]       = []
        frames:  list[list[int]] = []
        seen:    set[int]        = {sid}
        q:       deque[int]      = deque([sid])

        while q:
            node = q.popleft()
            visited.append(node)
            frames.append(list(visited))
            for nb in sorted(adj[node]):
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)

        return {
            "ok":            True,
            "op":            "bfs",
            "start":         sid,
            "visited_order": visited,
            "frames":        frames,
            "state":         self._state(),
        }

    # ── DFS — iterative, returns frames list for JS animation ────────
    def dfs(self, start: Optional[int] = None) -> dict:
        if not self._nodes:
            return {"ok": False, "error": "Graph is empty"}
        sid = int(start) if start is not None else self._nodes[0]
        if sid not in self._nodes:
            return {"ok": False, "error": f"Start node {sid} not found"}

        adj: dict[int, list[int]] = {n: [] for n in self._nodes}
        for a, b, w in self._edges:
            adj[a].append(b)
            adj[b].append(a)

        visited: list[int]       = []
        frames:  list[list[int]] = []
        seen:    set[int]        = set()
        stack:   list[int]       = [sid]

        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            visited.append(node)
            frames.append(list(visited))
            for nb in sorted(adj[node], reverse=True):   # reverse so smallest neighbour is visited first
                if nb not in seen:
                    stack.append(nb)

        return {
            "ok":            True,
            "op":            "dfs",
            "start":         sid,
            "visited_order": visited,
            "frames":        frames,
            "state":         self._state(),
        }

    # ── Kruskal's MST — returns ordered edge list for JS animation ────
    def kruskal(self) -> dict:
        if len(self._nodes) < 2:
            return {"ok": False, "error": "Need at least 2 nodes for Kruskal"}

        # Union-Find (Disjoint Set Union)
        parent = {n: n for n in self._nodes}
        rank   = {n: 0 for n in self._nodes}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]   # path compression
                x = parent[x]
            return x

        def union(x: int, y: int) -> bool:
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True

        sorted_edges = sorted(self._edges, key=lambda e: e[2])
        mst_edges:  list[list] = []
        frames:     list[list] = []   # each frame adds one MST edge

        for a, b, w in sorted_edges:
            if union(a, b):
                mst_edges.append([a, b, w])
                frames.append([list(e) for e in mst_edges])   # growing MST snapshot

        total_weight = sum(e[2] for e in mst_edges)
        is_spanning  = len(mst_edges) == len(self._nodes) - 1

        return {
            "ok":           True,
            "op":           "kruskal",
            "mst_edges":    mst_edges,
            "frames":       frames,
            "total_weight": total_weight,
            "is_spanning":  is_spanning,
            "state":        self._state(),
        }

    # ── Clear — fully resets nodes, edges, counter ───────
    def clear(self) -> dict:
        self._nodes   = []
        self._edges   = []
        self._next_id = 1
        return {"ok": True, "op": "clear", "state": self._state()}

    def _state(self) -> dict:
        return {
            "nodes":   list(self._nodes),
            "edges":   [[a, b, w] for a, b, w in self._edges],
            "size":    len(self._nodes),
            "next_id": self._next_id,
        }
