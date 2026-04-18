"""
Binary Tree — level-order insertion, max 2 children.
Algorithms: BFS, DFS, Inorder/Preorder/Postorder Traversal.
All logic in Python; JS only visualizes.
"""
from __future__ import annotations
from collections import deque
from typing import Optional


class _Node:
    def __init__(self, val: int):
        self.val   = int(val)
        self.left:  Optional[_Node] = None
        self.right: Optional[_Node] = None

    def to_dict(self) -> dict:
        return {
            "val":   self.val,
            "left":  self.left.to_dict()  if self.left  else None,
            "right": self.right.to_dict() if self.right else None,
        }


class BinaryTree:
    def __init__(self):
        self.root: Optional[_Node] = None

    # ── Insert (level-order so max 2 children always) ────
    def insert(self, val: int) -> dict:
        node = _Node(val)
        if not self.root:
            self.root = node
            return {"ok": True, "op": "insert", "value": int(val), "state": self._state()}
        q: deque[_Node] = deque([self.root])
        while q:
            n = q.popleft()
            if not n.left:  n.left  = node; break
            else:            q.append(n.left)
            if not n.right: n.right = node; break
            else:            q.append(n.right)
        return {"ok": True, "op": "insert", "value": int(val), "state": self._state()}

    # ── BFS traversal — returns step-by-step frames for JS animation ──
    def bfs(self) -> dict:
        if not self.root:
            return {"ok": False, "error": "Tree is empty", "frames": [], "order": []}
        visited: list[int] = []
        frames:  list[list[int]] = []
        q: deque[_Node] = deque([self.root])
        while q:
            node = q.popleft()
            visited.append(node.val)
            frames.append(list(visited))   # snapshot of visited so far
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        return {
            "ok":     True,
            "op":     "bfs",
            "order":  visited,
            "frames": frames,
            "state":  self._state(),
        }

    # ── DFS traversal (iterative, pre-order) — returns frames ────────
    def dfs(self) -> dict:
        if not self.root:
            return {"ok": False, "error": "Tree is empty", "frames": [], "order": []}
        visited: list[int] = []
        frames:  list[list[int]] = []
        stack: list[_Node] = [self.root]
        while stack:
            node = stack.pop()
            visited.append(node.val)
            frames.append(list(visited))   # snapshot
            # Push right first so left is processed first (pre-order)
            if node.right: stack.append(node.right)
            if node.left:  stack.append(node.left)
        return {
            "ok":     True,
            "op":     "dfs",
            "order":  visited,
            "frames": frames,
            "state":  self._state(),
        }

    # ── Tree Traversals — inorder / preorder / postorder  ───────────
    def traverse(self, mode: str = "inorder") -> dict:
        """mode: 'inorder' | 'preorder' | 'postorder'"""
        order: list[int] = []
        frames: list[list[int]] = []

        def _inorder(n: Optional[_Node]) -> None:
            if not n: return
            _inorder(n.left)
            order.append(n.val); frames.append(list(order))
            _inorder(n.right)

        def _preorder(n: Optional[_Node]) -> None:
            if not n: return
            order.append(n.val); frames.append(list(order))
            _preorder(n.left)
            _preorder(n.right)

        def _postorder(n: Optional[_Node]) -> None:
            if not n: return
            _postorder(n.left)
            _postorder(n.right)
            order.append(n.val); frames.append(list(order))

        if not self.root:
            return {"ok": False, "error": "Tree is empty", "frames": [], "order": [], "mode": mode}

        if mode == "preorder":
            _preorder(self.root)
        elif mode == "postorder":
            _postorder(self.root)
        else:
            _inorder(self.root)

        return {
            "ok":     True,
            "op":     "traverse",
            "mode":   mode,
            "order":  order,
            "frames": frames,
            "state":  self._state(),
        }

    # ── BFS-based search — returns path list for JS animation ──
    def search(self, val: int) -> dict:
        if not self.root:
            return {"ok": False, "error": "Tree is empty", "found": False, "path": []}
        q: deque = deque([(self.root, [])])
        while q:
            node, path = q.popleft()
            p = path + [node.val]
            if node.val == int(val):
                return {"ok": True, "op": "search", "found": True,
                        "value": int(val), "path": p, "state": self._state()}
            if node.left:  q.append((node.left,  p))
            if node.right: q.append((node.right, p))
        return {"ok": True, "op": "search", "found": False,
                "value": int(val), "path": [], "state": self._state()}

    # ── Remove deepest-rightmost node ────────────────────
    def remove_last(self) -> dict:
        if not self.root:
            return {"ok": False, "error": "Tree is empty"}
        if not self.root.left and not self.root.right:
            removed, self.root = self.root.val, None
            return {"ok": True, "op": "remove_last", "removed": removed, "state": self._state()}
        q: deque[_Node]       = deque([self.root])
        last_parent: Optional[_Node] = None
        last_node:   Optional[_Node] = None
        is_left = False
        while q:
            n = q.popleft()
            if n.left:  last_parent, last_node, is_left = n, n.left,  True;  q.append(n.left)
            if n.right: last_parent, last_node, is_left = n, n.right, False; q.append(n.right)
        removed = last_node.val if last_node else None
        if last_parent:
            if is_left: last_parent.left  = None
            else:       last_parent.right = None
        return {"ok": True, "op": "remove_last", "removed": removed, "state": self._state()}

    def clear(self) -> dict:
        self.root = None
        return {"ok": True, "op": "clear", "state": self._state()}

    def _size(self, n: Optional[_Node]) -> int:
        return 0 if not n else 1 + self._size(n.left) + self._size(n.right)

    def _state(self) -> dict:
        return {"tree": self.root.to_dict() if self.root else None,
                "size": self._size(self.root)}
