"""
DS Visualizer v3 — Flask API Server
======================================
Data Structures : Tree, Graph
Algorithms      : BFS, DFS, Traversal (Tree) | BFS, DFS, Kruskal (Graph)
ML Models       : KNN Regressor, Linear Regression

Run:  pip install flask numpy scikit-learn
      python app.py  →  http://localhost:5001
"""
from flask import Flask, request, jsonify, send_from_directory
from ds import BinaryTree, Graph
from ml import knn_smart_insert, linear_complexity_predict
import os

app = Flask(__name__)
FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")


# ── CORS ─────────────────────────────────────────────────
@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ── Serve frontend ────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")

@app.route("/<path:p>")
def assets(p):
    return send_from_directory(FRONTEND, p)


# ── Helpers ──────────────────────────────────────────────
def B() -> dict:
    return request.get_json(force=True, silent=True) or {}

def OPT():
    return jsonify({}), 200


# ═══════════════════════════════════════════════
# DS SINGLETONS  (one instance per server process)
# ═══════════════════════════════════════════════
_tree  = BinaryTree()
_graph = Graph()

# Pre-populate tree with some values
for v in [10, 6, 14, 4, 8, 12, 16]:
    _tree.insert(v)


# ═══════════════════════════════════════════════
# TREE
# ═══════════════════════════════════════════════
@app.route("/ds/tree/insert",      methods=["POST", "OPTIONS"])
def tree_insert():
    if request.method == "OPTIONS": return OPT()
    d = B()
    if "value" not in d: return jsonify({"ok": False, "error": "Missing value"}), 400
    return jsonify(_tree.insert(d["value"]))

@app.route("/ds/tree/search",      methods=["POST", "OPTIONS"])
def tree_search():
    if request.method == "OPTIONS": return OPT()
    d = B()
    if "value" not in d: return jsonify({"ok": False, "error": "Missing value"}), 400
    return jsonify(_tree.search(d["value"]))

@app.route("/ds/tree/remove_last", methods=["POST", "OPTIONS"])
def tree_remove_last():
    if request.method == "OPTIONS": return OPT()
    return jsonify(_tree.remove_last())

@app.route("/ds/tree/clear",       methods=["POST", "OPTIONS"])
def tree_clear():
    if request.method == "OPTIONS": return OPT()
    return jsonify(_tree.clear())

@app.route("/ds/tree/state",       methods=["GET"])
def tree_state():
    return jsonify({"ok": True, **_tree._state()})

# ── Tree Algorithm Endpoints ────────────────────────────

@app.route("/ds/tree/bfs",         methods=["POST", "OPTIONS"])
def tree_bfs():
    if request.method == "OPTIONS": return OPT()
    return jsonify(_tree.bfs())

@app.route("/ds/tree/dfs",         methods=["POST", "OPTIONS"])
def tree_dfs():
    if request.method == "OPTIONS": return OPT()
    return jsonify(_tree.dfs())

@app.route("/ds/tree/traverse",    methods=["POST", "OPTIONS"])
def tree_traverse():
    if request.method == "OPTIONS": return OPT()
    d    = B()
    mode = d.get("mode", "inorder")   # inorder | preorder | postorder
    return jsonify(_tree.traverse(mode))


# ═══════════════════════════════════════════════
# GRAPH  (number nodes, weighted edges)
# ═══════════════════════════════════════════════
@app.route("/ds/graph/add_node",    methods=["POST", "OPTIONS"])
def graph_add_node():
    if request.method == "OPTIONS": return OPT()
    d   = B()
    val = d.get("value")
    return jsonify(_graph.add_node(int(val) if val is not None else None))

@app.route("/ds/graph/remove_node", methods=["POST", "OPTIONS"])
def graph_remove_node():
    if request.method == "OPTIONS": return OPT()
    d   = B()
    val = d.get("value")
    if val is None: return jsonify({"ok": False, "error": "Missing value"}), 400
    return jsonify(_graph.remove_node(int(val)))

@app.route("/ds/graph/add_edge",    methods=["POST", "OPTIONS"])
def graph_add_edge():
    if request.method == "OPTIONS": return OPT()
    d = B()
    a, b = d.get("node_a"), d.get("node_b")
    if a is None or b is None:
        return jsonify({"ok": False, "error": "Missing node_a or node_b"}), 400
    weight = d.get("weight")   # optional — auto-generated if absent
    return jsonify(_graph.add_edge(int(a), int(b), int(weight) if weight is not None else None))

@app.route("/ds/graph/remove_edge", methods=["POST", "OPTIONS"])
def graph_remove_edge():
    if request.method == "OPTIONS": return OPT()
    d = B()
    a, b = d.get("node_a"), d.get("node_b")
    if a is None or b is None:
        return jsonify({"ok": False, "error": "Missing node_a or node_b"}), 400
    return jsonify(_graph.remove_edge(int(a), int(b)))

@app.route("/ds/graph/clear",       methods=["POST", "OPTIONS"])
def graph_clear():
    if request.method == "OPTIONS": return OPT()
    return jsonify(_graph.clear())

@app.route("/ds/graph/state",       methods=["GET"])
def graph_state():
    return jsonify({"ok": True, **_graph._state()})

# ── Graph Algorithm Endpoints ────────────────────────────

@app.route("/ds/graph/bfs",         methods=["POST", "OPTIONS"])
def graph_bfs():
    if request.method == "OPTIONS": return OPT()
    d     = B()
    start = d.get("start")
    return jsonify(_graph.bfs(int(start) if start is not None else None))

@app.route("/ds/graph/dfs",         methods=["POST", "OPTIONS"])
def graph_dfs():
    if request.method == "OPTIONS": return OPT()
    d     = B()
    start = d.get("start")
    return jsonify(_graph.dfs(int(start) if start is not None else None))

@app.route("/ds/graph/kruskal",     methods=["POST", "OPTIONS"])
def graph_kruskal():
    if request.method == "OPTIONS": return OPT()
    return jsonify(_graph.kruskal())


# ═══════════════════════════════════════════════
# ML ENDPOINTS
# ═══════════════════════════════════════════════

@app.route("/ml/knn-insert",        methods=["POST", "OPTIONS"])
def ml_knn_insert():
    if request.method == "OPTIONS": return OPT()
    d = B()
    return jsonify({
        "ok": True,
        **knn_smart_insert(
            new_value     = float(d.get("value", 50)),
            current_array = [float(x) for x in d.get("array", [])],
        )
    })

@app.route("/ml/linear-complexity", methods=["POST", "OPTIONS"])
def ml_linear_complexity():
    if request.method == "OPTIONS": return OPT()
    d = B()
    return jsonify({
        "ok": True,
        **linear_complexity_predict(
            n_nodes  = int(d.get("n_nodes", 0)),
            n_edges  = int(d.get("n_edges", 0)),
            depth    = int(d.get("depth",   1)),
            op_count = int(d.get("op_count", 1)),
        )
    })


# ═══════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════
@app.route("/ds/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "ds":  {"tree": _tree._state(), "graph": _graph._state()},
        "ml":  ["KNN Regressor", "Linear Regression"],
    })


if __name__ == "__main__":
    print("\n[DS Visualizer v3]  ->  http://localhost:5001")
    print("   DS   : /ds/tree  | /ds/graph")
    print("   Algo : BFS | DFS | Traversal(Tree) | Kruskal(Graph)")
    print("   ML   : KNN Regressor | Linear Regression\n")
    app.run(debug=False, port=5001)
