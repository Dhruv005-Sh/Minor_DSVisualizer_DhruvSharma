"""
ML Models for DS Visualizer v3 — TREE & GRAPH only
====================================================
2 models used as requested:

1. KNN Regressor      — Smart Insert Position (where to optimally insert a node)
2. Linear Regression  — Complexity Predictor  (predicts traversal steps / complexity score)

All training is done in Python at import time.
JS only displays results — no ML in the frontend.
"""

from __future__ import annotations
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


# ═══════════════════════════════════════════════════════════
# MODEL 1 — KNN Regressor: Smart Insert Position
# -------------------------------------------------------
# Input:  new_value, mean, median, std, size, sorted_ratio
# Output: suggested index to insert the value
# Used for both Tree (level-order position) and array-like views.
# ═══════════════════════════════════════════════════════════
def _train_knn_insert() -> tuple[KNeighborsRegressor, StandardScaler]:
    X, y = [], []
    for _ in range(1200):
        size = np.random.randint(2, 20)
        arr  = np.random.randint(1, 99, size).tolist()
        nv   = np.random.randint(1, 99)
        # Best insertion point = position in sorted order
        best = next((i for i, v in enumerate(sorted(arr)) if v >= nv), size)
        sp   = sum(1 for i in range(len(arr) - 1) if arr[i] <= arr[i + 1])
        sr   = sp / max(len(arr) - 1, 1)
        X.append([
            float(nv),
            float(np.mean(arr)),
            float(np.median(arr)),
            float(np.std(arr) + 0.1),
            float(size),
            float(sr),
        ])
        y.append(float(best))

    sc = StandardScaler()
    Xs = sc.fit_transform(np.array(X))
    m  = KNeighborsRegressor(n_neighbors=7, weights="distance")
    m.fit(Xs, np.array(y))
    return m, sc


# ═══════════════════════════════════════════════════════════
# MODEL 2 — Linear Regression: Complexity Predictor
# -------------------------------------------------------
# Input:  structure_size, num_edges, depth/density, op_count
# Output: predicted number of steps for traversal (complexity score)
# e.g. BFS/DFS steps ≈ V + E for a graph, or height for a tree.
# ═══════════════════════════════════════════════════════════
def _train_linear_complexity() -> tuple[LinearRegression, StandardScaler]:
    X, y = [], []
    for _ in range(1500):
        n_nodes = np.random.randint(2, 25)
        max_edges = max(1, n_nodes * (n_nodes - 1) // 2)
        n_edges = np.random.randint(0, min(max_edges, 30) + 1)
        depth   = np.random.randint(1, max(2, int(np.log2(n_nodes + 1)) + 3))
        op_cnt  = np.random.randint(1, 30)
        density = n_edges / max(n_nodes, 1)
        # Theoretical: BFS/DFS visits every node + edge once
        complexity = float(n_nodes + n_edges) + float(depth) * 0.5 + float(op_cnt) * 0.3
        X.append([float(n_nodes), float(n_edges), float(depth), float(density), float(op_cnt)])
        y.append(complexity)

    sc = StandardScaler()
    Xs = sc.fit_transform(np.array(X))
    m  = LinearRegression()
    m.fit(Xs, np.array(y))
    return m, sc


# ═══════════════════════════════════════════════════════════
# TRAIN ALL MODELS AT IMPORT TIME
# ═══════════════════════════════════════════════════════════
print("[ML] Training ML models (KNN + Linear Regression)...")
_knn_model,    _knn_scaler    = _train_knn_insert()
_linreg_model, _linreg_scaler = _train_linear_complexity()
print("[ML] ML models ready!")


# ═══════════════════════════════════════════════════════════
# PUBLIC API FUNCTIONS (called by app.py)
# ═══════════════════════════════════════════════════════════

def knn_smart_insert(new_value: float, current_array: list) -> dict:
    """
    KNN Regressor — suggest best position to insert a value.
    Works for both tree-node arrays and graph node lists.
    """
    if not current_array:
        return {
            "suggested_index": 0,
            "reasoning": "Insert at position [0] — structure is empty",
            "sorted_ratio": 100.0,
            "neighbors_used": 0,
        }

    arr = np.array(current_array, dtype=float)
    sp  = sum(1 for i in range(len(arr) - 1) if arr[i] <= arr[i + 1])
    sr  = sp / max(len(arr) - 1, 1)
    feat = np.array([[
        float(new_value),
        float(np.mean(arr)),
        float(np.median(arr)),
        float(np.std(arr) + 0.1),
        float(len(arr)),
        float(sr),
    ]])
    raw = float(_knn_model.predict(_knn_scaler.transform(feat))[0])
    idx = int(round(max(0, min(raw, len(current_array)))))

    if sr > 0.7:
        reason = f"Array is {int(sr*100)}% sorted — inserting at [{idx}] maintains sorted order"
    elif float(new_value) < float(np.mean(arr)):
        reason = f"{new_value} < mean ({np.mean(arr):.1f}) — optimal position near front [{idx}]"
    else:
        reason = f"{new_value} ≥ mean ({np.mean(arr):.1f}) — optimal position near back [{idx}]"

    return {
        "suggested_index": idx,
        "reasoning":       reason,
        "sorted_ratio":    round(float(sr) * 100, 1),
        "neighbors_used":  7,
    }


def linear_complexity_predict(n_nodes: int, n_edges: int, depth: int, op_count: int) -> dict:
    """
    Linear Regression — predict algorithm traversal complexity score.
    Returns expected steps and a complexity rating.
    """
    n_nodes  = max(int(n_nodes), 0)
    n_edges  = max(int(n_edges), 0)
    depth    = max(int(depth), 1)
    op_count = max(int(op_count), 1)
    density  = n_edges / max(n_nodes, 1)

    feat = np.array([[float(n_nodes), float(n_edges), float(depth), float(density), float(op_count)]])
    pred = float(_linreg_model.predict(_linreg_scaler.transform(feat))[0])
    score = max(0.0, round(pred, 1))

    # Theoretical comparison
    theoretical = n_nodes + n_edges
    ratio = score / max(theoretical, 1)

    if ratio < 0.8:
        rating = "Very Efficient"
        color  = "emerald"
    elif ratio < 1.2:
        rating = "Efficient"
        color  = "cyan"
    elif ratio < 2.0:
        rating = "Moderate"
        color  = "amber"
    else:
        rating = "High Complexity"
        color  = "rose"

    coefs = list(_linreg_model.coef_)
    return {
        "predicted_steps": score,
        "theoretical_min": theoretical,
        "rating":          rating,
        "color":           color,
        "formula_note":    f"Steps ≈ {coefs[0]:.2f}·V + {coefs[1]:.2f}·E + {coefs[2]:.2f}·D",
        "inputs":          {"nodes": n_nodes, "edges": n_edges, "depth": depth, "ops": op_count},
    }
