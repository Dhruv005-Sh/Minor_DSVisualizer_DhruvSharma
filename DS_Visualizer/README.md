# DS Visualizer v3
### Python Logic + ML · JavaScript Visualization

---

## What Changed in v3

1. **Graph uses numbers only** — nodes are integers (1, 2, 3…), auto-incremented by Python
2. **Fixed Clear & Remove Node** — Graph.clear() fully resets nodes, edges and counter; remove_node() correctly removes all connected edges
3. **4 ML models added** — all trained and running in Python, results sent to JS for display
4. **Strict separation** — zero data structure logic or ML in JavaScript
5. **New folder** — `ds-visualizer-v3/`

---

## Folder Structure

```
ds-visualizer-v3/
│
├── backend/
│   ├── app.py                  ← Flask routes only (DS + ML endpoints)
│   ├── requirements.txt
│   ├── ds/
│   │   ├── __init__.py
│   │   ├── stack.py            ← Stack class (Python)
│   │   ├── queue.py            ← Queue class (Python)
│   │   ├── tree.py             ← BinaryTree class (Python)
│   │   └── graph.py            ← Graph class, integer nodes (Python)
│   └── ml/
│       ├── __init__.py
│       └── models.py           ← 4 ML models (Python/scikit-learn)
│
└── frontend/
    ├── index.html              ← HTML shell with ML panels
    ├── css/style.css           ← All styles
    └── js/main.js              ← Canvas drawing + API calls only
```

---

## Architecture

```
JS (browser)             Python (backend)
─────────────────        ────────────────────────────
User clicks op     ───► DS operation (ds/*.py)
                   ◄─── New state JSON

JS draws state     ───► 4 ML endpoints (/ml/*)
                   ◄─── ML results JSON

JS renders ML results   (no ML logic in JS)
```

---

## ML Models

| Panel | Model | What it detects |
|---|---|---|
| Anomaly Detection | Isolation Forest | Is the inserted value unusual? |
| Next Op Predictor | Naive Bayes | What operation comes next? |
| Smart Insert | KNN Regressor | Best index to insert at |
| Algorithm Pattern | Random Forest | What algorithm are you doing? |

---

## Graph — Numbers Only

- Nodes are **integers** (1, 2, 3 …), Python auto-assigns IDs
- Click empty canvas → Python adds next integer node
- Click node → node → Python adds the edge
- Drag nodes to reposition (visual only, no Python call)
- **Remove Node**: type the number in the input and click Remove Node
- **Clear**: fully resets all nodes, edges, and the counter

---

## Running

```bash
cd backend
pip install -r requirements.txt
python app.py
# → http://localhost:5001
```

Then open `frontend/index.html` in your browser,
or visit `http://localhost:5001` (Flask serves it).

---

## API Reference

### DS Endpoints
```
POST /ds/stack/push        {"value": 42}
POST /ds/stack/pop
GET  /ds/stack/peek
POST /ds/stack/clear
GET  /ds/stack/state

POST /ds/queue/enqueue     {"value": 42}
POST /ds/queue/dequeue
GET  /ds/queue/peek
POST /ds/queue/clear
GET  /ds/queue/state

POST /ds/tree/insert       {"value": 42}
POST /ds/tree/search       {"value": 42}
POST /ds/tree/remove_last
POST /ds/tree/clear
GET  /ds/tree/state

POST /ds/graph/add_node              ← auto-assigns integer ID
POST /ds/graph/remove_node  {"value": 3}
POST /ds/graph/add_edge     {"node_a": 1, "node_b": 4}
POST /ds/graph/remove_edge  {"node_a": 1, "node_b": 4}
POST /ds/graph/bfs          {"start": 1}
POST /ds/graph/clear
GET  /ds/graph/state
```

### ML Endpoints
```
POST /ml/anomaly       {"value": 95, "current": [10,20,30]}
POST /ml/next-op       {"history": ["push","pop","push"]}
POST /ml/smart-insert  {"value": 45, "array": [10,30,70]}
POST /ml/pattern       {"op_counts": {"push":5,"pop":3}, "size": 4, "unique_ops": 2}
```

### Health
```
GET /ds/health
```
