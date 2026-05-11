# Day 08 Lab — LangGraph Agentic Orchestration

Build a production-style LangGraph workflow for a support-ticket agent with state management, conditional routing, retry loops, human-in-the-loop approval, persistence, and metrics.

---

## 🚀 Lab 22 Implementation Highlights

The project has been evolved from the starter skeleton into a robust agentic system with the following advanced features:

- **Semantic Routing Engine**: Uses a keyword-based heuristic with priority logic (Risky > Tool > Missing Info > Error > Simple).
- **SQLite Persistence**: Integrated `SqliteSaver` in WAL mode for thread-safe state management and recovery.
- **Human-in-the-Loop (HITL)**: Mandatory approval flow for risky actions (refunds, deletions) with UI integration.
- **Interactive Dashboard**: A custom Streamlit UI for real-time interaction, node execution tracking, and state visualization.
- **Time Travel**: Built-in state history exploration allowing you to view and audit previous execution steps.

---

## 🎮 How to Run (Current Version)

### 1. Installation
Ensure you have Python 3.11+ and install the dependencies in editable mode:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Environment Setup
Copy the `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### 3. Launch the Interactive UI (Recommended)
Experience the agent with a real-time chat interface and state auditing:
```bash
streamlit run scratch/app.py
```

### 4. Run Automated Scenario Tests
Execute all 7+ support scenarios and generate the performance metrics:
```bash
make run-scenarios
```
Results will be saved to `outputs/metrics.json`.

### 5. Generate Architecture Diagram
Visualize the LangGraph workflow:
```bash
python scratch/generate_diagram.py
```

---

---

## How you will be graded

| Category | Points | What we look for |
|---|---:|---|
| Architecture & state schema | 20 | Typed state with correct reducers, lean serializable fields, clear node boundaries |
| Graph behavior | 25 | All scenario routes correct, bounded retry loop, HITL approval path, all routes terminate |
| Persistence & recovery | 15 | Checkpointer wired, thread_id per run, state history or crash-resume evidence |
| Metrics & tests | 20 | `metrics.json` valid, scenario coverage, tests pass, meaningful counts |
| Report & demo | 15 | Architecture explanation, metrics table, failure analysis, improvement ideas |
| Production hygiene | 5 | Config, environment handling, lint/type discipline |

**Grade bands:**
- **90–100**: Production-quality graph + metrics + report + at least one bonus extension
- **75–89**: Core graph works, metrics valid, report explains trade-offs
- **60–74**: Graph mostly works but persistence/report/error handling incomplete
- **< 60**: Does not run, hard-codes scenarios, or lacks metrics/report

> **Critical rule**: Do NOT hard-code answers to specific scenario queries. Your graph must route based on **keywords and state logic**, not by matching exact scenario IDs. We grade with additional hidden scenarios that test the same routing rules but use different queries.

---

## Understanding `scenarios.jsonl`

The file `data/sample/scenarios.jsonl` contains **7 sample scenarios** your graph must handle:

```jsonl
{"id":"S01_simple",      "query":"How do I reset my password?",                          "expected_route":"simple"}
{"id":"S02_tool",        "query":"Please lookup order status for order 12345",            "expected_route":"tool"}
{"id":"S03_missing",     "query":"Can you fix it?",                                      "expected_route":"missing_info"}
{"id":"S04_risky",       "query":"Refund this customer and send confirmation email",      "expected_route":"risky"}
{"id":"S05_error",       "query":"Timeout failure while processing request",              "expected_route":"error"}
{"id":"S06_delete",      "query":"Delete customer account after support verification",    "expected_route":"risky"}
{"id":"S07_dead_letter", "query":"System failure cannot recover after multiple attempts", "expected_route":"error", "max_attempts":1}
```

### What each field means

| Field | Purpose |
|---|---|
| `id` | Unique scenario identifier — used in metrics output |
| `query` | The user's support-ticket text — input to your graph |
| `expected_route` | Which route your `classify_node` should pick: `simple`, `tool`, `missing_info`, `risky`, or `error` |
| `requires_approval` | If `true`, your graph must hit the approval/HITL node before answering |
| `should_retry` | If `true`, scenario simulates transient tool failure requiring retry |
| `max_attempts` | Override retry limit (default 3). S07 sets this to 1, so it exhausts retries immediately → dead letter |
| `tags` | Descriptive labels for your reference |

### How scenarios flow through your code

```
scenarios.jsonl  →  scenarios.py loads them  →  cli.py runs each through your graph
                                              →  metrics.py collects results
                                              →  outputs/metrics.json
```

1. `make run-scenarios` reads `data/sample/scenarios.jsonl`
2. For each scenario, it calls `initial_state(scenario)` → `graph.invoke(state)`
3. After execution, it checks: did `actual_route` match `expected_route`? Did HITL fire when required?
4. Results go to `outputs/metrics.json`

### How to design your routing logic

Your `classify_node` should use **keyword-based heuristics** to pick routes:

| Route | Trigger keywords (examples) |
|---|---|
| `risky` | refund, delete, send, cancel, remove, revoke |
| `tool` | status, order, lookup, check, track, find, search |
| `missing_info` | Very short/vague queries (e.g., < 5 words with pronouns like "it") |
| `error` | timeout, fail, error, crash, unavailable |
| `simple` | Default — anything that doesn't match above |

**Priority matters**: check risky keywords first (highest priority), then tool, then missing_info, then error, then default to simple. This prevents conflicts when a query contains keywords from multiple categories.

### Adding your own test scenarios

You can add extra lines to `scenarios.jsonl` to test edge cases:

```jsonl
{"id":"S08_custom","query":"Cancel my subscription immediately","expected_route":"risky","requires_approval":true,"tags":["custom"]}
```

This helps you verify your routing handles cases beyond the 7 samples. The grading script will also test with scenarios you haven't seen.

---

## Quick start

```bash
# Option A: conda
conda activate ai-lab
pip install -e '.[dev]'

# Option B: venv
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

# Verify setup
make test
```

`pip install -e '.[dev]'` installs this project in editable mode with dev dependencies (pytest, ruff, mypy). Editable mode means code changes take effect immediately without reinstalling.

---

## Step-by-step workflow

### Phase 1: Core graph (0–75 min) — worth 45 points

1. **`state.py`** — Confirm which fields use `Annotated[list, add]` (append-only reducer). Add `evaluation_result` field for retry loop gate.

2. **`nodes.py`** — Implement each node function. Key ones:
   - `classify_node`: keyword-based routing (see table above)
   - `evaluate_node`: check tool results for errors → set `evaluation_result` to `"needs_retry"` or `"success"`
   - `dead_letter_node`: log failures when max retries exceeded
   - `approval_node`: mock approval (return `approved=True` by default)

3. **`routing.py`** — Implement routing functions:
   - `route_after_classify`: map route string → next node name
   - `route_after_evaluate`: if `needs_retry` → `"retry"`, else → `"answer"`
   - `route_after_retry`: if `attempt < max_attempts` → back to tool, else → `"dead_letter"`

4. **`graph.py`** — Wire nodes and edges. Target architecture:

   ```
   START → intake → classify → [conditional routing]
     simple       → answer → finalize → END
     tool         → tool → evaluate → answer → finalize → END
     missing_info → clarify → finalize → END
     risky        → risky_action → approval → tool → evaluate → answer → finalize → END
     error        → retry → tool → evaluate → [retry loop or answer]
     max retry    → dead_letter → finalize → END
   ```

5. **Verify**: `make test` and `make run-scenarios`

### Phase 2: Persistence (75–120 min) — worth 15 points

6. **`persistence.py`** — Implement checkpointer factory:
   - `"memory"` → `MemorySaver()` (already works for dev)
   - `"sqlite"` → `SqliteSaver` with `sqlite3.connect()` and WAL mode
   - Show evidence: thread_id per run, state history, or crash-resume

### Phase 3: Metrics & report (120–180 min) — worth 35 points

7. **Run all scenarios**: `make run-scenarios` → generates `outputs/metrics.json`
8. **Validate**: `make grade-local` → checks metrics schema
9. **Write report**: Fill `reports/lab_report.md` — explain architecture, metrics, failures, improvements

### Phase 4: Bonus extensions (180+ min) — push toward 90+

Pick one or more:
- **Parallel fan-out**: Use `Send()` to run two tools concurrently, merge results via `add` reducer
- **Real HITL**: Set `LANGGRAPH_INTERRUPT=true`, use `interrupt()` in approval_node
- **Streamlit UI**: Build approval/reject interface with interrupt/resume
- **Time travel**: Use `get_state_history()` to replay from earlier checkpoint
- **Crash recovery**: Show SQLite checkpoint survives process kill + restart
- **Graph diagram**: Export Mermaid diagram via `graph.get_graph().draw_mermaid()`

---

## Make commands

| Command | What it does |
|---|---|
| `make install` | Install project + dev dependencies |
| `make test` | Run pytest |
| `make lint` | Run ruff linter |
| `make typecheck` | Run mypy type checker |
| `make run-scenarios` | Execute all scenarios → `outputs/metrics.json` |
| `make grade-local` | Validate metrics.json schema |
| `make clean` | Remove caches and generated files |
| `streamlit run scratch/app.py` | Launch the Streamlit chat interface |
| `python scratch/run_lab.py` | Run scenarios via simplified script |
| `python scratch/generate_diagram.py` | Generate graph diagram image |

---

## Submission checklist

- [ ] All `TODO(student)` sections completed
- [ ] `make test` passes
- [ ] `make run-scenarios` generates valid `outputs/metrics.json`
- [ ] `make grade-local` passes validation
- [ ] `reports/lab_report.md` filled in with architecture explanation, metrics analysis, and improvement ideas
- [ ] Can explain at least one route and one failure mode during demo

**For 90+ points, also include:**
- [ ] At least one bonus extension (persistence, parallel fan-out, HITL, time travel, diagram)
- [ ] Evidence of extension in report (screenshot, log output, or diagram)

---

## Common pitfalls

1. **Keyword conflicts**: "Check order status" contains both "check" (tool) and "order" (tool). Test priority carefully — risky keywords should take precedence over tool keywords.

2. **Word boundary matching**: "Can you fix it?" — match "it" as a whole word, not as substring of "item" or "iteration". Strip punctuation before checking.

3. **Unbounded retry**: Always check `attempt < max_attempts`. Without this bound, error scenarios loop forever.

4. **SqliteSaver API**: In `langgraph-checkpoint-sqlite` 3.x, use `SqliteSaver(conn=sqlite3.connect(...))` not `SqliteSaver.from_conn_string()` (returns context manager, not checkpointer).

5. **Forgetting finalize**: Every route must end at `finalize → END`. Missing this means the graph never terminates for some scenarios.
