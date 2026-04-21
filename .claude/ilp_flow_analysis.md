# ILP Solution Structure & Flow Bug Analysis

## Solution Structure

The ILP models a coverage path as an **Eulerian-style flow** over a directed graph of vertices (horizontal/vertical segments). Each vertex `u` has two neighbour sets:

| Set | Meaning |
|-----|---------|
| `minus_sets[u]` | Neighbours that enter `u` from the "minus" side (e.g. same-row left-neighbour) |
| `plus_sets[u]` | Neighbours that enter `u` from the "plus" side (e.g. same-row right-neighbour) |

Flow conservation is enforced **per parity** at each node:
- `(incoming from minus_sets) − (outgoing to plus_sets) = targets_plus[v]`  ← terminates via plus
- `(incoming from plus_sets) − (outgoing to minus_sets) = targets_minus[v]`  ← terminates via minus

One unit of flow departs the `source` node and terminates at exactly one `targets_minus` or `targets_plus` sink. Arc variables are **`GRB_INTEGER`** (≥ 0), so flow > 1 is allowed on a single arc.

### Solution reconstruction (`retrieve_solution`)

1. **`find_source_to_target`** — greedy walk consuming one unit of flow from `initial_position` to the terminal node. Records the vertex sequence of the main path.
2. **`retrieve_solution` loop** — iterates main-path nodes back-to-front; at each node calls `find_cycle` to harvest remaining (>0.5) flow as attached cycles, then splices those cycles into the path.
3. **`find_cycle`** — from `current_node` with direction `curr_minus`, repeatedly consumes one arc unit and appends the destination. After the primary chain, recursively searches each visited node for further sub-cycles.
4. **`set_solution`** — converts the flat integer vertex list into `segment` objects by merging consecutive same-direction vertices, then calls `coverage.reset`.

---

## Primary Bug — `add_subtour_elimination_constraint`: last subtour node is never processed

### Location
`include/models/ilp.hpp`, lines 177–185 (the `for (int node : subtour)` loop).

### What the loop does
```cpp
int prev = root;
bool curr_minus = root_minus;
for (int node : subtour){           // iterates n1 … n_{k-1}
    process_node(prev, curr_minus);
    curr_minus = ...;               // update direction
    prev = node;
}
// ← prev == n_k here, but process_node(n_k, ...) is NEVER called
```

The loop calls `process_node` for `{root, n1, ..., n_{k-1}}` but **skips the last subtour node `n_k`**.

`process_node(from, curr_minus)` does two things:
1. For arcs `(from → w)` where `w ∈ S`: adds the arc to `candidate_constraints` (arcs that must be forced to 0 if the subtour is isolated).
2. For arcs `(from → w)` where `w ∉ S`: adds the arc to `parity_outgoing_arcs_sum` (evidence the subtour is connected to the rest of the solution).

Because `n_k` is never processed:
- **Its outgoing arcs to nodes inside S** are never added as `candidate_constraints` → those arcs are unconstrained and may remain non-zero in a subtour.
- **The closing arc `n_k → root`** (which exits S, since `root ∉ S`) is never added to `parity_outgoing_arcs_sum` → the "evidence of connectivity" count is under-reported.

### Consequence
The lazy constraint:
```
M × parity_outgoing_arcs_sum − candidate_arc ≥ 0
```
is generated without considering `n_k`'s arcs. When the subtour's only non-trivial exit is through `n_k`, or when `n_k` has internal arcs, the relevant constraints are simply absent. Some subtours survive in the MIP solution.

### Why this causes coverage shortfall
Surviving subtours contribute coverage in the LP's coverage constraints (the coverage constraint sums flow over ALL arcs, including subtour arcs). When `retrieve_solution` reconstructs the path, it only calls `find_cycle` on nodes that appear in the **main path** produced by `find_source_to_target`. A subtour that shares no node with the main path is never harvested — its coverage contribution disappears from the solution.

### Fix
Add one call to `process_node` for the last subtour node after the loop:

```cpp
int prev = root;
bool curr_minus = root_minus;
for (int node : subtour){
    process_node(prev, curr_minus);
    curr_minus = std::find(math_program.minus_sets[node].V.begin(),
                           math_program.minus_sets[node].V.end(), prev)
               != math_program.minus_sets[node].V.end();
    prev = node;
}
// FIX: process the last subtour node
process_node(prev, curr_minus);
```

---

## Secondary Issue — `find_cycle`: direction for first-node recursive sub-cycle

### Location
`src/models/ilp.cpp`, lines 346–348.

```cpp
auto node = path.begin();           // = n_1, first visited node
curr_minus = std::find(minus_sets[*node].V.begin(),
                       minus_sets[*node].V.end(), path.back())  // path.back() = last visited
             != minus_sets[*node].V.end();
```

The intent is to determine the incoming direction at `n_1` (so that recursive sub-cycle search uses the correct outgoing direction). The correct predecessor of `n_1` is `starting_node` (the `current_node` argument at entry to `find_cycle`). `path.back()` is the last node visited by the main while-loop.

**When is this safe?** In a balanced integer flow, `find_cycle` always consumes the closing arc `n_k → starting_node` before stopping (because `starting_node` has balanced parity flow). So `path.back() == starting_node` in the common case, making the expression correct.

**When can it fail?** If the parity direction at `n_k` does not match the direction of the closing arc `n_k → starting_node` (e.g., due to an incomplete subtour that escaped the subtour elimination above), the traversal stops at `n_k` before reaching `starting_node`, and `path.back() = n_k ≠ starting_node`. The direction for the sub-cycle search at `n_1` is then computed against `n_k` instead of `starting_node`, which is geometrically unrelated.

**Fix:** save the original `current_node` before the loop:
```cpp
int starting_node = current_node;   // save before loop modifies current_node
while (true){ ... current_node = next_node; ... }
if (path.empty()) return;
auto node = path.begin();
curr_minus = std::find(minus_sets[*node].V.begin(),
                       minus_sets[*node].V.end(), starting_node)  // FIX
             != minus_sets[*node].V.end();
```

This secondary issue is a latent bug that becomes active precisely when Bug 1 lets a malformed cycle through.

---

## Summary

| # | Location | Bug | Effect |
|---|----------|-----|--------|
| 1 | `ilp.hpp:178-185` | Last subtour node `n_k` skipped in subtour elimination | Some subtours survive the MIP; their coverage is counted in the LP solution but lost during path reconstruction → **coverage constraint violated** |
| 2 | `ilp.cpp:346-348` | `curr_minus` for first node's recursive sub-cycle uses `path.back()` instead of `starting_node` | When Bug 1 produces a malformed cycle, sub-cycle search at `n_1` looks in the wrong direction, missing additional flow |
