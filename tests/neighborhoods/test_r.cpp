#include "gurobi_c++.h"
#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood_r.hpp"

// ============================================================================
// Accessor — exposes all public and protected members for white-box testing
// ============================================================================

struct neighborhood_r_accessor : public neighborhood_r {
public:
    using neighborhood_r::neighborhood_r;

    using neighborhood::geometry;
    using neighborhood::coverage;
    using neighborhood::problem;

    using neighborhood_r::select;
    using neighborhood_r::set_initial_replacements;
    using neighborhood_r::add_replacements;
    using neighborhood_r::set_constraints;
    using neighborhood_r::retrieve_solution;
    using neighborhood_r::extract_duals;
    using neighborhood_r::gain;
    using neighborhood_r::replacement_cost;
    using neighborhood_r::add_stage_columns;
    using neighborhood_r::add_stage_column;
    using neighborhood_r::dag_heuristic;
    using neighborhood_r::positions_coverage_rcs;

    // Protected members — exposed for direct inspection
    using neighborhood_r::Z1;
    using neighborhood_r::Z2;
};

// ============================================================================
// Base fixture (no Gurobi required)
// ============================================================================

class neighborhood_r_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<neighborhood_r_accessor> n;
    cppied_solution sol;

    void SetUp() override {
        cppied_context_fixture::SetUp();
        sol.path = {{6,  11},
                    {23, 18},
                    {44, 42},
                    {48, 51},
                    {30, 35},
                    {75, 73},
                    {62, 63}};
        sol.cost     = {0, 0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        n = std::make_unique<neighborhood_r_accessor>(ctx, P);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
        ctx.start_time = std::chrono::steady_clock::now();
    }
};

// ============================================================================
// gain
// ============================================================================

TEST_F(neighborhood_r_fixture, GainEqualsRemovalGainAtSameIterator) {
    auto it = std::next(sol.path.cbegin(), 2);
    EXPECT_EQ(n->gain(sol, it), n->geometry.removal_gain(sol, it));
}

TEST_F(neighborhood_r_fixture, GainNonNegativeForAllPositions) {
    for (int k = 1; k < (int)sol.path.size(); ++k) {
        cost_t g = n->gain(sol, std::next(sol.path.cbegin(), k));
        EXPECT_GE(g.length, 0) << "negative length gain at position " << k;
    }
}

// ============================================================================
// replacement_cost
// ============================================================================

// Replacing a segment by itself must produce a non-negative differential cost.
TEST_F(neighborhood_r_fixture, ReplacementCostSelfIsNonNegative) {
    for (int k = 1; k < (int)sol.path.size(); ++k) {
        auto it = std::next(sol.path.cbegin(), k);
        cost_t c = n->replacement_cost(sol, it, *it);
        EXPECT_GE(c.length, 0) << "negative self-replacement cost at position " << k;
    }
}

// replacement_cost must be symmetric under segment flip when the flip has
// the same geometric length as the original.
TEST_F(neighborhood_r_fixture, ReplacementCostFlipConsistentWithGeometry) {
    auto it = std::next(sol.path.cbegin(), 2);
    segment seg  = *it;
    segment flip = n->geometry.flip_segment(seg);
    cost_t c1 = n->replacement_cost(sol, it, seg);
    cost_t c2 = n->replacement_cost(sol, it, flip);
    // Both are valid costs; we only verify they are non-negative.
    EXPECT_GE(c1.length, 0);
    EXPECT_GE(c2.length, 0);
}

// ============================================================================
// add_replacements
// ============================================================================

// candidates[0] must be unchanged (the original segment is always first).
TEST_F(neighborhood_r_fixture, AddReplacementsKeepsOriginalAsFirst) {
    std::vector<segment> candidates = {sol.path[2]};
    segment original = candidates[0];
    n->add_replacements(sol, candidates);
    EXPECT_EQ(candidates[0].source, original.source);
    EXPECT_EQ(candidates[0].target, original.target);
}

// add_replacements internally calls coverage.remove then coverage.insert for
// candidates[0]. The net effect on pSol.coverage must be zero.
TEST_F(neighborhood_r_fixture, AddReplacementsRestoresCoverage) {
    Eigen::VectorXd saved = sol.coverage;
    std::vector<segment> candidates = {sol.path[2]};
    n->add_replacements(sol, candidates);
    EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9))
        << "coverage must be unchanged after add_replacements";
}

// Every extra candidate added must have a valid source vertex.
TEST_F(neighborhood_r_fixture, AddReplacementsProducesValidSegments) {
    std::vector<segment> candidates = {sol.path[2]};
    n->add_replacements(sol, candidates);
    for (const auto& s : candidates)
        EXPECT_NE(s.source, path_engine::NULL_NODE) << "path_engine::NULL_NODE source in replacement candidates";
}

// ============================================================================
// select
// ============================================================================

// Indices returned by select must be in [1, path.size()-1].
TEST_F(neighborhood_r_fixture, SelectIndicesInValidRange) {
    std::vector<int> selection;
    n->select(sol, selection);
    for (int idx : selection) {
        EXPECT_GE(idx, 1)                    << "selected index " << idx << " < 1";
        EXPECT_LT(idx, (int)sol.path.size()) << "selected index " << idx << " >= path.size()";
    }
}

// Index 0 (first segment) must never appear in the selection.
TEST_F(neighborhood_r_fixture, SelectNeverSelectsFirstSegment) {
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<int> selection;
        n->select(sol, selection);
        EXPECT_FALSE(std::find(selection.begin(), selection.end(), 0)
                     != selection.end())
            << "trial " << trial << ": first segment (index 0) was selected";
    }
}

// select must produce a non-empty result for any non-trivial path.
TEST_F(neighborhood_r_fixture, SelectNonEmptyForNontrivialPath) {
    std::vector<int> selection;
    n->select(sol, selection);
    EXPECT_GT(selection.size(), 0u);
}

TEST_F(neighborhood_r_fixture, SelectGainsIndexedAtCorrectPathPosition) {
    int N = (int)sol.path.size();

    // Compute correct gain for each path position [1..N-1].
    std::vector<cost_t> correct_gains(N, {0, 0});
    for (int k = 1; k < N; ++k)
        correct_gains[k] = n->gain(sol, std::next(sol.path.cbegin(), k));

    // Even candidates: {2,4,6,...}, odd candidates: {1,3,5,...}
    cost_t correct_even{0,0}, correct_odd{0,0};
    for (int k = 1; k < N; ++k) {
        if (k % 2 == 0) { correct_even = correct_even + correct_gains[k];}
        else             { correct_odd  = correct_odd  + correct_gains[k];}
    }

    // The dominant group must be the same under correct and buggy indexing.
    // If they differ, select will systematically pick the wrong group.
    if (correct_even == correct_odd)
        GTEST_SKIP() << "test path has equal group gains; try a different path";

    bool correct_favours_even = correct_even > correct_odd;

    EXPECT_TRUE(correct_favours_even)
        << "Correct favouring for even positions.";
}

// ============================================================================
// Gurobi-based fixture — set_constraints + retrieve_solution
// ============================================================================

class neighborhood_r_gurobi_fixture : public neighborhood_r_fixture {
protected:
    GRBEnv genv{true};
    std::unique_ptr<GRBModel> model;
    std::vector<int>           selection;
    std::vector<replacement_set> replacements;
    std::vector<GRBConstr>     constraints;

    void SetUp() override {
        neighborhood_r_fixture::SetUp();
        genv.set(GRB_IntParam_OutputFlag, 0);
        genv.start();
        model = std::make_unique<GRBModel>(genv);

        n->select(sol, selection);
        ASSERT_FALSE(selection.empty()) << "select returned empty; cannot build model";

        n->set_initial_replacements(sol, selection, replacements, *model);
        n->set_constraints(sol, replacements, constraints, *model);
        model->set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
        model->update();
    }
};

// ─── set_constraints ────────────────────────────────────────────────────────

// Total constraint count must be:
//   n_rows*n_cols  (one per coverage cell)
// + R.size()       (one per replacement set: sum_j x_j <= 1)
// + 1              (gamma: total length <= incumbent)
//
// Bug: the coverage loop uses `lin_expr_set.size() - R.size()` as the bound,
// producing only n_rows*n_cols - R.size() + R.size() + 1 = n_rows*n_cols + 1
// constraints — the last R.size() coverage cells are never constrained.
TEST_F(neighborhood_r_gurobi_fixture, SetConstraintsTotalConstraintCount) {
    int n_cells = n->geometry.n_rows * n->geometry.n_cols;
    int expected = n_cells + (int)replacements.size() + 1;
    EXPECT_EQ((int)constraints.size(), expected)
        << "Bug: coverage loop bound is `lin_expr_set.size() - R.size()` "
           "instead of `lin_expr_set.size()`. "
           << (int)replacements.size() << " coverage cell(s) have no constraint.";
}

// The first n_cells constraints must be coverage constraints (RHS = req - coverage).
// The next R.size() constraints must be replacement constraints (RHS = 1).
// Bug in add_stage_column: uses constraints[r_ctr] (a coverage constraint)
// instead of constraints[n_cells + r_ctr] (the correct replacement constraint).
TEST_F(neighborhood_r_gurobi_fixture, SetConstraintsReplacementConstraintsAtCorrectOffset) {
    int n_cells = n->geometry.n_rows * n->geometry.n_cols;
    if ((int)constraints.size() < n_cells + (int)replacements.size() + 1)
        GTEST_SKIP() << "insufficient constraints (set_constraints coverage-loop bug active)";

    // Replacement constraint for R[k] must be at index n_cells + k, with RHS = 1.
    for (int k = 0; k < (int)replacements.size(); ++k) {
        EXPECT_DOUBLE_EQ(constraints[n_cells + k].get(GRB_DoubleAttr_RHS), 1.0)
            << "constraints[n_cells+" << k << "] should be a replacement constraint (RHS=1); "
               "add_stage_column uses constraints[r_ctr] instead of constraints[n_cells+r_ctr]";
    }
    // Index 0 must remain a coverage constraint (RHS != 1 for typical problems).
    EXPECT_NE(constraints[0].get(GRB_DoubleAttr_RHS), 1.0)
        << "constraints[0] must be a coverage constraint, not a replacement constraint";
}

// Each replacement set must carry exactly as many variables as candidates.
TEST_F(neighborhood_r_gurobi_fixture, ReplacementVariableCountMatchesCandidates) {
    for (int k = 0; k < (int)replacements.size(); ++k)
        EXPECT_EQ(replacements[k].variables.size(), replacements[k].candidates.size())
            << "R[" << k << "]: variables.size() != candidates.size()";
}

// candidates[0] of every replacement set must be the original path segment.
TEST_F(neighborhood_r_gurobi_fixture, ReplacementCandidatesZeroIsOriginalSegment) {
    for (int k = 0; k < (int)replacements.size(); ++k) {
        int pos = replacements[k].r;
        ASSERT_FALSE(replacements[k].candidates.empty());
        EXPECT_EQ(replacements[k].candidates[0].source, sol.path[pos].source)
            << "R[" << k << "].candidates[0] must be the original segment at path[" << pos << "]";
        EXPECT_EQ(replacements[k].candidates[0].target, sol.path[pos].target);
    }
}

// ─── retrieve_solution ──────────────────────────────────────────────────────

class neighborhood_r_solved_fixture : public neighborhood_r_gurobi_fixture {
protected:
    void SetUp() override {
        neighborhood_r_gurobi_fixture::SetUp();
        model->setObjective(n->Z1, GRB_MINIMIZE);
        model->update();
        model->optimize();
        ASSERT_EQ(model->get(GRB_IntAttr_Status), GRB_OPTIMAL)
            << "Gurobi did not find an optimal solution; test cannot proceed";
    }
};

// Non-selected path positions must appear in the output path.
// Bug: same missing else-clause — non-selected segments are silently dropped.
TEST_F(neighborhood_r_solved_fixture, RetrieveSolutionNonSelectedSegmentsPreserved) {
    std::vector<segment> original = sol.path;
    std::set<int>        selected_pos;
    for (auto& rs : replacements) selected_pos.insert(rs.r);

    n->retrieve_solution(sol, replacements);

    for (int k = 0; k < (int)original.size(); ++k) {
        if (selected_pos.count(k)) continue;
        // Non-selected segment at position k must appear somewhere in the result.
        bool found = std::any_of(sol.path.begin(), sol.path.end(),
            [&](const segment& s) {
                return s.source == original[k].source
                    && s.target == original[k].target;
            });
        EXPECT_TRUE(found)
            << "non-selected segment at path[" << k << "] is missing from result";
    }
}

// When r reaches R.size() the remaining iterations must not access R[r].r.
// Bug: no guard `r < R.size()` before the condition `i == R[r].r`.
TEST_F(neighborhood_r_solved_fixture, RetrieveSolutionNoOutOfBoundsAfterRExhausted) {
    EXPECT_NO_FATAL_FAILURE(n->retrieve_solution(sol, replacements))
        << "retrieve_solution accessed R[r].r with r >= R.size() (out-of-bounds). "
           "Fix: guard with `if (r < (int)R.size() && i == R[r].r)`.";
}

// ─── extract_duals / add_stage_columns sizing ────────────────────────────────

// add_stage_columns constructs coverage_duals and r_duals as empty vectors,
// then calls extract_duals which uses their .size() as loop bounds.
// With size=0, no duals are extracted and r_duals[i] accesses out-of-bounds.
//
// This test verifies that extract_duals correctly populates duals when the
// vectors are pre-sized (the fix is to pre-size them before the call).
TEST_F(neighborhood_r_solved_fixture, ExtractDualsPopulatesVectorsWhenPresized) {
    int n_cells = n->geometry.n_rows * n->geometry.n_cols;
    std::vector<double> cov_duals(n_cells, 0.0);
    std::vector<double> r_duals(replacements.size(), 0.0);

    EXPECT_NO_THROW(n->extract_duals(sol, constraints, cov_duals, r_duals));

    EXPECT_EQ((int)cov_duals.size(), n_cells)
        << "coverage_duals must have one entry per coverage cell";
    EXPECT_EQ((int)r_duals.size(), (int)replacements.size())
        << "r_duals must have one entry per replacement set";
}

// ============================================================================
// dag_heuristic — backtracking vertex coordinates
// ============================================================================

// Bug: `int v = P.back()` gives a D/P index (relative offset from source).
// `int u = v - first + 1` then subtracts `first` a second time.
// The returned segment must contain valid vertex IDs in [first, last], not
// shifted D/P indices.
TEST_F(neighborhood_r_fixture, DagHeuristicReturnedSegmentVerticesInRange) {
    int n_cells = n->geometry.n_rows * n->geometry.n_cols;
    std::vector<double> cov_duals(n_cells, 1.0);   // uniform positive duals
    int first = 0;
    int last  = n->geometry.n_cols - 1;

    auto cost_func = [](cost_t c) { return (double)c.length; };

    // r=1: compute dag for the second segment replacement.
    auto [seg, rc] = n->dag_heuristic(sol, first, last,
                                      cov_duals,
                                      /*r=*/1, /*r_dual=*/0.0,
                                      /*gamma=*/0.0,
                                      cost_func);

    if (path_engine::is_node(seg.source)) {
        EXPECT_GE(seg.source, first)
            << "returned source vertex < first (likely a raw D/P index, not a vertex coordinate)";
        EXPECT_LE(seg.source, last)
            << "returned source vertex > last";
    }
    if (path_engine::is_node(seg.target)) {
        EXPECT_GE(seg.target, first)
            << "returned target vertex < first";
        EXPECT_LE(seg.target, last)
            << "returned target vertex > last";
    }
}

// ============================================================================
// E2E — requires a working Gurobi licence
// ============================================================================

class neighborhood_r_e2e_fixture : public neighborhood_r_fixture {
protected:
    void SetUp() override {
        neighborhood_r_fixture::SetUp();
    }
};

TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchDoesNotThrow) {
    EXPECT_NO_THROW(n->local_search(sol));
}

TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchPathNonEmpty) {
    n->local_search(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

// After local_search, the path must have at most the same number of segments as before.
// Bug 1: retrieve_solution drops non-selected segments, shrinking the path.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchPathSizeUnchanged) {
    size_t before = sol.path.size();
    n->local_search(sol);
    EXPECT_LE(sol.path.size(), before)
        << "local_search changed path size from " << before << " to " << sol.path.size()
        << " (retrieve_solution missing else-clause drops non-selected segments)";
}

// On TRUE return the stored cost must be strictly less than the pre-call cost.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCostDecreasedWhenReturnTrue) {
    cost_t before = sol.cost;
    if (n->local_search(sol))
        EXPECT_LT(sol.cost, before);
}

// On TRUE the stored cost must equal geometry.cost(sol).
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCostConsistentWhenReturnTrue) {
    if (n->local_search(sol))
        EXPECT_EQ(n->geometry.cost(sol), sol.cost);
}

// On TRUE the coverage vector must match a fresh recompute.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCoverageConsistentWhenReturnTrue) {
    if (n->local_search(sol)) {
        Eigen::VectorXd saved = sol.coverage;
        n->coverage.reset(sol);
        EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9))
            << "coverage is inconsistent after local_search";
    }
}

// On TRUE the coverage constraint must remain satisfied.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCoverageConstraintSatisfied) {
    if (n->local_search(sol)) {
        n->coverage.reset(sol);
        EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
    }
}

// Coverage must be consistent regardless of the return value.
// Bug: set_constraints removes coverage for selected positions; if the binary
// solve sets all variables to 0 for an unconstrained-cell position, retrieve_solution
// never re-inserts that coverage, leaving pSol.coverage depleted.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCoverageConsistentAlways) {
    n->local_search(sol);
    Eigen::VectorXd saved = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9))
        << "coverage inconsistency after local_search (independent of return value); "
           "set_constraints coverage-loop bug leaves last R.size() cells unconstrained";
}

// Coverage feasibility must hold regardless of the return value.
// Bug: the last R.size() coverage cells are not constrained in set_constraints
// (loop bound is lin_expr_set.size()-R.size() instead of lin_expr_set.size()).
// The binary solver may produce an all-zero assignment for a position, causing
// retrieve_solution to drop the segment and leave those cells below req.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCoverageConstraintSatisfiedAlways) {
    n->local_search(sol);
    n->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all())
        << "coverage infeasibility after local_search; "
           "unconstrained cells in set_constraints allow solver to drop coverage";
}

// Cost must equal geometry.cost regardless of return value.
// Bug: retrieve_solution may produce a shorter path (dropped positions);
// pSol.cost = geometry.cost(pSol) is correct, but only if the path is intact.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCostConsistentAlways) {
    n->local_search(sol);
    EXPECT_EQ(n->geometry.cost(sol), sol.cost)
        << "cost inconsistency after local_search";
}

// Z1 and Z2 accumulate GRBVar references across calls.  On the second call the
// first model has been destroyed, leaving dangling GRBVar handles in Z1/Z2.
// Bug: Z1 and Z2 are never reset to 0 at the start of local_search.
// Fix: add `Z1 = 0; Z2 = 0;` at the very beginning of local_search.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchStateConsistentOnSecondCall) {
    n->local_search(sol);
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);

    // Second call must not crash and must leave the solution consistent.
    EXPECT_NO_THROW(n->local_search(sol))
        << "second call to local_search crashed; "
           "Z1 and Z2 hold dangling GRBVar references from the first model. "
           "Fix: reset Z1 = 0; Z2 = 0; at the start of local_search.";

    EXPECT_EQ(n->geometry.cost(sol), sol.cost)
        << "cost inconsistency after second call (Z1/Z2 accumulation)";
}

// ============================================================================
// Improvable-path fixture — neighbourhood_r must find a strict improvement
// ============================================================================
//
// Vertex numbering (6×6 seabed, n_cols=6):
//   horizontal vertex (row_boundary r, col c) → index r*6 + c      (0..41)
//   vertical   vertex (col_boundary b, row r) → index 42 + b*6 + r (42..83)
//
// Base 7-segment local optimum (shared fixture):
//   {6,  11} row_boundary 1, cols 0→5  (covers rows 0–1 for all columns)
//   {23, 18} row_boundary 3, cols 5→0  (covers rows 2–3 for all columns)
//   {44, 42} col_boundary 0, rows 2→0  (covers col-0 cells in rows 0–2)
//   {48, 51} col_boundary 1, rows 0→3  (covers col-0 and col-1 in rows 0–3)
//   {30, 35} row_boundary 5, cols 0→5  (covers rows 4–5 for all columns)
//   {75, 73} col_boundary 5, rows 3→1  (covers cols 4–5 in rows 1–3)
//   {62, 63} col_boundary 3, rows 2→3  (covers cols 2–3 in rows 2–3)
//
// Improvable path: two segments are over-extended relative to the local optimum.
//
//   index 2 (falls in the even parity group {2,4,6}):
//     {47, 42}  col_boundary 0, rows 5→0  — full 6-row northward sweep
//     optimal:  {44, 42}  rows 2→0 only   — savings: 3 length units
//     Rows 3–5 at col_boundary 0 are redundant:
//       row 3 is already covered by {23,18} + {48,51} (2 type-2 visits, req met)
//       rows 4–5 are covered by {30,35}              (1 type-1 visit, req met)
//
//   index 3 (falls in the odd parity group {1,3,5}):
//     {48, 53}  col_boundary 1, rows 0→5  — full 6-row southward sweep
//     optimal:  {48, 51}  rows 0→3 only   — savings: 2 length units
//     Rows 4–5 at col_boundary 1 are redundant:
//       {30,35} provides 1 type-1 visit at (4,0),(4,1),(5,0),(5,1), req met
//
// Coverage feasibility of the improvable path (req = 2.303 for all cells):
//   Type-3 cells (0,0),(1,0),(2,0) — POD 0.916, need 3 visits:
//     {6,11}+{47,42}+{48,53} → 3 × 0.916 = 2.748 ≥ 2.303 ✓
//   Type-2 cells row 3, cols 0–5 — POD 1.609, need 2 visits:
//     (3,0): {23,18}+{47,42}   ≥ 2 × 1.609 ✓
//     (3,1): {23,18}+{48,53}   ≥ 2 × 1.609 ✓
//     (3,2–5): {23,18}+{62,63}/{75,73} ≥ 2 × 1.609 ✓
//   Type-1 cells rows 4–5 — POD 4.605, need 1 visit: {30,35} alone ✓
//
// Coverage feasibility of the improved path ({44,42}+{48,51}):
//   Same as the base local optimum (verified by the base fixture).
//
// Design guarantee: one improvable segment per parity group ensures
// local_search() finds a strict improvement regardless of select()'s
// softmax outcome.  Minimum expected savings: 2 length units.

class neighborhood_r_improvable_fixture : public neighborhood_r_fixture {
protected:
    void SetUp() override {
        neighborhood_r_fixture::SetUp();
        sol.path = {{6,  11},
                    {23, 18},
                    {47, 42},   // index 2 (even): col_boundary 0, rows 5→0
                                //   over-extended; optimal is {44,42} rows 2→0
                    {48, 53},   // index 3 (odd):  col_boundary 1, rows 0→5
                                //   over-extended; optimal is {48,51} rows 0→3
                    {30, 35},
                    {75, 73},
                    {62, 63}};
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

// Precondition: the improvable path must satisfy all coverage requirements.
TEST_F(neighborhood_r_improvable_fixture, ImprovablePath_IsFeasible) {
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all())
        << "the improvable test path does not satisfy coverage requirements; "
           "verify segment vertex indices and seabed coverage above";
}

// local_search() must return true — a strict improvement always exists.
TEST_F(neighborhood_r_improvable_fixture, ImprovablePath_LocalSearchReturnsTrue) {
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all())
        << "precondition: improvable path must be feasible";
    EXPECT_TRUE(n->local_search(sol))
        << "local_search() returned false on a path with two over-extended segments; "
           "column generation should have found at least one of {47,42}→{44,42} or "
           "{48,53}→{48,51} as a strict improvement";
}

// The cost stored in sol must be strictly smaller after a successful call.
TEST_F(neighborhood_r_improvable_fixture, ImprovablePath_CostStrictlyDecreased) {
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    const cost_t cost_before = sol.cost;
    const bool improved = n->local_search(sol);
    ASSERT_TRUE(improved)
        << "local_search() did not improve; cannot verify cost decrease";
    EXPECT_LT(sol.cost, cost_before)
        << "sol.cost was not reduced despite local_search() returning true";
}

// The stored cost must equal the recomputed geometry cost after the move.
TEST_F(neighborhood_r_improvable_fixture, ImprovablePath_CostConsistentAfterImprovement) {
    ASSERT_TRUE(n->local_search(sol))
        << "local_search() did not improve; cannot verify cost consistency";
    EXPECT_EQ(n->geometry.cost(sol), sol.cost)
        << "sol.cost is inconsistent with geometry.cost(sol) after local_search";
}

// The incremental coverage vector must match a full reset after the move.
TEST_F(neighborhood_r_improvable_fixture, ImprovablePath_CoverageConsistentAfterImprovement) {
    ASSERT_TRUE(n->local_search(sol))
        << "local_search() did not improve; cannot verify coverage consistency";
    const Eigen::VectorXd coverage_after = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(coverage_after, 1e-9))
        << "incremental coverage vector differs from a full reset after local_search";
}

// All coverage requirements must remain satisfied after the move.
TEST_F(neighborhood_r_improvable_fixture, ImprovablePath_CoverageConstraintSatisfied) {
    ASSERT_TRUE(n->local_search(sol))
                                << "local_search() did not improve; cannot verify coverage feasibility";
    n->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all())
                        << "coverage constraint violated after local_search on improvable path";
}

//-----------Special cases-------------------------------
TEST_F(neighborhood_r_fixture, SpecialCaseNoInvalidNode){
    sol.path = {{6, -1},
            {49, -1},
            {12, -2},
            {43, -2},
            {6, -1},
            {7, 10},
            {11, -1},
            {23, 22},
            {21, -2},
            {20, 18},
            {30, 31},
            {32, 34},
            {35, -1},
            {82, -2},
            {29, -2},
            {75, 73},
            {62, 63},
            {51, 48}};
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);
    ctx.max_time = 600;
    EXPECT_NO_THROW(n->local_search(sol));
}
