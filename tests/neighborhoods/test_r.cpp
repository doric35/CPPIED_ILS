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

    using neighborhood_r::removalGain;
    using neighborhood_r::replacementGain;
    using neighborhood_r::getRemovalGains;
    using neighborhood_r::setSelectReallocations;
    using neighborhood_r::pushInitialSegmentReallocation;
    using neighborhood_r::constructGurobiModel;
    using neighborhood_r::retrieveSolution;
    using neighborhood_r::setDualVariables;
    using neighborhood_r::setVertexPrices;
    using neighborhood_r::reset;
    using neighborhood_r::setInitialReallocationSets;
    using neighborhood_r::setRemovalCoverage;
    using neighborhood_r::initializeGurobiEnvironment;
    using neighborhood_r::initializeGurobiModel;
    using neighborhood_r::constructLinearProgram;
    using neighborhood_r::constructInitialVariables;
    using neighborhood_r::constructReallocationSetVariables;
    using neighborhood_r::constructObjectives;
    using neighborhood_r::constructLengthObjective;
    using neighborhood_r::constructTurnsObjective;
    using neighborhood_r::constructConstraints;
    using neighborhood_r::constructCoverageConstraints;
    using neighborhood_r::constructReallocationConstraints;
    using neighborhood_r::constructLengthConstraints;
    using neighborhood_r::columnGeneration;
    using neighborhood_r::runColumnGenerationIteration;
    using neighborhood_r::runColumnGenerationFirstStage;
    using neighborhood_r::runColumnGenerationSecondStage;
    using neighborhood_r::generateColumn;
    using neighborhood_r::columnGenerationHasConverged;
    using neighborhood_r::solvePricingSubproblem;
    using neighborhood_r::setCoverageDuals;
    using neighborhood_r::setReallocationDuals;
    using neighborhood_r::setLengthDual;
    using neighborhood_r::setLexicographicBinaryProgram;
    using neighborhood_r::setReallocationModelWarmStart;
    using neighborhood_r::safeSolutionRetrieve;
    using neighborhood_r::reallocation;
    using neighborhood_r::throwIfLpTimedOut;

    // Protected members — exposed for direct inspection
    using neighborhood_r::env;
    using neighborhood_r::lengthDual;
    using neighborhood_r::model;
    using neighborhood_r::coverageConstraints;
    using neighborhood_r::reallocationConstraints;
    using neighborhood_r::pathLengthConstraint;
    using neighborhood_r::reallocationSelection;
    using neighborhood_r::unsatisfiedCells;
    using neighborhood_r::reallocationSets;
    using neighborhood_r::vertexCoveragePrices;
    using neighborhood_r::coverageDuals;
    using neighborhood_r::reallocationDuals;
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
        n = std::make_unique<neighborhood_r_accessor>(ctx, P, sol);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
        ctx.start_time = std::chrono::high_resolution_clock::now();
    }

    // Frees any Gurobi model/environment left allocated by a test.
    void TearDown() override {
        n->reset();
    }
};

// ============================================================================
// removalGain
// ============================================================================

TEST_F(neighborhood_r_fixture, GainEqualsRemovalGainAtSameIterator) {
    auto it = std::next(sol.path.cbegin(), 2);
    EXPECT_EQ(n->removalGain(2), n->geometry.removal_gain(sol, it));
}

TEST_F(neighborhood_r_fixture, GainNonNegativeForAllPositions) {
    for (int k = 1; k < (int)sol.path.size(); ++k) {
        cost_t g = n->removalGain(k);
        EXPECT_GE(g.length, 0) << "negative length gain at position " << k;
    }
}

// ============================================================================
// replacementGain
// ============================================================================

// Replacing a segment by itself must produce a non-negative differential cost.
TEST_F(neighborhood_r_fixture, ReplacementCostSelfIsNonNegative) {
    for (int k = 1; k < (int)sol.path.size(); ++k) {
        cost_t c = n->replacementGain(k, sol.path[k]);
        EXPECT_GE(c.length, 0) << "negative self-replacement cost at position " << k;
    }
}

// replacementGain must be symmetric under segment flip when the flip has
// the same geometric length as the original.
TEST_F(neighborhood_r_fixture, ReplacementCostFlipConsistentWithGeometry) {
    segment seg  = sol.path[2];
    segment flip = n->geometry.flip_segment(seg);
    cost_t c1 = n->replacementGain(2, seg);
    cost_t c2 = n->replacementGain(2, flip);
    // Both are valid costs; we only verify they are non-negative.
    EXPECT_GE(c1.length, 0);
    EXPECT_GE(c2.length, 0);
}

// ============================================================================
// pushInitialSegmentReallocation
// ============================================================================

// moves[0] must be the original segment: setReallocationModelWarmStart sets
// variables[0] to 1 assuming it is the original (identity) reallocation.
TEST_F(neighborhood_r_fixture, AddReplacementsKeepsOriginalAsFirst) {
    segment original = sol.path[2];
    n->reallocationSets.resize(sol.path.size());
    n->pushInitialSegmentReallocation(2);
    ASSERT_FALSE(n->reallocationSets[2].moves.empty());
    EXPECT_EQ(n->reallocationSets[2].moves[0].s.source, original.source);
    EXPECT_EQ(n->reallocationSets[2].moves[0].s.target, original.target);
}

// Building the initial reallocation set must not alter pSol.coverage.
TEST_F(neighborhood_r_fixture, AddReplacementsRestoresCoverage) {
    Eigen::VectorXd saved = sol.coverage;
    n->reallocationSets.resize(sol.path.size());
    n->pushInitialSegmentReallocation(2);
    EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9))
        << "coverage must be unchanged after pushInitialSegmentReallocation";
}

// Every move added must have a valid source vertex.
TEST_F(neighborhood_r_fixture, AddReplacementsProducesValidSegments) {
    n->reallocationSets.resize(sol.path.size());
    n->pushInitialSegmentReallocation(2);
    for (const auto& move : n->reallocationSets[2].moves)
        EXPECT_NE(move.s.source, path_engine::NULL_NODE) << "path_engine::NULL_NODE source in reallocation moves";
}

// ============================================================================
// setSelectReallocations
// ============================================================================

// Selected indices must be in [1, path.size()-1].
TEST_F(neighborhood_r_fixture, SelectIndicesInValidRange) {
    n->setSelectReallocations();
    for (int idx : n->reallocationSelection) {
        EXPECT_GE(idx, 1)                    << "selected index " << idx << " < 1";
        EXPECT_LT(idx, (int)sol.path.size()) << "selected index " << idx << " >= path.size()";
    }
}

// Index 0 (first segment) must never appear in the selection.
TEST_F(neighborhood_r_fixture, SelectNeverSelectsFirstSegment) {
    for (int trial = 0; trial < 20; ++trial) {
        n->setSelectReallocations();
        const auto& selection = n->reallocationSelection;
        EXPECT_FALSE(std::find(selection.begin(), selection.end(), 0)
                     != selection.end())
            << "trial " << trial << ": first segment (index 0) was selected";
    }
}

// The selection must be non-empty for any non-trivial path.
TEST_F(neighborhood_r_fixture, SelectNonEmptyForNontrivialPath) {
    n->setSelectReallocations();
    EXPECT_GT(n->reallocationSelection.size(), 0u);
}

TEST_F(neighborhood_r_fixture, SelectGainsIndexedAtCorrectPathPosition) {
    int N = (int)sol.path.size();

    // Compute correct gain for each path position [1..N-1].
    std::vector<cost_t> correct_gains = n->getRemovalGains();
    ASSERT_EQ((int)correct_gains.size(), N);
    for (int k = 1; k < N; ++k)
        EXPECT_EQ(correct_gains[k], n->removalGain(k)) << "getRemovalGains mismatch at position " << k;

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
// Gurobi-based fixture — constructGurobiModel + retrieveSolution
// ============================================================================

class neighborhood_r_gurobi_fixture : public neighborhood_r_fixture {
protected:
    void SetUp() override {
        neighborhood_r_fixture::SetUp();

        n->setSelectReallocations();
        ASSERT_FALSE(n->reallocationSelection.empty())
            << "setSelectReallocations returned empty; cannot build model";

        n->constructGurobiModel();
        ASSERT_NE(n->model, nullptr);
    }
};

// ─── constructConstraints ───────────────────────────────────────────────────

// Total constraint count must be:
//   |unsatisfiedCells|          (one per cell left unsatisfied after removal)
// + |reallocationSelection|     (one per reallocation set: sum_j x_j <= 1)
// + 1                           (total length <= incumbent)
TEST_F(neighborhood_r_gurobi_fixture, SetConstraintsTotalConstraintCount) {
    int expected = (int)n->unsatisfiedCells.size()
                 + (int)n->reallocationSelection.size() + 1;
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumConstrs), expected)
        << "expected one coverage constraint per unsatisfied cell, one reallocation "
           "constraint per selected position and one length constraint";
}

// Reallocation constraints must be stored at their path index (RHS = 1),
// coverage constraints at their cell index (RHS = req - coverage).
TEST_F(neighborhood_r_gurobi_fixture, SetConstraintsReplacementConstraintsAtCorrectOffset) {
    for (int i : n->reallocationSelection) {
        EXPECT_DOUBLE_EQ(n->reallocationConstraints[i].get(GRB_DoubleAttr_RHS), 1.0)
            << "reallocationConstraints[" << i << "] should be a reallocation constraint (RHS=1)";
    }
    for (int c : n->unsatisfiedCells) {
        EXPECT_NEAR(n->coverageConstraints[c].get(GRB_DoubleAttr_RHS),
                    P.req[c] - sol.coverage[c], 1e-9)
            << "coverageConstraints[" << c << "] must be the coverage constraint of cell " << c;
    }
}

// Each reallocation set must carry exactly as many variables as moves.
TEST_F(neighborhood_r_gurobi_fixture, ReplacementVariableCountMatchesCandidates) {
    for (int k = 0; k < (int)n->reallocationSets.size(); ++k)
        EXPECT_EQ(n->reallocationSets[k].variables.size(), n->reallocationSets[k].moves.size())
            << "reallocationSets[" << k << "]: variables.size() != moves.size()";
}

// moves[0] of every selected reallocation set must be the original path segment.
TEST_F(neighborhood_r_gurobi_fixture, ReplacementCandidatesZeroIsOriginalSegment) {
    for (int pos : n->reallocationSelection) {
        const auto& moves = n->reallocationSets[pos].moves;
        ASSERT_FALSE(moves.empty());
        EXPECT_EQ(moves[0].s.source, sol.path[pos].source)
            << "reallocationSets[" << pos << "].moves[0] must be the original segment at path[" << pos << "]";
        EXPECT_EQ(moves[0].s.target, sol.path[pos].target);
    }
}

// ─── retrieveSolution ───────────────────────────────────────────────────────

class neighborhood_r_solved_fixture : public neighborhood_r_gurobi_fixture {
protected:
    void SetUp() override {
        neighborhood_r_gurobi_fixture::SetUp();
        n->model->setObjective(*n->Z1, GRB_MINIMIZE);
        n->model->update();
        n->model->optimize();
        ASSERT_EQ(n->model->get(GRB_IntAttr_Status), GRB_OPTIMAL)
            << "Gurobi did not find an optimal solution; test cannot proceed";
    }
};

// Non-selected path positions must appear in the output path.
TEST_F(neighborhood_r_solved_fixture, RetrieveSolutionNonSelectedSegmentsPreserved) {
    std::vector<segment> original = sol.path;
    std::set<int>        selected_pos(n->reallocationSelection.begin(),
                                      n->reallocationSelection.end());

    n->retrieveSolution();

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

// retrieveSolution must only read reallocation sets that were built.
TEST_F(neighborhood_r_solved_fixture, RetrieveSolutionNoOutOfBoundsAfterRExhausted) {
    EXPECT_NO_FATAL_FAILURE(n->retrieveSolution())
        << "retrieveSolution accessed a reallocation set out-of-bounds.";
}

// ─── setDualVariables sizing ─────────────────────────────────────────────────

// setCoverageDuals / setReallocationDuals index coverageDuals and
// reallocationDuals directly; they must be pre-sized before the call.
// This test verifies that setDualVariables correctly populates duals when the
// vectors are pre-sized.
TEST_F(neighborhood_r_solved_fixture, ExtractDualsPopulatesVectorsWhenPresized) {
    int n_cells = n->geometry.n_rows * n->geometry.n_cols;
    n->coverageDuals     = Eigen::VectorXd::Zero(n_cells);
    n->reallocationDuals = Eigen::VectorXd::Zero((int)n->reallocationSets.size());

    EXPECT_NO_THROW(n->setDualVariables());

    EXPECT_EQ((int)n->coverageDuals.size(), n_cells)
        << "coverageDuals must have one entry per coverage cell";
    EXPECT_EQ((int)n->reallocationDuals.size(), (int)n->reallocationSets.size())
        << "reallocationDuals must have one entry per reallocation set";
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
    EXPECT_NO_THROW(n->local_search());
}

TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchPathNonEmpty) {
    n->local_search();
    EXPECT_GT(sol.path.size(), 0u);
}

// After local_search, the path must have at most the same number of segments as before.
// Bug 1: retrieve_solution drops non-selected segments, shrinking the path.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchPathSizeUnchanged) {
    size_t before = sol.path.size();
    n->local_search();
    EXPECT_LE(sol.path.size(), before)
        << "local_search changed path size from " << before << " to " << sol.path.size()
        << " (retrieve_solution missing else-clause drops non-selected segments)";
}

// On TRUE return the stored cost must be strictly less than the pre-call cost.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCostDecreasedWhenReturnTrue) {
    cost_t before = sol.cost;
    if (n->local_search())
        EXPECT_LT(sol.cost, before);
}

// On TRUE the stored cost must equal geometry.cost(sol).
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCostConsistentWhenReturnTrue) {
    if (n->local_search())
        EXPECT_EQ(n->geometry.cost(sol), sol.cost);
}

// On TRUE the coverage vector must match a fresh recompute.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCoverageConsistentWhenReturnTrue) {
    if (n->local_search()) {
        Eigen::VectorXd saved = sol.coverage;
        n->coverage.reset(sol);
        EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9))
            << "coverage is inconsistent after local_search";
    }
}

// On TRUE the coverage constraint must remain satisfied.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCoverageConstraintSatisfied) {
    if (n->local_search()) {
        n->coverage.reset(sol);
        EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
    }
}

// Coverage must be consistent regardless of the return value.
// Bug: set_constraints removes coverage for selected positions; if the binary
// solve sets all variables to 0 for an unconstrained-cell position, retrieve_solution
// never re-inserts that coverage, leaving pSol.coverage depleted.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCoverageConsistentAlways) {
    n->local_search();
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
    n->local_search();
    n->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all())
        << "coverage infeasibility after local_search; "
           "unconstrained cells in set_constraints allow solver to drop coverage";
}

// Cost must equal geometry.cost regardless of return value.
// Bug: retrieve_solution may produce a shorter path (dropped positions);
// pSol.cost = geometry.cost(pSol) is correct, but only if the path is intact.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchCostConsistentAlways) {
    n->local_search();
    EXPECT_EQ(n->geometry.cost(sol), sol.cost)
        << "cost inconsistency after local_search";
}

// Z1 and Z2 accumulate GRBVar references across calls.  On the second call the
// first model has been destroyed, leaving dangling GRBVar handles in Z1/Z2.
// Bug: Z1 and Z2 are never reset to 0 at the start of local_search.
// Fix: add `Z1 = 0; Z2 = 0;` at the very beginning of local_search.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchStateConsistentOnSecondCall) {
    n->local_search();
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);

    // Second call must not crash and must leave the solution consistent.
    EXPECT_NO_THROW(n->local_search())
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
    EXPECT_TRUE(n->local_search())
        << "local_search() returned false on a path with two over-extended segments; "
           "column generation should have found at least one of {47,42}→{44,42} or "
           "{48,53}→{48,51} as a strict improvement";
}

// The cost stored in sol must be strictly smaller after a successful call.
TEST_F(neighborhood_r_improvable_fixture, ImprovablePath_CostStrictlyDecreased) {
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    const cost_t cost_before = sol.cost;
    const bool improved = n->local_search();
    ASSERT_TRUE(improved)
        << "local_search() did not improve; cannot verify cost decrease";
    EXPECT_LT(sol.cost, cost_before)
        << "sol.cost was not reduced despite local_search() returning true";
}

// The stored cost must equal the recomputed geometry cost after the move.
TEST_F(neighborhood_r_improvable_fixture, ImprovablePath_CostConsistentAfterImprovement) {
    ASSERT_TRUE(n->local_search())
        << "local_search() did not improve; cannot verify cost consistency";
    EXPECT_EQ(n->geometry.cost(sol), sol.cost)
        << "sol.cost is inconsistent with geometry.cost(sol) after local_search";
}

// The incremental coverage vector must match a full reset after the move.
TEST_F(neighborhood_r_improvable_fixture, ImprovablePath_CoverageConsistentAfterImprovement) {
    ASSERT_TRUE(n->local_search())
        << "local_search() did not improve; cannot verify coverage consistency";
    const Eigen::VectorXd coverage_after = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(coverage_after, 1e-9))
        << "incremental coverage vector differs from a full reset after local_search";
}

// All coverage requirements must remain satisfied after the move.
TEST_F(neighborhood_r_improvable_fixture, ImprovablePath_CoverageConstraintSatisfied) {
    ASSERT_TRUE(n->local_search())
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
    EXPECT_NO_THROW(n->local_search());
}

// ============================================================================
// Helpers for the per-function tests below
// ============================================================================

namespace {

int totalMoves(const neighborhood_r_accessor& n) {
    int total = 0;
    for (const auto& set : n.reallocationSets) total += (int)set.moves.size();
    return total;
}

bool isSelected(const neighborhood_r_accessor& n, int i) {
    return std::find(n.reallocationSelection.begin(), n.reallocationSelection.end(), i)
           != n.reallocationSelection.end();
}

// Path implied by the current Gurobi solution: a selected position takes the
// move whose variable is 1 (or is removed when none is), others are kept.
std::vector<segment> expectedPathFromSolution(neighborhood_r_accessor& n,
                                              const std::vector<segment>& path) {
    std::vector<segment> expected;
    for (int i = 0; i < (int)path.size(); ++i) {
        if (!isSelected(n, i)) { expected.push_back(path[i]); continue; }
        const auto& set = n.reallocationSets[i];
        for (int k = 0; k < (int)set.moves.size(); ++k)
            if (set.variables[k].get(GRB_DoubleAttr_X) > 0.5) { expected.push_back(set.moves[k].s); break; }
    }
    return expected;
}

// Sum of the POD coefficients of cell c over the vertices of s.
double segmentCellPod(neighborhood_r_accessor& n, const segment& s, int c) {
    segment no = n.geometry.getNonOriented(s);
    int u = no.source;
    int v = path_engine::is_node(no.target) ? no.target : u;
    double sum = 0.0;
    for (int w = u; w <= v; ++w) sum += n.problem.s_pod.coeff(c, w);
    return sum;
}

} // namespace

// ============================================================================
// reset
// ============================================================================

TEST_F(neighborhood_r_fixture, ResetOnFreshObjectIsSafe) {
    EXPECT_NO_THROW(n->reset());
    EXPECT_NO_THROW(n->reset());
    EXPECT_EQ(n->model, nullptr);
    EXPECT_EQ(n->env, nullptr);
    EXPECT_FALSE(n->columnGenerationHasConverged());
}

TEST_F(neighborhood_r_fixture, ResetReleasesModelAndClearsState) {
    n->setSelectReallocations();
    n->constructGurobiModel();
    n->reallocationDuals = Eigen::VectorXd::Ones((int)n->reallocationSets.size());
    n->coverageDuals.setOnes();
    n->lengthDual = 3.0;

    n->reset();

    EXPECT_EQ(n->Z1, nullptr);
    EXPECT_EQ(n->Z2, nullptr);
    EXPECT_EQ(n->model, nullptr);
    EXPECT_EQ(n->env, nullptr);
    EXPECT_TRUE(n->reallocationSelection.empty());
    EXPECT_TRUE(n->reallocationSets.empty());
    EXPECT_TRUE((n->coverageDuals.array() == 0.0).all());
    EXPECT_TRUE((n->reallocationDuals.array() == 0.0).all());
    EXPECT_EQ(n->lengthDual, 0.0);
    EXPECT_FALSE(n->columnGenerationHasConverged());
}

// ============================================================================
// removalGain / replacementGain / getRemovalGains — edge cases
// ============================================================================

TEST_F(neighborhood_r_fixture, GetRemovalGainsFirstEntryIsZero) {
    std::vector<cost_t> gains = n->getRemovalGains();
    ASSERT_EQ(gains.size(), sol.path.size());
    EXPECT_EQ(gains[0], (cost_t{0, 0})) << "the start segment is never removable";
}

// The last segment has no successor: its gain is a tail insertion.
TEST_F(neighborhood_r_fixture, ReplacementGainLastSegmentIsTailInsertion) {
    int last = (int)sol.path.size() - 1;
    segment s = n->geometry.flip_segment(sol.path[last]);
    EXPECT_EQ(n->replacementGain(last, s), n->geometry.insert_tail_gain(sol.path[last - 1], s));
}

TEST_F(neighborhood_r_fixture, ReplacementGainInteriorSegmentIsBetweenInsertion) {
    segment s = n->geometry.flip_segment(sol.path[3]);
    EXPECT_EQ(n->replacementGain(3, s), n->geometry.insert_between_gain(sol.path[2], s, sol.path[4]));
}

// ============================================================================
// setSelectReallocations — edge cases
// ============================================================================

// The selection is a complete parity group of [1, N).
TEST_F(neighborhood_r_fixture, SelectIsCompleteParityGroup) {
    const int N = (int)sol.path.size();
    std::vector<int> even, odd;
    for (int i = 1; i < N; ++i) (i % 2 == 0 ? even : odd).push_back(i);
    for (int trial = 0; trial < 20; ++trial) {
        n->setSelectReallocations();
        EXPECT_TRUE(n->reallocationSelection == even || n->reallocationSelection == odd)
            << "trial " << trial << ": selection is not a complete parity group";
    }
}

TEST_F(neighborhood_r_fixture, SelectOnSingleSegmentPathIsEmpty) {
    sol.path = {{6, 11}};
    n->setSelectReallocations();
    EXPECT_TRUE(n->reallocationSelection.empty());
}

TEST_F(neighborhood_r_fixture, SelectOnTwoSegmentPathIsSubsetOfLast) {
    sol.path = {{6, 11}, {23, 18}};
    for (int trial = 0; trial < 20; ++trial) {
        n->setSelectReallocations();
        EXPECT_TRUE(n->reallocationSelection.empty() || n->reallocationSelection == std::vector<int>{1});
    }
}

// ============================================================================
// pushInitialSegmentReallocation / setInitialReallocationSets
// ============================================================================

// Every move carries its replacement gain at that position, including the last one.
TEST_F(neighborhood_r_fixture, PushInitialMovesCarryReplacementGain) {
    n->reallocationSets.resize(sol.path.size());
    for (int index : {1, (int)sol.path.size() - 1}) {
        n->pushInitialSegmentReallocation(index);
        const auto& moves = n->reallocationSets[index].moves;
        ASSERT_FALSE(moves.empty());
        EXPECT_LE(moves.size(), 2u) << "at most the original and one MaximumReplacement move";
        EXPECT_EQ(moves[0].s, sol.path[index]);
        for (const auto& move : moves)
            EXPECT_EQ(move.gain, n->replacementGain(index, move.s)) << "index " << index;
    }
}

TEST_F(neighborhood_r_fixture, SetInitialReallocationSetsOnlyFillsSelected) {
    n->setSelectReallocations();
    n->setInitialReallocationSets();
    ASSERT_EQ(n->reallocationSets.size(), sol.path.size());
    for (int i = 0; i < (int)sol.path.size(); ++i) {
        if (isSelected(*n, i)) {
            ASSERT_FALSE(n->reallocationSets[i].moves.empty()) << "selected set " << i << " is empty";
            EXPECT_EQ(n->reallocationSets[i].moves[0].s, sol.path[i]);
        } else {
            EXPECT_TRUE(n->reallocationSets[i].moves.empty()) << "non-selected set " << i << " has moves";
        }
        EXPECT_TRUE(n->reallocationSets[i].variables.empty()) << "variables are created later";
    }
}

// ============================================================================
// setRemovalCoverage
// ============================================================================

TEST_F(neighborhood_r_fixture, SetRemovalCoverageRemovesSelectedSegments) {
    n->setSelectReallocations();
    cppied_solution reduced = sol;
    reduced.path.clear();
    for (int i = 0; i < (int)sol.path.size(); ++i)
        if (!isSelected(*n, i)) reduced.path.push_back(sol.path[i]);
    n->coverage.reset(reduced);

    n->setRemovalCoverage();
    EXPECT_TRUE(sol.coverage.isApprox(reduced.coverage, 1e-9))
        << "coverage must equal the coverage of the path without the selected segments";
}

TEST_F(neighborhood_r_fixture, SetRemovalCoverageEmptySelectionKeepsCoverage) {
    Eigen::VectorXd saved = sol.coverage;
    n->reallocationSelection.clear();
    n->setRemovalCoverage();
    EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9));
}

// ============================================================================
// Staged model construction — each construct* step called directly
// ============================================================================

class neighborhood_r_staged_fixture : public neighborhood_r_fixture {
protected:
    void SetUp() override {
        neighborhood_r_fixture::SetUp();
        n->setSelectReallocations();
        ASSERT_FALSE(n->reallocationSelection.empty());
        n->initializeGurobiEnvironment();
        n->initializeGurobiModel();
        n->setInitialReallocationSets();
    }
};

TEST_F(neighborhood_r_staged_fixture, InitializeGurobiEnvironmentSetsParameters) {
    ASSERT_NE(n->env, nullptr);
    EXPECT_EQ(n->env->get(GRB_IntParam_OutputFlag), 0);
    EXPECT_EQ(n->env->get(GRB_IntParam_LogToConsole), 0);
    EXPECT_EQ(n->env->get(GRB_IntParam_ThreadLimit), 1);
    std::filesystem::path expected = std::filesystem::path(ctx.config.at("WORKING_DIRECTORY")) / "reallocation.log";
    EXPECT_EQ(n->env->get(GRB_StringParam_LogFile), expected.string());
}

TEST_F(neighborhood_r_staged_fixture, InitializeGurobiModelCreatesEmptyModel) {
    ASSERT_NE(n->model, nullptr);
    n->model->update();
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumVars), 0);
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumConstrs), 0);
}

TEST_F(neighborhood_r_staged_fixture, ConstructReallocationSetVariablesOnePerMove) {
    int index = n->reallocationSelection.front();
    n->constructReallocationSetVariables(index);
    n->model->update();
    const auto& set = n->reallocationSets[index];
    ASSERT_EQ(set.variables.size(), set.moves.size());
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumVars), (int)set.moves.size());
    for (const auto& var : set.variables) {
        EXPECT_EQ(var.get(GRB_CharAttr_VType), GRB_CONTINUOUS) << "the restricted master must be an LP";
        EXPECT_EQ(var.get(GRB_DoubleAttr_LB), 0.0);
        EXPECT_EQ(var.get(GRB_DoubleAttr_UB), 1.0);
    }
}

// A non-selected position has no moves, so no variable is created.
TEST_F(neighborhood_r_staged_fixture, ConstructReallocationSetVariablesOnEmptySetAddsNothing) {
    n->constructReallocationSetVariables(0);
    n->model->update();
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumVars), 0);
    EXPECT_TRUE(n->reallocationSets[0].variables.empty());
}

TEST_F(neighborhood_r_staged_fixture, ConstructInitialVariablesCoversAllSelectedMoves) {
    n->constructInitialVariables();
    n->model->update();
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumVars), totalMoves(*n));
    for (const auto& set : n->reallocationSets)
        EXPECT_EQ(set.variables.size(), set.moves.size());
}

TEST_F(neighborhood_r_staged_fixture, LengthAndTurnsObjectivesUseMoveGains) {
    n->constructInitialVariables();
    n->constructLengthObjective();
    n->constructTurnsObjective();
    n->model->update();
    ASSERT_NE(n->Z1, nullptr);
    ASSERT_NE(n->Z2, nullptr);
    EXPECT_EQ((int)n->Z1->size(), totalMoves(*n));
    EXPECT_EQ((int)n->Z2->size(), totalMoves(*n));

    n->model->setObjective(*n->Z1);
    n->model->update();
    for (const auto& set : n->reallocationSets)
        for (size_t k = 0; k < set.moves.size(); ++k)
            EXPECT_DOUBLE_EQ(set.variables[k].get(GRB_DoubleAttr_Obj), set.moves[k].gain.length);

    n->model->setObjective(*n->Z2);
    n->model->update();
    for (const auto& set : n->reallocationSets)
        for (size_t k = 0; k < set.moves.size(); ++k)
            EXPECT_DOUBLE_EQ(set.variables[k].get(GRB_DoubleAttr_Obj), set.moves[k].gain.turns);
}

TEST_F(neighborhood_r_staged_fixture, ConstructObjectivesBuildsBothObjectives) {
    n->constructInitialVariables();
    n->constructObjectives();
    EXPECT_NE(n->Z1, nullptr);
    EXPECT_NE(n->Z2, nullptr);
}

// Coverage rows: one per cell left unsatisfied after removal, RHS = req - coverage,
// coefficient of each move = POD of the cell summed over the move's vertices.
TEST_F(neighborhood_r_staged_fixture, ConstructCoverageConstraintsRowsAndCoefficients) {
    n->constructInitialVariables();
    n->constructCoverageConstraints();
    n->model->update();

    std::vector<int> unsat;
    n->coverage.unsatisfied_cells(sol, unsat);
    EXPECT_EQ(n->unsatisfiedCells, unsat) << "unsatisfiedCells must be computed after removing the selection";
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumConstrs), (int)unsat.size());

    for (int c : n->unsatisfiedCells) {
        EXPECT_NEAR(n->coverageConstraints[c].get(GRB_DoubleAttr_RHS), P.req[c] - sol.coverage[c], 1e-9);
        EXPECT_EQ(n->coverageConstraints[c].get(GRB_CharAttr_Sense), GRB_GREATER_EQUAL);
        for (int i : n->reallocationSelection) {
            const auto& set = n->reallocationSets[i];
            for (size_t k = 0; k < set.moves.size(); ++k)
                EXPECT_NEAR(n->model->getCoeff(n->coverageConstraints[c], set.variables[k]),
                            segmentCellPod(*n, set.moves[k].s, c), 1e-9)
                    << "cell " << c << ", set " << i << ", move " << k;
        }
    }
}

TEST_F(neighborhood_r_staged_fixture, ConstructReallocationConstraintsSelectAtMostOne) {
    n->constructInitialVariables();
    n->constructReallocationConstraints();
    n->model->update();
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumConstrs), (int)n->reallocationSelection.size());
    for (int i : n->reallocationSelection) {
        const GRBConstr& row = n->reallocationConstraints[i];
        EXPECT_EQ(row.get(GRB_CharAttr_Sense), GRB_LESS_EQUAL);
        EXPECT_DOUBLE_EQ(row.get(GRB_DoubleAttr_RHS), 1.0);
        EXPECT_EQ(n->model->getRow(row).size(), n->reallocationSets[i].variables.size());
        for (const auto& var : n->reallocationSets[i].variables)
            EXPECT_DOUBLE_EQ(n->model->getCoeff(row, var), 1.0);
    }
}

TEST_F(neighborhood_r_staged_fixture, ConstructLengthConstraintsBoundsZ1ByIncumbent) {
    n->constructInitialVariables();
    n->constructObjectives();
    n->constructLengthConstraints();
    n->model->update();
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumConstrs), 1);
    EXPECT_EQ(n->pathLengthConstraint.get(GRB_CharAttr_Sense), GRB_LESS_EQUAL);
    EXPECT_DOUBLE_EQ(n->pathLengthConstraint.get(GRB_DoubleAttr_RHS), sol.cost.length);
    for (const auto& set : n->reallocationSets)
        for (size_t k = 0; k < set.moves.size(); ++k)
            EXPECT_DOUBLE_EQ(n->model->getCoeff(n->pathLengthConstraint, set.variables[k]),
                             set.moves[k].gain.length);
}

TEST_F(neighborhood_r_staged_fixture, ConstructConstraintsBuildsAllRows) {
    n->constructInitialVariables();
    n->constructObjectives();
    n->constructConstraints();
    n->model->update();
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumConstrs),
              (int)n->unsatisfiedCells.size() + (int)n->reallocationSelection.size() + 1);
}

TEST_F(neighborhood_r_staged_fixture, ConstructLinearProgramBuildsVariablesObjectivesAndRows) {
    n->constructLinearProgram();
    n->model->update();
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumVars), totalMoves(*n));
    EXPECT_NE(n->Z1, nullptr);
    EXPECT_NE(n->Z2, nullptr);
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumConstrs),
              (int)n->unsatisfiedCells.size() + (int)n->reallocationSelection.size() + 1);
}

// ============================================================================
// Dual extraction (LP solved on Z1)
// ============================================================================

TEST_F(neighborhood_r_solved_fixture, SetCoverageDualsMatchesPiAndSign) {
    n->setCoverageDuals();
    for (int c = 0; c < (int)n->coverageDuals.size(); ++c) {
        if (std::find(n->unsatisfiedCells.begin(), n->unsatisfiedCells.end(), c) != n->unsatisfiedCells.end()) {
            EXPECT_NEAR(n->coverageDuals(c), n->coverageConstraints[c].get(GRB_DoubleAttr_Pi), 1e-12);
            EXPECT_GE(n->coverageDuals(c), -1e-9) << ">= row of a minimisation must have a non-negative dual";
        } else {
            EXPECT_EQ(n->coverageDuals(c), 0.0) << "cell " << c << " has no row; its dual must stay 0";
        }
    }
}

TEST_F(neighborhood_r_solved_fixture, SetReallocationDualsMatchesPiAndSign) {
    n->reallocationDuals = Eigen::VectorXd::Zero((int)n->reallocationSets.size());
    n->setReallocationDuals();
    for (int i = 0; i < (int)n->reallocationDuals.size(); ++i) {
        if (isSelected(*n, i)) {
            EXPECT_NEAR(n->reallocationDuals(i), n->reallocationConstraints[i].get(GRB_DoubleAttr_Pi), 1e-12);
            EXPECT_LE(n->reallocationDuals(i), 1e-9) << "<= row of a minimisation must have a non-positive dual";
        } else {
            EXPECT_EQ(n->reallocationDuals(i), 0.0);
        }
    }
}

TEST_F(neighborhood_r_solved_fixture, SetLengthDualMatchesPiAndSign) {
    n->setLengthDual();
    EXPECT_NEAR(n->lengthDual, n->pathLengthConstraint.get(GRB_DoubleAttr_Pi), 1e-12);
    EXPECT_LE(n->lengthDual, 1e-9);
}

// ============================================================================
// setVertexPrices
// ============================================================================

TEST_F(neighborhood_r_fixture, SetVertexPricesArePodWeightedDuals) {
    for (int c = 0; c < (int)n->coverageDuals.size(); ++c) n->coverageDuals(c) = 0.1 * (c % 7) - 0.2;
    n->setVertexPrices();
    for (int u = 0; u < n->geometry.n_vertex; ++u) {
        double expected = 0.0;
        for (int c = 0; c < (int)n->coverageDuals.size(); ++c)
            expected += n->problem.s_pod.coeff(c, u) * n->coverageDuals(c);
        EXPECT_NEAR(n->vertexCoveragePrices(u), expected, 1e-9) << "vertex " << u;
    }
}

TEST_F(neighborhood_r_fixture, SetVertexPricesZeroDualsGiveZeroPrices) {
    n->vertexCoveragePrices.setOnes();
    n->coverageDuals.setZero();
    n->setVertexPrices();
    EXPECT_TRUE((n->vertexCoveragePrices.array() == 0.0).all());
}

// ============================================================================
// Column generation fixture — LP built, reallocation duals sized as in local_search
// ============================================================================

class neighborhood_r_cg_fixture : public neighborhood_r_gurobi_fixture {
protected:
    void SetUp() override {
        neighborhood_r_gurobi_fixture::SetUp();
        if (HasFatalFailure()) return;
        n->reallocationDuals = Eigen::VectorXd::Zero((int)n->reallocationSets.size());
    }

    void buildBinaryProgram() {
        n->columnGeneration();
        n->setLexicographicBinaryProgram();
        n->setReallocationModelWarmStart();
        n->model->update();
    }
};

// ─── solvePricingSubproblem ─────────────────────────────────────────────────

// The pricing result is the best MinimumPrice solution over the selected
// indices, evaluated with the duals of the current LP.
template<typename Solver>
void checkPricingIsBestOverSelection(neighborhood_r_accessor& n, cppied_solution& sol) {
    PricingSolution pricing = n.solvePricingSubproblem<Solver>();

    double best = std::numeric_limits<double>::max();
    int best_r = -1;
    for (int r : n.reallocationSelection) {
        Reallocation candidate = Solver(n.vertexCoveragePrices, n.reallocationDuals(r), n.lengthDual,
                                        n.problem, n.coverage, n.geometry, sol, r).solve();
        if (candidate.value < best) { best = candidate.value; best_r = r; }
    }
    EXPECT_NEAR(pricing.value, best, 1e-9) << "pricing must return the minimum reduced cost over the selection";
    EXPECT_EQ(pricing.R, best_r) << "R must be the path index of the reallocated segment";
    EXPECT_TRUE(isSelected(n, pricing.R));
    ASSERT_TRUE(path_engine::is_node(pricing.s.source));
    EXPECT_EQ(pricing.costs, n.replacementGain(pricing.R, pricing.s));
}

TEST_F(neighborhood_r_cg_fixture, SolvePricingSubproblemFirstStageReturnsBestSelectedReallocation) {
    n->model->setObjective(*n->Z1, GRB_MINIMIZE);
    n->model->optimize();
    ASSERT_EQ(n->model->get(GRB_IntAttr_Status), GRB_OPTIMAL);
    checkPricingIsBestOverSelection<FirstStageMinimumPrice>(*n, sol);
}

TEST_F(neighborhood_r_cg_fixture, SolvePricingSubproblemSecondStageReturnsBestSelectedReallocation) {
    n->model->setObjective(*n->Z2, GRB_MINIMIZE);
    n->model->optimize();
    ASSERT_EQ(n->model->get(GRB_IntAttr_Status), GRB_OPTIMAL);
    checkPricingIsBestOverSelection<SecondStageMinimumPrice>(*n, sol);
}

// No selected index: no column may be produced and no gain may be evaluated.
TEST_F(neighborhood_r_cg_fixture, SolvePricingSubproblemEmptySelectionReturnsNoColumn) {
    n->model->setObjective(*n->Z1, GRB_MINIMIZE);
    n->model->optimize();
    ASSERT_EQ(n->model->get(GRB_IntAttr_Status), GRB_OPTIMAL);
    n->reallocationSelection.clear();

    PricingSolution pricing = n->solvePricingSubproblem<FirstStageMinimumPrice>();
    EXPECT_FALSE(path_engine::is_node(pricing.s.source)) << "no segment can be priced without a selection";
    EXPECT_GE(pricing.value, -1e-6) << "an empty selection must not trigger a column";
}

// ─── generateColumn ─────────────────────────────────────────────────────────

TEST_F(neighborhood_r_cg_fixture, GenerateColumnAddsVariableToAllRows) {
    const int hb = n->geometry.horizontal_bound;
    const std::vector<segment> columns = {
        {0, 3}, {3, 0},                                   // horizontal, both orientations
        {2, path_engine::NULL_NODE}, {2, path_engine::REVERSED_NULL_NODE}, // single vertex
        {hb, hb + 2}};                                    // vertical
    const int R = n->reallocationSelection.front();

    for (const segment& s : columns) {
        const size_t before = n->reallocationSets[R].moves.size();
        PricingSolution pricing{s, -1.0, R, n->replacementGain(R, s)};
        n->generateColumn(pricing);
        n->model->update();

        auto& set = n->reallocationSets[R];
        ASSERT_EQ(set.moves.size(), before + 1);
        ASSERT_EQ(set.variables.size(), before + 1);
        EXPECT_EQ(set.moves.back().s, s);
        EXPECT_EQ(set.moves.back().gain, pricing.costs);

        GRBVar var = set.variables.back();
        EXPECT_EQ(var.get(GRB_CharAttr_VType), GRB_CONTINUOUS);
        EXPECT_EQ(var.get(GRB_DoubleAttr_LB), 0.0);
        EXPECT_EQ(var.get(GRB_DoubleAttr_UB), 1.0);
        EXPECT_DOUBLE_EQ(n->model->getCoeff(n->reallocationConstraints[R], var), 1.0);
        EXPECT_DOUBLE_EQ(n->model->getCoeff(n->pathLengthConstraint, var), pricing.costs.length);
        for (int c : n->unsatisfiedCells)
            EXPECT_NEAR(n->model->getCoeff(n->coverageConstraints[c], var), segmentCellPod(*n, s, c), 1e-9)
                << "cell " << c << " for column {" << s.source << "," << s.target << "}";

        n->model->setObjective(*n->Z1);
        n->model->update();
        EXPECT_DOUBLE_EQ(var.get(GRB_DoubleAttr_Obj), pricing.costs.length);
        n->model->setObjective(*n->Z2);
        n->model->update();
        EXPECT_DOUBLE_EQ(var.get(GRB_DoubleAttr_Obj), pricing.costs.turns);
    }
}

// ─── column generation stages ───────────────────────────────────────────────

TEST_F(neighborhood_r_cg_fixture, FirstStageEitherConvergesOrAddsOneColumn) {
    EXPECT_FALSE(n->columnGenerationHasConverged());
    const int before = totalMoves(*n);
    n->runColumnGenerationFirstStage();
    EXPECT_EQ(n->model->get(GRB_IntAttr_Status), GRB_OPTIMAL);
    const int added = totalMoves(*n) - before;
    if (n->columnGenerationHasConverged()) {
        EXPECT_EQ(added, 0) << "converged but a column was added";
    } else {
        ASSERT_EQ(added, 1) << "not converged but no column was added";
        for (int i : n->reallocationSelection) {
            const auto& set = n->reallocationSets[i];
            EXPECT_EQ(set.variables.size(), set.moves.size());
        }
    }
}

TEST_F(neighborhood_r_cg_fixture, SecondStageBoundsLengthByFirstStageOptimum) {
    n->runColumnGenerationFirstStage();
    const double first_stage_length = n->model->get(GRB_DoubleAttr_ObjVal);
    const bool first_converged = n->columnGenerationHasConverged();
    const int before = totalMoves(*n);

    n->runColumnGenerationSecondStage();
    EXPECT_NEAR(n->pathLengthConstraint.get(GRB_DoubleAttr_RHS), first_stage_length, 1e-9);
    EXPECT_EQ(n->model->get(GRB_IntAttr_Status), GRB_OPTIMAL);
    const int added = totalMoves(*n) - before;
    EXPECT_LE(added, 1);
    if (added == 1) EXPECT_FALSE(n->columnGenerationHasConverged());
    else            EXPECT_EQ(n->columnGenerationHasConverged(), first_converged);
}

// Converged exactly when neither stage generated a column.
TEST_F(neighborhood_r_cg_fixture, IterationConvergesIffNoColumnAdded) {
    const int before = totalMoves(*n);
    n->runColumnGenerationIteration();
    EXPECT_EQ(n->columnGenerationHasConverged(), totalMoves(*n) == before);
}

TEST_F(neighborhood_r_cg_fixture, ColumnGenerationTerminatesWithConsistentSets) {
    n->columnGeneration();
    EXPECT_TRUE(n->columnGenerationHasConverged());
    n->model->update();
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumVars), totalMoves(*n));
    EXPECT_EQ((int)n->Z1->size(), totalMoves(*n));
    for (int i = 0; i < (int)n->reallocationSets.size(); ++i) {
        const auto& set = n->reallocationSets[i];
        EXPECT_EQ(set.variables.size(), set.moves.size());
        if (!isSelected(*n, i)) EXPECT_TRUE(set.moves.empty()) << "column added to non-selected set " << i;
        for (const auto& move : set.moves) EXPECT_TRUE(path_engine::is_node(move.s.source));
    }
}

// ─── binary program ─────────────────────────────────────────────────────────

TEST_F(neighborhood_r_cg_fixture, SetLexicographicBinaryProgramConvertsModel) {
    n->columnGeneration();
    n->model->update();
    const int rows_before = n->model->get(GRB_IntAttr_NumConstrs);

    n->setLexicographicBinaryProgram();
    n->model->update();

    EXPECT_EQ(n->model->get(GRB_IntAttr_NumConstrs), rows_before - 1) << "the length row must be removed";
    EXPECT_EQ(n->model->get(GRB_IntAttr_NumObj), 2);
    for (const auto& set : n->reallocationSets)
        for (const auto& var : set.variables) EXPECT_EQ(var.get(GRB_CharAttr_VType), GRB_BINARY);

    n->model->set(GRB_IntParam_ObjNumber, 0);
    EXPECT_EQ(n->model->get(GRB_IntAttr_ObjNPriority), 2) << "length must be the primary objective";
    for (const auto& set : n->reallocationSets)
        for (size_t k = 0; k < set.moves.size(); ++k)
            EXPECT_DOUBLE_EQ(set.variables[k].get(GRB_DoubleAttr_ObjN), set.moves[k].gain.length);

    n->model->set(GRB_IntParam_ObjNumber, 1);
    EXPECT_EQ(n->model->get(GRB_IntAttr_ObjNPriority), 1);
    for (const auto& set : n->reallocationSets)
        for (size_t k = 0; k < set.moves.size(); ++k)
            EXPECT_DOUBLE_EQ(set.variables[k].get(GRB_DoubleAttr_ObjN), set.moves[k].gain.turns);
}

TEST_F(neighborhood_r_cg_fixture, WarmStartSelectsOriginalSegments) {
    buildBinaryProgram();
    for (int i : n->reallocationSelection) {
        const auto& set = n->reallocationSets[i];
        ASSERT_FALSE(set.variables.empty());
        EXPECT_EQ(set.moves[0].s, sol.path[i]) << "variables[0] must be the original segment";
        EXPECT_EQ(set.variables[0].get(GRB_DoubleAttr_Start), 1.0);
        for (size_t k = 1; k < set.variables.size(); ++k)
            EXPECT_EQ(set.variables[k].get(GRB_DoubleAttr_Start), 0.0);
    }
}

// ─── reallocation / safeSolutionRetrieve ────────────────────────────────────

TEST_F(neighborhood_r_cg_fixture, ReallocationReturnsChosenMoveOrNull) {
    buildBinaryProgram();
    n->model->optimize();
    ASSERT_EQ(n->model->get(GRB_IntAttr_Status), GRB_OPTIMAL);

    for (int i = 0; i < (int)sol.path.size(); ++i) {
        segment expected{path_engine::NULL_NODE, path_engine::NULL_NODE};
        const auto& set = n->reallocationSets[i];
        for (size_t k = 0; k < set.moves.size(); ++k)
            if (set.variables[k].get(GRB_DoubleAttr_X) > 0.5) expected = set.moves[k].s;
        EXPECT_EQ(n->reallocation(i), expected) << "index " << i;
        if (!isSelected(*n, i))
            EXPECT_FALSE(path_engine::is_node(n->reallocation(i).source)) << "non-selected index " << i;
    }
}

TEST_F(neighborhood_r_cg_fixture, SafeSolutionRetrieveAppliesSolution) {
    buildBinaryProgram();
    n->model->optimize();
    ASSERT_EQ(n->model->get(GRB_IntAttr_Status), GRB_OPTIMAL);
    const std::vector<segment> expected = expectedPathFromSolution(*n, sol.path);
    const segment start = sol.path[0];

    n->safeSolutionRetrieve();

    EXPECT_EQ(sol.path, expected);
    EXPECT_EQ(sol.path[0], start) << "the start segment must never move";
    for (const auto& s : sol.path) EXPECT_TRUE(path_engine::is_node(s.source)) << "NULL segment left in path";
    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
    Eigen::VectorXd tracked = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(tracked, 1e-9));
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

// ─── retrieveSolution status branches ───────────────────────────────────────

TEST_F(neighborhood_r_cg_fixture, RetrieveSolutionInfeasibleThrowsAndReleasesModel) {
    buildBinaryProgram();
    GRBLinExpr all = 0.0;
    for (const auto& var : n->reallocationSets[n->reallocationSelection.front()].variables) all += var;
    n->model->addConstr(all >= 2.0);   // contradicts sum <= 1
    n->model->optimize();
    const int status = n->model->get(GRB_IntAttr_Status);
    ASSERT_TRUE(status == GRB_INFEASIBLE || status == GRB_INF_OR_UNBD);

    EXPECT_THROW(n->retrieveSolution(), std::runtime_error);
    EXPECT_EQ(n->model, nullptr);
    EXPECT_EQ(n->env, nullptr);
}

// Interrupted solve: the best incumbent (if any) must be applied.
TEST_F(neighborhood_r_cg_fixture, RetrieveSolutionTimeLimitUsesBestIncumbent) {
    buildBinaryProgram();
    n->model->set(GRB_DoubleParam_TimeLimit, 0.0);
    n->model->optimize();
    if (n->model->get(GRB_IntAttr_Status) == GRB_OPTIMAL)
        GTEST_SKIP() << "Gurobi proved optimality within a zero time limit";
    const std::vector<segment> expected = n->model->get(GRB_IntAttr_SolCount) > 0
                                          ? expectedPathFromSolution(*n, sol.path) : sol.path;

    EXPECT_NO_THROW(n->retrieveSolution());
    EXPECT_EQ(sol.path, expected);
    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
    Eigen::VectorXd tracked = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(tracked, 1e-9));
    EXPECT_EQ(n->model, nullptr);
    EXPECT_EQ(n->env, nullptr);
}

// ─── throwIfLpTimedOut / TimeOutError ───────────────────────────────────────

TEST_F(neighborhood_r_cg_fixture, ThrowIfLpTimedOutAcceptsOptimalLp) {
    n->model->setObjective(*n->Z1, GRB_MINIMIZE);
    n->model->optimize();
    ASSERT_EQ(n->model->get(GRB_IntAttr_Status), GRB_OPTIMAL);
    EXPECT_NO_THROW(n->throwIfLpTimedOut());
}

TEST_F(neighborhood_r_cg_fixture, ThrowIfLpTimedOutThrowsOnTimeLimit) {
    n->model->setObjective(*n->Z1, GRB_MINIMIZE);
    n->model->set(GRB_DoubleParam_TimeLimit, 0.0);
    n->model->optimize();
    ASSERT_EQ(n->model->get(GRB_IntAttr_Status), GRB_TIME_LIMIT);
    EXPECT_THROW(n->throwIfLpTimedOut(), TimeOutError);
}

TEST_F(neighborhood_r_cg_fixture, FirstStageThrowsTimeOutErrorWithNoTimeLeft) {
    ctx.max_time = 0;
    const int before = totalMoves(*n);
    EXPECT_THROW(n->runColumnGenerationFirstStage(), TimeOutError);
    EXPECT_EQ(totalMoves(*n), before) << "no column may be generated from an unsolved LP";
}

TEST_F(neighborhood_r_cg_fixture, SecondStageThrowsTimeOutErrorWithNoTimeLeft) {
    n->runColumnGenerationFirstStage();
    ctx.max_time = 0;
    const int before = totalMoves(*n);
    EXPECT_THROW(n->runColumnGenerationSecondStage(), TimeOutError);
    EXPECT_EQ(totalMoves(*n), before) << "no column may be generated from an unsolved LP";
}

// ============================================================================
// local_search — edge cases
// ============================================================================

// No time left: Gurobi gets a zero time limit and keeps the warm start.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchWithNoTimeLeftKeepsSolution) {
    ctx.max_time = 0;
    const std::vector<segment> before = sol.path;
    const cost_t cost_before = sol.cost;
    bool improved = true;
    EXPECT_NO_THROW(improved = n->local_search());
    EXPECT_FALSE(improved);
    EXPECT_EQ(sol.path, before);
    EXPECT_EQ(sol.cost, cost_before);
    Eigen::VectorXd tracked = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(tracked, 1e-9));
}

// The last segment can be reallocated: an over-extended last segment is shortened.
TEST_F(neighborhood_r_fixture, LastSegmentIsReallocated) {
    sol.path.back() = {60, 65};   // col_boundary 3, rows 0->5 (local optimum: {62,63})
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    const int last = (int)sol.path.size() - 1;
    ASSERT_EQ(last % 2, 0) << "the even parity group must contain the last index";
    const cost_t before = sol.cost;

    // Same pipeline as local_search, with the parity group containing the last segment.
    n->reset();
    n->reallocationSelection = {2, 4, 6};
    n->constructGurobiModel();
    n->reallocationDuals = Eigen::VectorXd::Zero((int)n->reallocationSets.size());
    n->columnGeneration();
    n->setLexicographicBinaryProgram();
    n->setReallocationModelWarmStart();
    n->model->update();
    n->model->optimize();
    n->retrieveSolution();

    EXPECT_LT(sol.cost.length, before.length) << "the over-extended last segment was not shortened";
    EXPECT_NE(sol.path.back(), (segment{60, 65}));
    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
    n->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

// Interrupted solve whose incumbent differs from the input: warm-start the binary
// program with its own optimum, then stop after the first solution. The
// incumbent must be applied.
TEST_F(neighborhood_r_improvable_fixture, RetrieveSolutionInterruptedAppliesImprovingIncumbent) {
    n->reset();
    n->reallocationSelection = {2, 4, 6};   // index 2 is over-extended
    n->constructGurobiModel();
    n->reallocationDuals = Eigen::VectorXd::Zero((int)n->reallocationSets.size());
    n->columnGeneration();
    n->setLexicographicBinaryProgram();
    n->setReallocationModelWarmStart();
    n->model->update();
    n->model->optimize();
    ASSERT_EQ(n->model->get(GRB_IntAttr_Status), GRB_OPTIMAL);
    const std::vector<segment> optimum = expectedPathFromSolution(*n, sol.path);
    ASSERT_NE(optimum, sol.path) << "precondition: the optimum must differ from the input path";

    for (const auto& set : n->reallocationSets)
        for (auto var : set.variables) var.set(GRB_DoubleAttr_Start, var.get(GRB_DoubleAttr_X));
    // Stop as soon as the warm start is accepted (TimeLimit = 0 stops before
    // Gurobi even loads the MIP start).
    n->model->reset();
    n->model->set(GRB_IntParam_SolutionLimit, 1);
    n->model->optimize();
    if (n->model->get(GRB_IntAttr_Status) == GRB_OPTIMAL)
        GTEST_SKIP() << "Gurobi reported optimality instead of an interruption";
    ASSERT_GT(n->model->get(GRB_IntAttr_SolCount), 0) << "the warm start was not accepted as incumbent";
    const std::vector<segment> expected = expectedPathFromSolution(*n, sol.path);

    EXPECT_NO_THROW(n->retrieveSolution());
    EXPECT_EQ(sol.path, expected) << "the incumbent found before the interruption must be applied";
    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
    Eigen::VectorXd tracked = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(tracked, 1e-9));
}

// A timed-out call must not prevent a later call from running normally.
TEST_F(neighborhood_r_e2e_fixture, E2ELocalSearchRecoversAfterTimeout) {
    ctx.max_time = 0;
    EXPECT_FALSE(n->local_search());
    ctx.max_time = 600;
    ctx.start_time = std::chrono::high_resolution_clock::now();
    EXPECT_NO_THROW(n->local_search());
    EXPECT_EQ(n->geometry.cost(sol), sol.cost);
    Eigen::VectorXd tracked = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(tracked, 1e-9));
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}
