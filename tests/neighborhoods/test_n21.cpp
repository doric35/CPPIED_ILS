#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_n21.hpp"

struct neighborhood_n21_accessor : public neighborhood_n21 {
public:
    using neighborhood_n21::neighborhood_n21;

    using neighborhood::geometry;
    using neighborhood::coverage;
    using neighborhood::problem;

    using neighborhood_n21::setMovesAndGains;
    using neighborhood_n21::candidatesMoves;
    using neighborhood_n21::transformToInterval;
    using neighborhood_n21::spacingConstraint;
};

class neighborhood_n21_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<neighborhood_n21_accessor> n;
    cppied_solution sol;

    // Neighborhoods are bound to a solution at construction.
    bool search(cppied_solution& s){
        return neighborhood_n21_accessor(ctx, P, s).local_search();
    }

    void SetUp() override{
        cppied_context_fixture::SetUp();
        sol.path =  {{6, 11},
                     {23,18},
                     {44,43},
                     {6, path_engine::NULL_NODE},
                     {48, 51},
                     {30, 35},
                     {75,73},
                     {62, 63}};
        sol.cost = {0,0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        n = std::make_unique<neighborhood_n21_accessor>(ctx, P, sol);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

TEST_F(neighborhood_n21_fixture, ImpromentFound){
    cost_t c_0 = sol.cost;
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    bool improved = search(sol);
    ASSERT_EQ(n->geometry.cost(sol), sol.cost);
    n->coverage.reset(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    EXPECT_TRUE(improved);
    EXPECT_LT(sol.cost, c_0);
}

TEST_F(neighborhood_n21_fixture, ValidMarginalGainNoImprovement){
    sol.path = {{6, 11},
                {23,18},
                {44,42},
                {48, 51},
                {30, 33},
                {77, 76},
                {75,73},
                {62, 63}};
    sol.cost = n->geometry.cost(sol);
    n->coverage.reset(sol);
    search(sol);
    EXPECT_EQ(n->geometry.cost(sol),sol.cost);
}

TEST_F(neighborhood_n21_fixture, LocalOptimaFound){
    sol.path = {{6, 11},
                {23,18},
                {44,42},
                {48, 51},
                {30, 35},
                {75,73},
                {62, 63}};
    n->coverage.reset(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    sol.cost = n->geometry.cost(sol);
    cost_t c = sol.cost;
    bool improved = search(sol);
    ASSERT_EQ(n->geometry.cost(sol), sol.cost);
    n->coverage.reset(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    EXPECT_FALSE(improved);
    EXPECT_EQ(sol.cost, c);
}

TEST_F(neighborhood_n21_fixture, ValidCoverageUpdate){
    ASSERT_TRUE(search(sol));
    Eigen::VectorXd C = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(C, 1e-9));
}

TEST_F(neighborhood_n21_fixture, CoverageConstraintSatisfied){
    ASSERT_TRUE(search(sol));
    n->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

TEST_F(neighborhood_n21_fixture, LocalOptimaAfterImprovement){
    ASSERT_TRUE(search(sol));
    EXPECT_FALSE(search(sol));
}

//-----------Test specific cases-----------------------

// ── Direct-removal fast path ──────────────────────────────────────────────────
//
// When coverage.can_remove() is true for a segment, replace() must take the
// fast path (no explore_replacements call) and return a trial that:
//   · carries the removal gain of the segment
//   · has no replacement segment  (s == {path_engine::NULL_NODE, path_engine::NULL_NODE})
//   · has no "other" partner      (other == -1)
//   · has no insertion position   (position == -1)
//
// The test solution is built on the known 7-segment local optimum used in
// LocalOptimaFound, extended with a single-node segment {6, path_engine::NULL_NODE}.
// Vertex 6 is already visited by sweep {6, 11}, so its coverage contribution
// is completely redundant and the segment may be removed outright.

// Helper fixture for the direct-removal case.
class neighborhood_n21_direct_removal_fixture : public neighborhood_n21_fixture {
protected:
    // 8-segment path: the 7-segment local optimum with {6, path_engine::NULL_NODE} inserted
    // between {44,42} and {48,51}.  Vertex 6 is already covered by {6, 11}.
    static constexpr int REDUNDANT_IDX = 3; // 0-based index of {6, path_engine::NULL_NODE}

    void SetUp() override {
        neighborhood_n21_fixture::SetUp();
        sol.path = {{6, 11},
                    {23, 18},
                    {44, 42},
                    {6, path_engine::NULL_NODE},   // redundant: vertex 6 already in {6,11}
                    {48, 51},
                    {30, 35},
                    {75, 73},
                    {62, 63}};
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

// MaximumTwoOneReallocation must return the direct-removal move when the segment is directly removable.
TEST_F(neighborhood_n21_direct_removal_fixture, DirectRemoval_ReplaceReturnsFastPathTrial) {
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());

    // Precondition: the redundant segment is indeed removable without a replacement.
    ASSERT_TRUE(n->coverage.can_remove(sol, sol.path[REDUNDANT_IDX]));

    auto it = std::next(sol.path.cbegin(), REDUNDANT_IDX);
    const cost_t expected_gain = n->geometry.removal_gain(sol, it);

    // The removal gain must be positive (detour through vertex 6 is costly).
    ASSERT_GT(expected_gain, (cost_t{0, 0}));

    // Solve the move search – path and coverage must be left unmodified.
    Eigen::VectorXd coverage_before = sol.coverage;
    std::vector<segment> path_before = sol.path;

    MaximumTwoOneReallocation algorithm(n->problem, n->coverage, n->geometry, sol, REDUNDANT_IDX);
    TwoOneReallocationMove t = algorithm.solve();

    // Path and coverage are restored after the call.
    EXPECT_EQ(sol.path, path_before);
    EXPECT_TRUE(sol.coverage.isApprox(coverage_before, 1e-9));

    // Trial fields match the fast-path contract.
    EXPECT_EQ(t.gain,     expected_gain);
    EXPECT_FALSE(path_engine::is_node(t.replacement.source));   // no replacement segment
    EXPECT_FALSE(path_engine::is_node(t.replacement.target));
    EXPECT_EQ(t.removalIndex,      -1);  // no second segment to remove
    EXPECT_EQ(t.reallocationIndex, -1);  // no insertion position
}

// local_search() must find improvement by removing the redundant segment.
TEST_F(neighborhood_n21_direct_removal_fixture, DirectRemoval_LocalSearchFindsImprovement) {
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    const cost_t cost_before = sol.cost;

    const bool improved = search(sol);

    EXPECT_TRUE(improved);
    EXPECT_LT(sol.cost, cost_before);
}

// The stored cost must equal the recomputed cost after the move.
TEST_F(neighborhood_n21_direct_removal_fixture, DirectRemoval_CostConsistentAfterMove) {
    search(sol);
    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
}

// Coverage must remain valid (all cells at or above their requirements).
TEST_F(neighborhood_n21_direct_removal_fixture, DirectRemoval_CoverageRemainsValid) {
    search(sol);
    n->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

// The incremental coverage vector must match a full reset.
TEST_F(neighborhood_n21_direct_removal_fixture, DirectRemoval_CoverageVectorConsistent) {
    search(sol);
    const Eigen::VectorXd coverage_after = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(coverage_after, 1e-9));
}

// After removing the redundant segment we must reach the known 7-segment local
// optimum; a second call to local_search() must therefore report no improvement.
TEST_F(neighborhood_n21_direct_removal_fixture, DirectRemoval_ReachesLocalOptimum) {
    ASSERT_TRUE(search(sol));  // first call removes the redundant segment
    EXPECT_FALSE(search(sol)); // second call: already at local optimum
}

TEST_F(neighborhood_n21_fixture, SpecialCaseCoverageOverlap){
    sol.path = {
            {6, -1},
    {48, 49},
    {12, -2},
    {18, 19},
    {62, 63},
    {75, 73},
    {66, -2},
    {72, 73},
    {74, 76},
    {77, -1},
    {65, 63},
    {62, 61},
    {60, -2},
    {48, -1},
    {49, 52},
    {53, -1}
    };
    sol.cost = n->geometry.cost(sol);
    n->coverage.reset(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    search(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}
// ============================================================================
// Interval independence of selected moves
//
// The interval scheduler co-selects two moves of the same orientation when
// they lie on the same line with disjoint intervals, or on lines at least
// `spacing` apart. Any such pair must be jointly feasible. Each move removes
// path[i] and, for a paired move, path[removalIndex]; its interval must
// therefore cover the paired removal's footprint on the line.
// Regression: getHorizontalOffset/getVerticalOffset ignored their second
// argument, so the interval never reached the paired removal; two moves each
// feasible alone were selected together and broke coverage (nested ->
// n21_search -> reallocation LP infeasible in ils_full DSolveFull*).
//
// The paths below are the n21 inputs on which that happened (captured from
// the ils_full fixture runs).
// ============================================================================

class neighborhood_n21_independence_fixture : public cppied_context_fixture,
                                              public ::testing::WithParamInterface<std::vector<segment>> {
protected:
    cppied_solution sol;
    std::unique_ptr<neighborhood_n21_accessor> m;

    void SetUp() override {
        sol.path = GetParam();
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        m = std::make_unique<neighborhood_n21_accessor>(ctx, P, sol);
        m->coverage.reset(sol);
        sol.cost = m->geometry.cost(sol);
        ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all()) << "captured input must be feasible";
    }

    bool jointlyFeasible(const std::vector<int>& moves) {
        cppied_solution t = sol;
        std::vector<int> drop;
        for (int i : moves) {
            drop.push_back(i);
            if (m->candidatesMoves[i].removalIndex >= 0) drop.push_back(m->candidatesMoves[i].removalIndex);
        }
        std::sort(drop.begin(), drop.end());
        drop.erase(std::unique(drop.begin(), drop.end()), drop.end());
        for (auto it = drop.rbegin(); it != drop.rend(); ++it) t.path.erase(t.path.begin() + *it);
        for (int i : moves)
            if (path_engine::is_node(m->candidatesMoves[i].replacement.source))
                t.path.push_back(m->candidatesMoves[i].replacement);
        m->coverage.reset(t);
        return (t.coverage.array() >= P.req.array()).all();
    }

    bool shareIndex(int a, int b) {
        auto idx = [&](int i){ return std::array<int, 2>{i, m->candidatesMoves[i].removalIndex}; };
        for (int x : idx(a)) for (int y : idx(b)) if (x >= 0 && x == y) return true;
        return false;
    }
};

static std::string segStr(const segment& s) {
    return "{" + std::to_string(s.source) + "," + std::to_string(s.target) + "}";
}

TEST_P(neighborhood_n21_independence_fixture, CoSelectableMovesAreJointlyFeasible) {
    m->setMovesAndGains();
    const int spacing = m->spacingConstraint * P.max_range;
    std::vector<int> improving;
    for (int i = 1; i < static_cast<int>(m->candidatesMoves.size()); ++i)
        if (m->candidatesMoves[i].gain > cost_t{0, 0}) improving.push_back(i);

    for (int i : improving)
        ASSERT_TRUE(jointlyFeasible({i})) << "move " << i << " infeasible alone";

    for (size_t a = 0; a < improving.size(); ++a) {
        for (size_t b = a + 1; b < improving.size(); ++b) {
            const int i = improving[a], j = improving[b];
            if (shareIndex(i, j)) continue;           // never in the same packing
            Interval u = m->transformToInterval(i), v = m->transformToInterval(j);
            if (u.rotated != v.rotated) continue;     // never in the same orientation group
            const bool sameLineDisjoint = u.y == v.y && (u.x2 < v.x1 || v.x2 < u.x1);
            const bool farLines = std::abs(u.y - v.y) >= spacing;
            if (!sameLineDisjoint && !farLines) continue;
            const auto& mi = m->candidatesMoves[i];
            const auto& mj = m->candidatesMoves[j];
            EXPECT_TRUE(jointlyFeasible({i, j}))
                << "co-selectable moves are jointly infeasible:\n"
                << "  i=" << i << " " << segStr(sol.path[i]) << " removal=" << mi.removalIndex
                << (mi.removalIndex >= 0 ? " " + segStr(sol.path[mi.removalIndex]) : std::string())
                << " repl=" << segStr(mi.replacement) << " interval [" << u.x1 << "," << u.x2 << "] y=" << u.y << "\n"
                << "  j=" << j << " " << segStr(sol.path[j]) << " removal=" << mj.removalIndex
                << (mj.removalIndex >= 0 ? " " + segStr(sol.path[mj.removalIndex]) : std::string())
                << " repl=" << segStr(mj.replacement) << " interval [" << v.x1 << "," << v.x2 << "] y=" << v.y;
        }
    }
}

// Applying the neighborhood on the captured inputs never breaks coverage
// (any selection drawn by the randomized packing/scheduling).
TEST_P(neighborhood_n21_independence_fixture, LocalSearchKeepsCoverageFeasible) {
    const auto input = sol;
    for (int draw = 0; draw < 20; ++draw) {
        cppied_solution s = input;
        neighborhood_n21_accessor(ctx, P, s).local_search();
        cppied_solution fresh = s;
        m->coverage.reset(fresh);
        ASSERT_TRUE((fresh.coverage.array() >= P.req.array()).all()) << "draw " << draw;
    }
}

INSTANTIATE_TEST_SUITE_P(CapturedInputs, neighborhood_n21_independence_fixture, ::testing::Values(
    std::vector<segment>{{6,-1},{12,-2},{6,11},{79,80},{35,30},{45,42},{54,57},{18,23},{29,28},{63,62},{16,-1}},
    std::vector<segment>{{6,-1},{48,-2},{7,6},{18,22},{23,-1},{35,30},{44,-2},{6,11},{79,-1},{75,74},{16,-2},{15,12},{18,21}},
    std::vector<segment>{{6,-1},{48,51},{30,35},{23,18},{44,-2},{6,9},{72,73},{62,63},{75,73},{11,-1}},
    std::vector<segment>{{6,-1},{50,-1},{18,19},{51,48},{60,-1},{62,65},{77,73},{61,63},{75,72},{48,53}},
    std::vector<segment>{{6,-1},{50,53},{18,-1},{50,49},{48,-2},{6,7},{14,16},{11,9},{48,-2},{54,57},{65,63},{21,23},{17,-2},{75,77}},
    std::vector<segment>{{6,-1},{7,11},{23,20},{25,24},{45,42},{0,1},{9,10},{79,81},{35,30},{44,-2},{12,13},{56,57},{27,28},{16,15}},
    std::vector<segment>{{6,-1},{7,11},{79,-1},{23,20},{19,18},{6,7},{67,-1},{20,18},{44,-2},{6,-1},{21,23},{29,24},{30,35}},
    std::vector<segment>{{6,-1},{7,11},{80,81},{29,-2},{77,-1},{71,68},{56,59},{53,48},{42,44},{18,-1},{20,22},{73,-2},{22,23}},
    std::vector<segment>{{6,-1},{7,8},{48,51},{52,53},{65,62},{61,60},{10,11},{73,75},{76,77},{23,22},{21,20},{19,18},{44,-2},{43,42}},
    std::vector<segment>{{6,-1},{7,9},{73,75},{69,67},{10,11},{79,81},{35,33},{63,62},{56,59},{53,51},{50,-2},{6,-2},{18,-1},{50,48},{0,-2}}
));

// ============================================================================
// transformToInterval: the interval of a paired move covers the positions on
// its line that the paired removal's coverage reaches.
//
// Geometry: horizontal vertex (row r in [0, n_rows], col j) sits at x = j+0.5;
// vertical vertex (col c in [0, n_cols], row i) sits at x = c, y = i+0.5.
// A perpendicular paired removal on vertical line c therefore reaches the
// horizontal positions [c - max_range, c + max_range - 1] (clamped to the
// row), and a horizontal line r reaches the vertical positions
// [r - max_range, r + max_range - 1]. Intervals are relative positions on
// their line.
// ============================================================================

class neighborhood_n21_interval_fixture : public neighborhood_n21_fixture {
protected:
    // Interval of a move removing path[1] together with path[2].
    Interval pairedInterval(const segment& first, const segment& paired) {
        sol.path = {{6, path_engine::NULL_NODE}, first, paired};
        auto m = std::make_unique<neighborhood_n21_accessor>(ctx, P, sol);
        m->candidatesMoves.assign(sol.path.size(), TwoOneReallocationMove{});
        m->candidatesMoves[1].removalIndex = 2;
        m->candidatesMoves[1].gain = {1, 0};
        return m->transformToInterval(1);
    }
    int hId(int row, int col) { return row * n->geometry.n_cols + col; }
    int vId(int col, int row) { return n->geometry.horizontal_bound + col * n->geometry.n_rows + row; }
    int lastHorizontalPosition() { return n->geometry.n_cols - 1; }
    int lastVerticalPosition() { return n->geometry.n_rows - 1; }
};

TEST_F(neighborhood_n21_interval_fixture, HorizontalFirstPerpendicularInteriorColumn) {
    const int r = P.max_range;
    // First removal: row 3, cols 2..3. Paired: vertical line 5, rows 0..1.
    Interval iv = pairedInterval({hId(3, 2), hId(3, 3)}, {vId(5, 0), vId(5, 1)});
    EXPECT_FALSE(iv.rotated);
    EXPECT_EQ(iv.y, 3);
    EXPECT_LE(iv.x1, 2);
    EXPECT_GE(iv.x2, std::min(lastHorizontalPosition(), 5 + r - 1));
}

TEST_F(neighborhood_n21_interval_fixture, HorizontalFirstPerpendicularLeftColumn) {
    const int r = P.max_range;
    // Paired: vertical line 1 left of the first removal (cols 3..4).
    Interval iv = pairedInterval({hId(3, 3), hId(3, 4)}, {vId(1, 0), vId(1, 1)});
    EXPECT_LE(iv.x1, std::max(0, 1 - r));
    EXPECT_GE(iv.x2, 4);
}

// Vertical line n_cols is the right edge of the grid: it must not be read as
// line 0. It reaches the last horizontal position.
TEST_F(neighborhood_n21_interval_fixture, HorizontalFirstPerpendicularRightEdgeColumn) {
    const int c = n->geometry.n_cols;
    Interval iv = pairedInterval({hId(3, 2), hId(3, 3)}, {vId(c, 0), vId(c, 1)});
    EXPECT_GE(iv.x2, lastHorizontalPosition())
        << "paired removal on the right edge line " << c << " does not extend the interval";
}

TEST_F(neighborhood_n21_interval_fixture, HorizontalFirstParallelPairedRemoval) {
    // First removal: row 3, cols 4..5. Paired: row 2, cols 0..3 (reversed).
    Interval iv = pairedInterval({hId(3, 4), hId(3, 5)}, {hId(2, 3), hId(2, 0)});
    EXPECT_LE(iv.x1, 0);
    EXPECT_GE(iv.x2, 5);
}

TEST_F(neighborhood_n21_interval_fixture, VerticalFirstPerpendicularInteriorRow) {
    const int r = P.max_range;
    // First removal: vertical line 2, rows 1..2. Paired: horizontal row 4.
    Interval iv = pairedInterval({vId(2, 1), vId(2, 2)}, {hId(4, 2), hId(4, 3)});
    EXPECT_TRUE(iv.rotated);
    EXPECT_EQ(iv.y, 2);
    EXPECT_LE(iv.x1, 1);
    EXPECT_GE(iv.x2, std::min(lastVerticalPosition(), 4 + r - 1));
}

// Horizontal row n_rows is the bottom edge: it must not be read as row 0. It
// reaches the last vertical position.
TEST_F(neighborhood_n21_interval_fixture, VerticalFirstPerpendicularBottomEdgeRow) {
    const int rows = n->geometry.n_rows;
    Interval iv = pairedInterval({vId(2, 1), vId(2, 2)}, {hId(rows, 2), hId(rows, 3)});
    EXPECT_GE(iv.x2, lastVerticalPosition())
        << "paired removal on the bottom edge row " << rows << " does not extend the interval";
}

TEST_F(neighborhood_n21_interval_fixture, VerticalFirstParallelPairedRemoval) {
    // First removal: vertical line 2, rows 4..5. Paired: line 3, rows 0..3 (reversed).
    Interval iv = pairedInterval({vId(2, 4), vId(2, 5)}, {vId(3, 3), vId(3, 0)});
    EXPECT_LE(iv.x1, 0);
    EXPECT_GE(iv.x2, 5);
}
