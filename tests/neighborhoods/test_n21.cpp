#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_n21.hpp"

struct neighborhood_n21_accessor : public neighborhood_n21 {
public:
    using neighborhood_n21::neighborhood_n21;

    using neighborhood::geometry;
    using neighborhood::coverage;
    using neighborhood::problem;

    //using neighborhood_n21::gain;
    using neighborhood_n21::update;
};

class neighborhood_n21_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<neighborhood_n21_accessor> n;
    cppied_solution sol;

    void SetUp() override{
        cppied_context_fixture::SetUp();
        sol.path =  {{6, 11},
                     {23,18},
                     {44,43},
                     {6, NULL_NODE},
                     {48, 51},
                     {30, 35},
                     {75,73},
                     {62, 63}};
        sol.cost = {0,0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        n = std::make_unique<neighborhood_n21_accessor>(ctx, P);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

TEST_F(neighborhood_n21_fixture, ImpromentFound){
    cost_t c_0 = sol.cost;
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    bool improved = n->local_search(sol);
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
    n->local_search(sol);
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
    bool improved = n->local_search(sol);
    ASSERT_EQ(n->geometry.cost(sol), sol.cost);
    n->coverage.reset(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    EXPECT_FALSE(improved);
    EXPECT_EQ(sol.cost, c);
}

TEST_F(neighborhood_n21_fixture, ValidCoverageUpdate){
    ASSERT_TRUE(n->local_search(sol));
    Eigen::VectorXd C = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(C, 1e-9));
}

TEST_F(neighborhood_n21_fixture, CoverageConstraintSatisfied){
    ASSERT_TRUE(n->local_search(sol));
    n->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

TEST_F(neighborhood_n21_fixture, LocalOptimaAfterImprovement){
    ASSERT_TRUE(n->local_search(sol));
    EXPECT_FALSE(n->local_search(sol));
}

//-----------Test specific cases-----------------------

// ── Direct-removal fast path ──────────────────────────────────────────────────
//
// When coverage.can_remove() is true for a segment, replace() must take the
// fast path (no explore_replacements call) and return a trial that:
//   · carries the removal gain of the segment
//   · has no replacement segment  (s == {NULL_NODE, NULL_NODE})
//   · has no "other" partner      (other == -1)
//   · has no insertion position   (position == -1)
//
// The test solution is built on the known 7-segment local optimum used in
// LocalOptimaFound, extended with a single-node segment {6, NULL_NODE}.
// Vertex 6 is already visited by sweep {6, 11}, so its coverage contribution
// is completely redundant and the segment may be removed outright.

// Helper fixture for the direct-removal case.
class neighborhood_n21_direct_removal_fixture : public neighborhood_n21_fixture {
protected:
    // 8-segment path: the 7-segment local optimum with {6, NULL_NODE} inserted
    // between {44,42} and {48,51}.  Vertex 6 is already covered by {6, 11}.
    static constexpr int REDUNDANT_IDX = 3; // 0-based index of {6, NULL_NODE}

    void SetUp() override {
        neighborhood_n21_fixture::SetUp();
        sol.path = {{6, 11},
                    {23, 18},
                    {44, 42},
                    {6, NULL_NODE},   // redundant: vertex 6 already in {6,11}
                    {48, 51},
                    {30, 35},
                    {75, 73},
                    {62, 63}};
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

// replace() must return the fast-path trial when the segment is directly removable.
TEST_F(neighborhood_n21_direct_removal_fixture, DirectRemoval_ReplaceReturnsFastPathTrial) {
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());

    // Precondition: the redundant segment is indeed removable without a replacement.
    ASSERT_TRUE(n->coverage.can_remove(sol, sol.path[REDUNDANT_IDX]));

    auto it = std::next(sol.path.cbegin(), REDUNDANT_IDX);
    const cost_t expected_gain = n->geometry.removal_gain(sol, it);

    // The removal gain must be positive (detour through vertex 6 is costly).
    ASSERT_GT(expected_gain, (cost_t{0, 0}));

    // Call replace() – path and coverage must be left unmodified.
    Eigen::VectorXd coverage_before = sol.coverage;
    std::vector<segment> path_before = sol.path;

    n21::trial t = n->replace(sol, it);

    // Path and coverage are restored after the call.
    EXPECT_EQ(sol.path, path_before);
    EXPECT_TRUE(sol.coverage.isApprox(coverage_before, 1e-9));

    // Trial fields match the fast-path contract.
    EXPECT_EQ(t.gain,     expected_gain);
    EXPECT_FALSE(is_node(t.s.source));   // no replacement segment
    EXPECT_FALSE(is_node(t.s.target));
    EXPECT_EQ(t.other,    -1);           // no second segment to remove
    EXPECT_EQ(t.position, -1);           // no insertion position
}

// local_search() must find improvement by removing the redundant segment.
TEST_F(neighborhood_n21_direct_removal_fixture, DirectRemoval_LocalSearchFindsImprovement) {
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    const cost_t cost_before = sol.cost;

    const bool improved = n->local_search(sol);

    EXPECT_TRUE(improved);
    EXPECT_LT(sol.cost, cost_before);
}

// The stored cost must equal the recomputed cost after the move.
TEST_F(neighborhood_n21_direct_removal_fixture, DirectRemoval_CostConsistentAfterMove) {
    n->local_search(sol);
    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
}

// Coverage must remain valid (all cells at or above their requirements).
TEST_F(neighborhood_n21_direct_removal_fixture, DirectRemoval_CoverageRemainsValid) {
    n->local_search(sol);
    n->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

// The incremental coverage vector must match a full reset.
TEST_F(neighborhood_n21_direct_removal_fixture, DirectRemoval_CoverageVectorConsistent) {
    n->local_search(sol);
    const Eigen::VectorXd coverage_after = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(coverage_after, 1e-9));
}

// After removing the redundant segment we must reach the known 7-segment local
// optimum; a second call to local_search() must therefore report no improvement.
TEST_F(neighborhood_n21_direct_removal_fixture, DirectRemoval_ReachesLocalOptimum) {
    ASSERT_TRUE(n->local_search(sol));  // first call removes the redundant segment
    EXPECT_FALSE(n->local_search(sol)); // second call: already at local optimum
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
    n->local_search(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}