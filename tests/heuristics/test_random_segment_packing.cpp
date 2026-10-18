#include "../test_fixture.hpp"
#include "../../include/heuristics/RandomSegmentPacking.hpp"

// RandomSegmentPacking selects, in random order, a maximal subset of the
// candidate path indices whose coverage footprints are pairwise disjoint.
// Regression: solve() returned every (shuffled) candidate instead of the
// packed subset, so overlapping GTSP clusters were selected together.
class random_segment_packing_fixture : public cppied_method_fixture {
protected:
    cppied_solution sol;

    void SetUp() override {
        cppied_method_fixture::SetUp();
        sol.path = {{6, 11},
                    {23, 18},
                    {24, 29},
                    {44, 42},
                    {48, 51},
                    {30, 35},
                    {36, 41},
                    {75, 73},
                    {62, 63}};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        method_instance.coverage.reset(sol);
        sol.cost = method_instance.geometry.cost(sol);
    }

    bool footprintsIntersect(int i, int j) {
        Eigen::VectorXi mask = Eigen::VectorXi::Zero(P.req.size());
        method_instance.coverage.mask(sol, sol.path[i], mask);
        return method_instance.coverage.intersect_mask(sol, sol.path[j], mask);
    }

    std::vector<int> solve(std::vector<int> candidates) {
        RandomSegmentPacking packing(P, method_instance.coverage, method_instance.geometry, sol, candidates);
        return packing.solve();
    }

    void expectValidPacking(const std::vector<int>& candidates, const std::vector<int>& items) {
        // Subset of the candidates, without duplicates.
        std::set<int> unique(items.begin(), items.end());
        EXPECT_EQ(unique.size(), items.size()) << "duplicate items in packing";
        for (int i : items)
            EXPECT_NE(std::find(candidates.begin(), candidates.end(), i), candidates.end())
                << "item " << i << " is not a candidate";
        // Pairwise disjoint footprints.
        for (size_t a = 0; a < items.size(); ++a)
            for (size_t b = a + 1; b < items.size(); ++b)
                EXPECT_FALSE(footprintsIntersect(items[a], items[b]))
                    << "items " << items[a] << " and " << items[b] << " overlap";
        // Maximal: every left-out candidate overlaps a packed item.
        for (int c : candidates) {
            if (unique.count(c)) continue;
            bool blocked = std::any_of(items.begin(), items.end(),
                                       [&](int i){ return footprintsIntersect(c, i); });
            EXPECT_TRUE(blocked) << "candidate " << c << " could have been packed";
        }
    }
};

// The fixture must contain overlapping candidates for the tests to be meaningful.
TEST_F(random_segment_packing_fixture, FixtureHasOverlappingCandidates) {
    EXPECT_TRUE(footprintsIntersect(1, 2));   // {23,18} and {24,29}
    EXPECT_TRUE(footprintsIntersect(5, 6));   // {30,35} and {36,41}
}

// Every draw is a valid maximal packing (checked over many shuffles).
TEST_F(random_segment_packing_fixture, SolveReturnsMaximalDisjointSubset) {
    std::vector<int> candidates = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int draw = 0; draw < 50; ++draw) {
        SCOPED_TRACE("draw " + std::to_string(draw));
        expectValidPacking(candidates, solve(candidates));
    }
}

// Overlapping candidates are never returned together.
TEST_F(random_segment_packing_fixture, SolveNeverReturnsAllOverlappingCandidates) {
    std::vector<int> candidates = {1, 2};
    for (int draw = 0; draw < 50; ++draw)
        EXPECT_EQ(solve(candidates).size(), 1u);
}

// Candidates with pairwise disjoint footprints are all packed.
TEST_F(random_segment_packing_fixture, SolveKeepsAllDisjointCandidates) {
    // Greedily build a set of pairwise disjoint candidates from the path.
    std::vector<int> candidates;
    for (int i = 1; i < static_cast<int>(sol.path.size()); ++i)
        if (std::none_of(candidates.begin(), candidates.end(),
                         [&](int c){ return footprintsIntersect(c, i); }))
            candidates.push_back(i);
    ASSERT_GE(candidates.size(), 2u);
    auto items = solve(candidates);
    std::sort(items.begin(), items.end());
    EXPECT_EQ(items, candidates);
}

// Repeated solve() calls on the same object do not accumulate items.
TEST_F(random_segment_packing_fixture, RepeatedSolveDoesNotAccumulate) {
    std::vector<int> candidates = {1, 2, 3, 4, 5, 6, 7, 8};
    RandomSegmentPacking packing(P, method_instance.coverage, method_instance.geometry, sol, candidates);
    auto first = packing.solve();
    auto second = packing.solve();
    expectValidPacking(candidates, first);
    expectValidPacking(candidates, second);
}

// Each candidate is selectable: over many draws every candidate is packed at
// least once (each one is first in the shuffle with probability 1/8).
TEST_F(random_segment_packing_fixture, EveryCandidateIsEventuallySelected) {
    std::vector<int> candidates = {1, 2, 3, 4, 5, 6, 7, 8};
    std::set<int> seen;
    for (int draw = 0; draw < 400; ++draw)
        for (int i : solve(candidates)) seen.insert(i);
    EXPECT_EQ(seen, std::set<int>(candidates.begin(), candidates.end()));
}
