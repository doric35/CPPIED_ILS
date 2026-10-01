#include <gtest/gtest.h>
#include <numeric>
#include <set>
#include "../../include/heuristics/ris_heuristic.hpp"

// ============================================================================
// ris_heuristic unit tests
//
// What is being tested:
//   solve_selection() is a 2D interval-scheduling selection heuristic:
//     1. filter_function  — removes candidates with non-positive gain
//     2. split_function   — partitions survivors into independent groups
//     3. interval_function — maps each candidate id → interval{gain,x1,x2,y,id}
//     4. interval_dp      — per y-row: non-overlapping weighted interval scheduling
//     5. spacing_dp       — across y-rows: selects rows ≥ spacing apart
//     6. softmaxSample   — picks one group proportionally to total gain
//
// Test strategy: controlled lambdas with exact spatial coordinates to
// exercise each stage deterministically.  Probabilistic tests use a fixed
// seed (std::mt19937) to guarantee reproducibility.
// ============================================================================

namespace {

// ---------------------------------------------------------------------------
// Lambda factories — build the three injection points for ris_heuristic.
// ---------------------------------------------------------------------------

// Filter: erase every candidate index whose entry in `gains` is <= {0,0}.
auto make_filter(const std::vector<cost_t>& gains) {
    return [&gains](std::vector<int>& candidates) {
        std::erase_if(candidates,
                      [&gains](int i) { return gains[i] <= cost_t{0, 0}; });
    };
}

// Splitter: puts ALL surviving candidates into a single group (solution[0]).
// This isolates spacing_dp and interval_dp tests from the split dimension.
auto make_single_group_splitter() {
    return [](const std::vector<int>& candidates,
              std::vector<std::vector<int>>& solution) {
        solution.resize(1);
        solution[0] = candidates;
    };
}

// Interval function: maps each candidate id to the interval stored in `coords`.
// The caller controls x1, x2, y exactly.
auto make_interval_fn(const std::map<int, interval>& coords) {
    return [coords](const std::vector<int>& candidates,
                    std::vector<interval>& out) {
        for (int id : candidates)
            out.push_back(coords.at(id));
    };
}

// Convenience: build a ris_heuristic with the single-group splitter and a
// coordinate map.  Lifetime of `gains` must exceed that of the heuristic.
ris_heuristic make_heuristic(int spacing,
                              const std::vector<cost_t>& gains,
                              const std::map<int, interval>& coords) {
    return ris_heuristic(spacing,
                         make_single_group_splitter(),
                         make_filter(gains),
                         make_interval_fn(coords));
}

} // anonymous namespace

// ============================================================================
// Filter stage
// ============================================================================

// When every gain is zero or negative the filter removes all candidates and
// the selection must be empty.
TEST(RisHeuristic, AllZeroGains_SelectionIsEmpty) {
    std::vector<cost_t> gains = {{0,0},{0,0},{0,0}};
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 2, 2, 0, false, 1}},
        {2, {gains[2], 4, 4, 0, false, 2}}
    };
    auto h = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    EXPECT_TRUE(selection.empty());
}

// A vector of all negative gains must also yield an empty selection.
TEST(RisHeuristic, AllNegativeGains_SelectionIsEmpty) {
    std::vector<cost_t> gains = {{-3,0},{-1,2},{-2,-1}};
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 2, 2, 0, false, 1}},
        {2, {gains[2], 4, 4, 0, false, 2}}
    };
    auto h = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    EXPECT_TRUE(selection.empty());
}

// Only the single candidate with a positive gain should be selected.
TEST(RisHeuristic, SinglePositiveGain_ThatIdSelected) {
    std::vector<cost_t> gains = {{0,0},{5,2},{0,0}};
    std::map<int, interval> coords = {
        {1, {gains[1], 2, 2, 0, false, 1}}
    };
    auto h = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    ASSERT_EQ(selection.size(), 1u);
    EXPECT_EQ(selection[0], 1);
}

// ============================================================================
// interval_dp — non-overlapping subset selection within a single y-row
// ============================================================================

// Three candidates whose x-intervals are disjoint: all three must appear.
// Intervals [0,1], [3,4], [6,7] are pairwise non-overlapping (gap ≥ 1 apart).
TEST(RisHeuristic, IntervalDp_NonOverlapping_AllSelected) {
    std::vector<cost_t> gains = {{5,0},{5,0},{5,0}};
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 1, 0, false, 0}},
        {1, {gains[1], 3, 4, 0, false, 1}},
        {2, {gains[2], 6, 7, 0, false, 2}}
    };
    auto h = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    EXPECT_EQ(selection.size(), 3u);
}

// Two overlapping candidates: at most one can be selected.
// Intervals [0,5] and [3,8] overlap (3 <= 5).
TEST(RisHeuristic, IntervalDp_TwoOverlapping_OnlyOneSelected) {
    std::vector<cost_t> gains = {{5,0},{5,0}};
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 5, 0, false, 0}},
        {1, {gains[1], 3, 8, 0, false, 1}}
    };
    auto h = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    EXPECT_EQ(selection.size(), 1u);
}

// Among two overlapping candidates the one with strictly higher gain must
// always be selected (no tie, no randomness involved).
TEST(RisHeuristic, IntervalDp_HigherGainOverlapAlwaysWins) {
    std::vector<cost_t> gains = {{3,0},{10,0}};
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 5, 0, false, 0}},
        {1, {gains[1], 2, 8, 0, false, 1}}
    };
    auto h = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    ASSERT_EQ(selection.size(), 1u);
    EXPECT_EQ(selection[0], 1) << "Candidate with higher gain must be selected";
}

// Three candidates: the two short non-overlapping ones beat the single long one
// when their combined gain is higher.
// Short pair: [0,2] gain={5,0} and [4,6] gain={5,0} → total {10,0}
// Long one:   [0,6] gain={8,0}
// interval_dp must prefer the pair.
TEST(RisHeuristic, IntervalDp_PairBeatsLongSegment_WhenCombinedGainHigher) {
    std::vector<cost_t> gains = {{5,0},{5,0},{8,0}};
    // id0: [0,2], id1: [4,6], id2: [0,6] (overlaps both)
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 2, 0, false, 0}},
        {1, {gains[1], 4, 6, 0, false, 1}},
        {2, {gains[2], 0, 6, 0, false, 2}}
    };
    auto h = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    // Total gain of pair ({10,0}) beats single ({8,0}).
    EXPECT_EQ(selection.size(), 2u);
    std::set<int> sel_set(selection.begin(), selection.end());
    EXPECT_TRUE(sel_set.count(0) && sel_set.count(1))
        << "Pair {id0, id1} expected, not the single overlapping id2";
}

// ============================================================================
// spacing_dp — row separation constraint across y-values
// ============================================================================

// Two y-rows closer than spacing: only one row must be selected.
// y=0 and y=1, spacing=3 → distance 1 < 3, impossible to pick both.
TEST(RisHeuristic, SpacingDp_RowsTooClose_OnlyOneRow) {
    std::vector<cost_t> gains = {{5,0},{5,0}};
    // id0 at y=0, id1 at y=1, non-overlapping x so interval_dp keeps both
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 0, 0, 1, false, 1}}
    };
    auto h = make_heuristic(3, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    EXPECT_EQ(selection.size(), 1u) << "spacing=3, rows at y=0 and y=1 are too close";
}

// Two y-rows exactly spacing apart: both must be selectable.
// y=0 and y=3, spacing=3 → distance 3 == spacing, dp can include both.
TEST(RisHeuristic, SpacingDp_RowsExactlySpacingApart_BothSelected) {
    std::vector<cost_t> gains = {{5,0},{5,0}};
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 0, 0, 3, false, 1}}
    };
    auto h = make_heuristic(3, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    EXPECT_EQ(selection.size(), 2u) << "spacing=3, rows at y=0 and y=3 should both be selected";
}

// Three y-rows with spacing=2: rows 0, 2, 4 can all be selected (each pair
// is exactly spacing apart); row 1 and 3 must not be chosen if they conflict.
TEST(RisHeuristic, SpacingDp_EveryOtherRow_AllThreeSelected) {
    std::vector<cost_t> gains = {{5,0},{5,0},{5,0}};
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 0, 0, 2, false, 1}},
        {2, {gains[2], 0, 0, 4, false, 2}}
    };
    auto h = make_heuristic(2, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    EXPECT_EQ(selection.size(), 3u) << "y=0,2,4 with spacing=2: all three are selectable";
}

// A high-gain row at y=0 and a low-gain row at y=1, spacing=2.
// Since y=1 is alone and y=0 is within spacing, spacing_dp must pick y=0.
TEST(RisHeuristic, SpacingDp_HighGainRowPreferred_WhenConflicting) {
    std::vector<cost_t> gains = {{10,0},{2,0}};
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 0, 0, 1, false, 1}}
    };
    auto h = make_heuristic(2, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    ASSERT_EQ(selection.size(), 1u);
    EXPECT_EQ(selection[0], 0) << "Higher-gain row y=0 must be preferred over y=1";
}

// ============================================================================
// Structural contracts on the result
// ============================================================================

// Every id in the selection must be a valid index into gains[].
TEST(RisHeuristic, SelectionIds_WithinValidRange) {
    std::vector<cost_t> gains = {{3,1},{7,0},{0,0},{5,2},{0,0}};
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 2, 2, 0, false, 1}},
        {3, {gains[3], 4, 4, 0, false, 3}}
    };
    auto h = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    for (int id : selection) {
        EXPECT_GE(id, 0);
        EXPECT_LT(id, static_cast<int>(gains.size()));
    }
}

// The selection must contain no duplicate ids.
TEST(RisHeuristic, SelectionIds_NoDuplicates) {
    std::vector<cost_t> gains = {{5,0},{5,0},{5,0},{5,0}};
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 2, 2, 0, false, 1}},
        {2, {gains[2], 4, 4, 0, false, 2}},
        {3, {gains[3], 6, 6, 0, false, 3}}
    };
    auto h = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    std::set<int> unique(selection.begin(), selection.end());
    EXPECT_EQ(unique.size(), selection.size()) << "Selection must not contain duplicate ids";
}

// Selected ids must not appear at positions with non-positive gain.
TEST(RisHeuristic, SelectionIds_AllHavePositiveGain) {
    std::vector<cost_t> gains = {{5,0},{0,0},{4,0},{0,0},{3,0}};
    std::map<int, interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {2, {gains[2], 2, 2, 0, false, 2}},
        {4, {gains[4], 4, 4, 0, false, 4}}
    };
    auto h = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    h.solve_selection(gains, selection);
    for (int id : selection)
        EXPECT_GT(gains[id], (cost_t{0,0})) << "Selected id=" << id << " has non-positive gain";
}

// ============================================================================
// softmaxSample (public static overload)
// ============================================================================

// With a single gain the only valid return index is 0.
TEST(RisHeuristic, SoftmaxSample_SingleGain_ReturnsZero) {
    std::vector<cost_t> gains = {{5,3}};
    std::mt19937 gen(42);
    int idx = ris_heuristic::softmaxSample(gains, gen);
    EXPECT_EQ(idx, 0);
}

// With multiple gains the returned index must be in [0, gains.size()).
TEST(RisHeuristic, SoftmaxSample_MultipleGains_IndexInRange) {
    std::vector<cost_t> gains = {{5,3},{10,1},{1,7}};
    std::mt19937 gen(42);
    int idx = ris_heuristic::softmaxSample(gains, gen);
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, static_cast<int>(gains.size()));
}

// With all-equal gains every index must be reachable over many draws.
// This is a distribution sanity-check: no single index should dominate if
// gains are equal (softmax collapses to uniform in this regime).
TEST(RisHeuristic, SoftmaxSample_AllEqualGains_AllIndicesReachable) {
    std::vector<cost_t> gains = {{5,0},{5,0},{5,0}};
    std::mt19937 gen(1234);
    std::set<int> seen;
    for (int t = 0; t < 300; ++t)
        seen.insert(ris_heuristic::softmaxSample(gains, gen));
    EXPECT_EQ(seen.size(), 3u) << "All three equal-gain indices should be reachable";
}

// The index with a vastly higher gain must dominate after many samples.
// score = M*length + turns; large length difference makes index 1 dominate.
TEST(RisHeuristic, SoftmaxSample_HighGainDominates) {
    // gains[1] has length=1000, far exceeding gains[0].length=1
    std::vector<cost_t> gains = {{1,0},{1000,0}};
    std::mt19937 gen(99);
    int ones = 0;
    for (int t = 0; t < 200; ++t)
        if (ris_heuristic::softmaxSample(gains, gen) == 1) ++ones;
    // Expect at least 190/200 draws to pick index 1
    EXPECT_GE(ones, 190) << "Index 1 with gain 1000 should dominate over index 0 with gain 1";
}
