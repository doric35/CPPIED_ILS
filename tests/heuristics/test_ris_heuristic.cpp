#include <gtest/gtest.h>
#include <numeric>
#include <set>
#include <cmath>
#include <functional>
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
auto make_interval_fn(const std::map<int, Interval>& coords) {
    return [coords](const std::vector<int>& candidates,
                    IntervalSet& out) {
        for (int id : candidates)
            out.push_back(coords.at(id));
    };
}

// ris_heuristic now receives pre-built interval buckets; the filter / split /
// interval stages live in the neighborhoods. This harness rebuilds the buckets
// from the three injection points so the pipeline is tested end to end.
struct ris_harness {
    using splitter_t = std::function<void(const std::vector<int>&, std::vector<std::vector<int>>&)>;
    using filter_t = std::function<void(std::vector<int>&)>;
    using interval_fn_t = std::function<void(const std::vector<int>&, IntervalSet&)>;

    ris_harness(int spacing, int size, splitter_t splitter, filter_t filter, interval_fn_t interval_fn) :
        h(spacing, size), size(size), splitter(std::move(splitter)),
        filter(std::move(filter)), interval_fn(std::move(interval_fn)){}

    std::vector<int> solve_selection() {
        std::vector<int> candidates(size);
        std::iota(candidates.begin(), candidates.end(), 0);
        filter(candidates);
        std::vector<std::vector<int>> candidate_sets;
        splitter(candidates, candidate_sets);
        GroupedIntervalSets buckets;
        for (const auto& candidate_set : candidate_sets) {
            IntervalSet intervals;
            interval_fn(candidate_set, intervals);
            buckets.push(intervals);
        }
        return h.solve_selection(buckets);
    }
    void clear() { h.clear(); }

    ris_heuristic h;
    int size;
    splitter_t splitter;
    filter_t filter;
    interval_fn_t interval_fn;
};

// Convenience: build a ris_harness with the single-group splitter and a
// coordinate map.  Lifetime of `gains` must exceed that of the heuristic.
ris_harness make_heuristic(int spacing,
                            const std::vector<cost_t>& gains,
                            const std::map<int, Interval>& coords) {
    return ris_harness(spacing,
                       static_cast<int>(gains.size()),
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
    std::vector<cost_t>     gains  = {{0,0},{0,0},{0,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 2, 2, 0, false, 1}},
        {2, {gains[2], 4, 4, 0, false, 2}}
    };
    auto                    h      = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
    EXPECT_TRUE(selection.empty());
}

// A vector of all negative gains must also yield an empty selection.
TEST(RisHeuristic, AllNegativeGains_SelectionIsEmpty) {
    std::vector<cost_t>     gains  = {{-3,0},{-1,2},{-2,-1}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 2, 2, 0, false, 1}},
        {2, {gains[2], 4, 4, 0, false, 2}}
    };
    auto                    h      = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
    EXPECT_TRUE(selection.empty());
}

// Only the single candidate with a positive gain should be selected.
TEST(RisHeuristic, SinglePositiveGain_ThatIdSelected) {
    std::vector<cost_t>     gains  = {{0,0},{5,2},{0,0}};
    std::map<int, Interval> coords = {
        {1, {gains[1], 2, 2, 0, false, 1}}
    };
    auto                    h      = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
    ASSERT_EQ(selection.size(), 1u);
    EXPECT_EQ(selection[0], 1);
}

// ============================================================================
// interval_dp — non-overlapping subset selection within a single y-row
// ============================================================================

// Three candidates whose x-intervals are disjoint: all three must appear.
// Intervals [0,1], [3,4], [6,7] are pairwise non-overlapping (gap ≥ 1 apart).
TEST(RisHeuristic, IntervalDp_NonOverlapping_AllSelected) {
    std::vector<cost_t>     gains  = {{5,0},{5,0},{5,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 1, 0, false, 0}},
        {1, {gains[1], 3, 4, 0, false, 1}},
        {2, {gains[2], 6, 7, 0, false, 2}}
    };
    auto                    h      = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
    EXPECT_EQ(selection.size(), 3u);
}

// Two overlapping candidates: at most one can be selected.
// Intervals [0,5] and [3,8] overlap (3 <= 5).
TEST(RisHeuristic, IntervalDp_TwoOverlapping_OnlyOneSelected) {
    std::vector<cost_t>     gains  = {{5,0},{5,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 5, 0, false, 0}},
        {1, {gains[1], 3, 8, 0, false, 1}}
    };
    auto                    h      = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
    EXPECT_EQ(selection.size(), 1u);
}

// Among two overlapping candidates the one with strictly higher gain must
// always be selected (no tie, no randomness involved).
TEST(RisHeuristic, IntervalDp_HigherGainOverlapAlwaysWins) {
    std::vector<cost_t>     gains  = {{3,0},{10,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 5, 0, false, 0}},
        {1, {gains[1], 2, 8, 0, false, 1}}
    };
    auto                    h      = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
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
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 2, 0, false, 0}},
        {1, {gains[1], 4, 6, 0, false, 1}},
        {2, {gains[2], 0, 6, 0, false, 2}}
    };
    auto                    h      = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
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
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 0, 0, 1, false, 1}}
    };
    auto                    h      = make_heuristic(3, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
    EXPECT_EQ(selection.size(), 1u) << "spacing=3, rows at y=0 and y=1 are too close";
}

// Two y-rows exactly spacing apart: both must be selectable.
// y=0 and y=3, spacing=3 → distance 3 == spacing, dp can include both.
TEST(RisHeuristic, SpacingDp_RowsExactlySpacingApart_BothSelected) {
    std::vector<cost_t>     gains  = {{5,0},{5,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 0, 0, 3, false, 1}}
    };
    auto                    h      = make_heuristic(3, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
    EXPECT_EQ(selection.size(), 2u) << "spacing=3, rows at y=0 and y=3 should both be selected";
}

// Three y-rows with spacing=2: rows 0, 2, 4 can all be selected (each pair
// is exactly spacing apart); row 1 and 3 must not be chosen if they conflict.
TEST(RisHeuristic, SpacingDp_EveryOtherRow_AllThreeSelected) {
    std::vector<cost_t>     gains  = {{5,0},{5,0},{5,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 0, 0, 2, false, 1}},
        {2, {gains[2], 0, 0, 4, false, 2}}
    };
    auto                    h      = make_heuristic(2, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
    EXPECT_EQ(selection.size(), 3u) << "y=0,2,4 with spacing=2: all three are selectable";
}

// A high-gain row at y=0 and a low-gain row at y=1, spacing=2.
// Since y=1 is alone and y=0 is within spacing, spacing_dp must pick y=0.
TEST(RisHeuristic, SpacingDp_HighGainRowPreferred_WhenConflicting) {
    std::vector<cost_t>     gains  = {{10,0},{2,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 0, 0, 1, false, 1}}
    };
    auto                    h      = make_heuristic(2, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
    ASSERT_EQ(selection.size(), 1u);
    EXPECT_EQ(selection[0], 0) << "Higher-gain row y=0 must be preferred over y=1";
}

// ============================================================================
// Structural contracts on the result
// ============================================================================

// Every id in the selection must be a valid index into gains[].
TEST(RisHeuristic, SelectionIds_WithinValidRange) {
    std::vector<cost_t>     gains  = {{3,1},{7,0},{0,0},{5,2},{0,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 2, 2, 0, false, 1}},
        {3, {gains[3], 4, 4, 0, false, 3}}
    };
    auto                    h      = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
    for (int id : selection) {
        EXPECT_GE(id, 0);
        EXPECT_LT(id, static_cast<int>(gains.size()));
    }
}

// The selection must contain no duplicate ids.
TEST(RisHeuristic, SelectionIds_NoDuplicates) {
    std::vector<cost_t>     gains  = {{5,0},{5,0},{5,0},{5,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 2, 2, 0, false, 1}},
        {2, {gains[2], 4, 4, 0, false, 2}},
        {3, {gains[3], 6, 6, 0, false, 3}}
    };
    auto                    h      = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
    std::set<int> unique(selection.begin(), selection.end());
    EXPECT_EQ(unique.size(), selection.size()) << "Selection must not contain duplicate ids";
}

// Selected ids must not appear at positions with non-positive gain.
TEST(RisHeuristic, SelectionIds_AllHavePositiveGain) {
    std::vector<cost_t>     gains  = {{5,0},{0,0},{4,0},{0,0},{3,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {2, {gains[2], 2, 2, 0, false, 2}},
        {4, {gains[4], 4, 4, 0, false, 4}}
    };
    auto                    h      = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
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

// ============================================================================
// Group selection and repeated calls
// ============================================================================

// Two independent groups (one per candidate): the group whose schedule has a
// vastly higher gain must dominate the softmax draw. This relies on the value
// of each schedule built by merging row solutions being the sum of its gains.
TEST(RisHeuristic, TwoGroups_HighGainScheduleDominates) {
    std::vector<cost_t> gains = {{1000,0},{1,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 0, 0, 0, false, 1}}
    };
    auto one_group_per_candidate = [](const std::vector<int>& candidates,
                                      std::vector<std::vector<int>>& solution) {
        for (int c : candidates) solution.push_back({c});
    };
    int high = 0;
    for (int t = 0; t < 200; ++t) {
        ris_harness h(1, static_cast<int>(gains.size()), one_group_per_candidate, make_filter(gains),
                        make_interval_fn(coords));
        std::vector<int> selection;
        selection = h.solve_selection();
        ASSERT_EQ(selection.size(), 1u);
        if (selection[0] == 0) ++high;
    }
    EXPECT_GE(high, 190) << "Group with gain 1000 should dominate group with gain 1";
}

// The same heuristic instance is reused across calls by the neighborhoods.
// Schedules from a previous call must not leak into the next one.
TEST(RisHeuristic, RepeatedCalls_OnlyCurrentCandidatesSelected) {
    std::vector<cost_t> gains = {{5,0},{0,0}};
    std::map<int, Interval> coords = {
        {0, {{5,0}, 0, 0, 0, false, 0}},
        {1, {{5,0}, 4, 4, 0, false, 1}}
    };
    auto h = make_heuristic(1, gains, coords);
    std::vector<int> selection;
    selection = h.solve_selection();
    ASSERT_EQ(selection, (std::vector<int>{0}));

    gains = {{0,0},{5,0}};   // filter now only keeps id 1
    for (int t = 0; t < 50; ++t) {
        selection.clear();
        selection = h.solve_selection();
        ASSERT_EQ(selection, (std::vector<int>{1})) << "call " << t
            << ": a schedule from an earlier call was selected";
    }
}

// ============================================================================
// Candidate count passed at construction
// ============================================================================

// size == 0: no candidate ids exist, the selection must be empty.
TEST(RisHeuristic, ZeroSize_SelectionIsEmpty) {
    std::vector<cost_t> gains = {{5,0}};
    std::map<int, Interval> coords = {{0, {gains[0], 0, 0, 0, false, 0}}};
    ris_harness h(1, 0, make_single_group_splitter(), make_filter(gains),
                    make_interval_fn(coords));
    EXPECT_TRUE(h.solve_selection().empty());
}

// Only ids in [0, size) are candidates, even if more gains are available.
TEST(RisHeuristic, SizeSmallerThanGains_OnlyFirstIdsCandidates) {
    std::vector<cost_t> gains = {{1,0},{9,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 5, 0, false, 0}},
        {1, {gains[1], 2, 8, 0, false, 1}}   // overlaps id 0, higher gain
    };
    ris_harness h(1, 1, make_single_group_splitter(), make_filter(gains),
                    make_interval_fn(coords));
    EXPECT_EQ(h.solve_selection(), (std::vector<int>{0}));
}

// ============================================================================
// setFlatScores — flattens cost_t into a scalar preserving lexicographic order
// ============================================================================

TEST(RisHeuristic, SetFlatScores_SingleGain) {
    std::vector<cost_t> gains = {{3,2}};
    auto scores = ris_heuristic::setFlatScores(gains);
    ASSERT_EQ(scores.size(), 1u);
    EXPECT_DOUBLE_EQ(scores[0], 3.0);   // M = t_max - t_min + 1 = 1, turns shifted by t_min = 2
}

// Score: M * length + (turns - t_min), M = t_max - t_min + 1.
TEST(RisHeuristic, SetFlatScores_ShiftsTurnsByMinimum) {
    std::vector<cost_t> gains = {{2,2},{1,-6}};
    auto scores = ris_heuristic::setFlatScores(gains);
    EXPECT_EQ(scores, (std::vector<double>{9.0 * 2 + 8, 9.0 * 1 + 0}));   // M = 9
}

TEST(RisHeuristic, SetFlatScores_ZeroTurns_ScoreIsLength) {
    std::vector<cost_t> gains = {{3,0},{0,0},{7,0}};
    auto scores = ris_heuristic::setFlatScores(gains);
    EXPECT_EQ(scores, (std::vector<double>{3.0, 0.0, 7.0}));
}

TEST(RisHeuristic, SetFlatScores_DoesNotModifyGains) {
    std::vector<cost_t> gains = {{3,2},{1,5}};
    auto copy = gains;
    ris_heuristic::setFlatScores(gains);
    EXPECT_EQ(gains, copy);
}

// The score order must match the cost_t (lexicographic) order for any pair.
TEST(RisHeuristic, SetFlatScores_PreservesLexicographicOrder) {
    std::mt19937 gen(5);
    std::uniform_int_distribution<int> len(0, 20), turns(0, 6);
    for (int trial = 0; trial < 300; ++trial) {
        std::vector<cost_t> gains(6);
        for (auto& g : gains) g = {len(gen), turns(gen)};
        auto scores = ris_heuristic::setFlatScores(gains);
        for (std::size_t i = 0; i < gains.size(); ++i)
            for (std::size_t j = 0; j < gains.size(); ++j)
                ASSERT_EQ(gains[i] < gains[j], scores[i] < scores[j])
                    << gains[i] << " vs " << gains[j];
    }
}

// Negative lengths and turns: the order still matches the cost_t order (the
// turns are shifted by their minimum, softmax does not depend on a common
// shift of the scores).
TEST(RisHeuristic, SetFlatScores_PreservesOrderWithNegativeComponents) {
    std::mt19937 gen(11);
    std::uniform_int_distribution<int> len(-20, 20), turns(-8, 8);
    for (int trial = 0; trial < 300; ++trial) {
        std::vector<cost_t> gains(6);
        for (auto& g : gains) g = {len(gen), turns(gen)};
        auto scores = ris_heuristic::setFlatScores(gains);
        for (std::size_t i = 0; i < gains.size(); ++i)
            for (std::size_t j = 0; j < gains.size(); ++j)
                ASSERT_EQ(gains[i] < gains[j], scores[i] < scores[j])
                    << gains[i] << " vs " << gains[j];
    }
}

// Equal turns: the longer gain must score higher, {10,3} > {1,3}.
TEST(RisHeuristic, SetFlatScores_EqualTurns_LengthOrders) {
    std::vector<cost_t> gains = {{10,3},{1,3}};
    auto scores = ris_heuristic::setFlatScores(gains);
    EXPECT_GT(scores[0], scores[1]);
}

// Length dominates turns: {2,0} > {1,3} as cost_t.
TEST(RisHeuristic, SetFlatScores_LengthDominatesTurns) {
    std::vector<cost_t> gains = {{2,0},{1,3}};
    auto scores = ris_heuristic::setFlatScores(gains);
    EXPECT_GT(scores[0], scores[1]);
}

// Large values: M * length must not overflow int before conversion to double.
TEST(RisHeuristic, SetFlatScores_LargeValues_NoOverflow) {
    std::vector<cost_t> gains = {{2'000'000, 2000},{1'999'999, 0}};
    auto scores = ris_heuristic::setFlatScores(gains);
    EXPECT_DOUBLE_EQ(scores[0], 2001.0 * 2'000'000 + 2000);
    EXPECT_GT(scores[0], scores[1]);
}

// ============================================================================
// softmaxSample — additional edge cases
// ============================================================================

TEST(RisHeuristic, SoftmaxSample_DoesNotModifyGains) {
    std::vector<cost_t> gains = {{5,3},{10,1}};
    auto copy = gains;
    std::mt19937 gen(1);
    ris_heuristic::softmaxSample(gains, gen);
    EXPECT_EQ(gains, copy);
}

TEST(RisHeuristic, SoftmaxSample_AllZeroGains_AllIndicesReachable) {
    std::vector<cost_t> gains(3, cost_t{0,0});
    std::mt19937 gen(3);
    std::set<int> seen;
    for (int t = 0; t < 300; ++t) seen.insert(ris_heuristic::softmaxSample(gains, gen));
    EXPECT_EQ(seen.size(), 3u);
}

// Equal length, more turns gained → higher score → drawn more often.
TEST(RisHeuristic, SoftmaxSample_TurnsBreakLengthTie) {
    std::vector<cost_t> gains = {{5,0},{5,1}};
    std::mt19937 gen(11);
    int ones = 0;
    for (int t = 0; t < 2000; ++t)
        if (ris_heuristic::softmaxSample(gains, gen) == 1) ++ones;
    EXPECT_GT(ones, 1000);
}

// Huge scores must not overflow exp(): softmax is computed relative to the max.
// Softmax uses the raw scores (temperature 1), so only the difference matters:
// scores 100000 and 100001 give P(1) = 1 / (1 + e^-1) ~ 0.731.
TEST(RisHeuristic, SoftmaxSample_HugeScores_Stable) {
    std::vector<cost_t> gains = {{100000,0},{100001,0}};
    std::mt19937 gen(4);
    const int draws = 2000;
    int ones = 0;
    for (int t = 0; t < draws; ++t) {
        int idx = ris_heuristic::softmaxSample(gains, gen);
        ASSERT_TRUE(idx == 0 || idx == 1);
        if (idx == 1) ++ones;
    }
    const double expected = 1.0 / (1.0 + std::exp(-1.0));
    EXPECT_NEAR(static_cast<double>(ones) / draws, expected, 0.04);   // ~4 standard deviations
}

// A large gap in gains: the larger gain dominates every draw.
TEST(RisHeuristic, SoftmaxSample_LargeGap_LargestDominates) {
    std::vector<cost_t> gains = {{1,0},{1000,0}};
    std::mt19937 gen(8);
    for (int t = 0; t < 50; ++t)
        ASSERT_EQ(ris_heuristic::softmaxSample(gains, gen), 1);
}

// Same seed → same draw (reproducibility relied on by the tests above).
TEST(RisHeuristic, SoftmaxSample_SameSeedSameDraws) {
    std::vector<cost_t> gains = {{1,0},{1,0},{1,0}};
    std::mt19937 a(77), b(77);
    for (int t = 0; t < 20; ++t)
        ASSERT_EQ(ris_heuristic::softmaxSample(gains, a),
                  ris_heuristic::softmaxSample(gains, b));
}

// ============================================================================
// getCoverIntervals — merge selected rows into one schedule
// ============================================================================

namespace {
IntervalSet row_set(std::initializer_list<Interval> items) {
    IntervalSet set;
    for (auto& i : items) set.push(i);
    return set;
}
std::vector<int> set_ids(const IntervalSet& set) {
    std::vector<int> out;
    for (auto& i : set) out.push_back(i.id);
    return out;
}
} // anonymous namespace

TEST(RisHeuristic, GetCoverIntervals_EmptyCover_EmptySchedule) {
    GroupedIntervalSets rows;
    rows.push(row_set({{{3,0}, 0, 1, 0, false, 0}}));
    auto schedule = ris_heuristic::getCoverIntervals({}, rows);
    EXPECT_TRUE(schedule.empty());
    EXPECT_EQ(schedule.getValue(), (cost_t{0,0}));
}

TEST(RisHeuristic, GetCoverIntervals_MergesSelectedRowsInCoverOrder) {
    GroupedIntervalSets rows;
    rows.push(row_set({{{3,0}, 0, 1, 0, false, 0}, {{1,0}, 3, 4, 0, false, 1}}));
    rows.push(row_set({{{9,0}, 0, 1, 1, false, 2}}));
    rows.push(row_set({{{2,1}, 0, 1, 2, false, 3}}));
    auto schedule = ris_heuristic::getCoverIntervals({2, 0}, rows);
    EXPECT_EQ(set_ids(schedule), (std::vector<int>{3, 0, 1}));
    EXPECT_EQ(schedule.getValue(), (cost_t{6,1}));
}

TEST(RisHeuristic, GetCoverIntervals_EmptyRowSelected_NoContribution) {
    GroupedIntervalSets rows;
    rows.push(IntervalSet{});
    rows.push(row_set({{{4,0}, 0, 1, 1, false, 7}}));
    auto schedule = ris_heuristic::getCoverIntervals({0, 1}, rows);
    EXPECT_EQ(set_ids(schedule), (std::vector<int>{7}));
    EXPECT_EQ(schedule.getValue(), (cost_t{4,0}));
}

// ============================================================================
// clear() and solve_selection — degenerate splits
// ============================================================================

TEST(RisHeuristic, Clear_OnFreshInstance_IsNoOp) {
    std::vector<cost_t> gains = {{5,0}};
    std::map<int, Interval> coords = {{0, {gains[0], 0, 0, 0, false, 0}}};
    auto h = make_heuristic(1, gains, coords);
    h.clear();
    EXPECT_EQ(h.solve_selection(), (std::vector<int>{0}));
}

TEST(RisHeuristic, Clear_BetweenCalls_SameResultAsFresh) {
    std::vector<cost_t> gains = {{5,0},{3,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 5, 0, false, 0}},
        {1, {gains[1], 2, 8, 0, false, 1}}
    };
    auto h = make_heuristic(1, gains, coords);
    auto first = h.solve_selection();
    h.clear();
    EXPECT_EQ(h.solve_selection(), first);
}

// Splitter producing no group at all → empty selection.
TEST(RisHeuristic, SplitterNoGroups_SelectionIsEmpty) {
    std::vector<cost_t> gains = {{5,0}};
    std::map<int, Interval> coords = {{0, {gains[0], 0, 0, 0, false, 0}}};
    auto no_groups = [](const std::vector<int>&, std::vector<std::vector<int>>&) {};
    ris_harness h(1, 1, no_groups, make_filter(gains), make_interval_fn(coords));
    EXPECT_TRUE(h.solve_selection().empty());
}

// An empty group next to a non-empty one must be skipped, not crash.
TEST(RisHeuristic, SplitterWithEmptyGroup_SelectionFromNonEmptyGroup) {
    std::vector<cost_t> gains = {{5,0}};
    std::map<int, Interval> coords = {{0, {gains[0], 0, 0, 0, false, 0}}};
    auto with_empty = [](const std::vector<int>& c, std::vector<std::vector<int>>& s) {
        s = {{}, c, {}};
    };
    ris_harness h(1, 1, with_empty, make_filter(gains), make_interval_fn(coords));
    EXPECT_EQ(h.solve_selection(), (std::vector<int>{0}));
}

// Interval function producing no interval for a group → that group is skipped.
TEST(RisHeuristic, IntervalFunctionNoIntervals_SelectionIsEmpty) {
    std::vector<cost_t> gains = {{5,0}};
    auto no_intervals = [](const std::vector<int>&, IntervalSet&) {};
    ris_harness h(1, 1, make_single_group_splitter(), make_filter(gains), no_intervals);
    EXPECT_TRUE(h.solve_selection().empty());
}

// The selection always comes from a single group (one softmax draw).
TEST(RisHeuristic, TwoGroups_SelectionFromSingleGroup) {
    std::vector<cost_t> gains = {{5,0},{5,0},{5,0},{5,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}}, {1, {gains[1], 2, 2, 0, false, 1}},
        {2, {gains[2], 0, 0, 0, false, 2}}, {3, {gains[3], 2, 2, 0, false, 3}}
    };
    auto halves = [](const std::vector<int>& c, std::vector<std::vector<int>>& s) {
        s = {{}, {}};
        for (int id : c) s[id < 2 ? 0 : 1].push_back(id);
    };
    for (int t = 0; t < 20; ++t) {
        ris_harness h(1, 4, halves, make_filter(gains), make_interval_fn(coords));
        std::set<int> sel;
        for (int id : h.solve_selection()) sel.insert(id);
        EXPECT_TRUE(sel == std::set<int>({0,1}) || sel == std::set<int>({2,3}));
    }
}

// Negative row indices: rows are indexed relative to the minimum y.
TEST(RisHeuristic, NegativeRows_SpacingRespected) {
    std::vector<cost_t> gains = {{5,0},{5,0},{5,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, -3, false, 0}},
        {1, {gains[1], 0, 0, -2, false, 1}},
        {2, {gains[2], 0, 0,  0, false, 2}}
    };
    auto h = make_heuristic(2, gains, coords);
    auto selection = h.solve_selection();
    std::set<int> sel(selection.begin(), selection.end());
    EXPECT_EQ(sel.size(), 2u);
    EXPECT_TRUE(sel.count(2)) << "row y=0 is compatible with either y=-3 or y=-2";
    EXPECT_FALSE(sel.count(0) && sel.count(1)) << "rows -3 and -2 are too close";
}

// Spacing larger than the row span: only one row can be used.
TEST(RisHeuristic, SpacingLargerThanRowSpan_OneRowSelected) {
    std::vector<cost_t> gains = {{5,0},{7,0}};
    std::map<int, Interval> coords = {
        {0, {gains[0], 0, 0, 0, false, 0}},
        {1, {gains[1], 0, 0, 2, false, 1}}
    };
    auto h = make_heuristic(10, gains, coords);
    EXPECT_EQ(h.solve_selection(), (std::vector<int>{1}));
}
