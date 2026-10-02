#include <gtest/gtest.h>
#include <random>
#include <set>
#include "../../include/heuristics/IntervalScheduler.hpp"

// ============================================================================
// IntervalScheduler unit tests
//
// What is being tested:
//   IntervalScheduler solves weighted interval scheduling on an IntervalSet
//   whose intervals are sorted by right end-point x2. Two intervals a, b
//   (a.x2 <= b.x2) are compatible iff a.x2 < b.x1 (closed intervals: sharing
//   an end-point counts as overlap).
//
//   Public API: constructor(const IntervalSet&), solve(), getSolution(), getCost(),
//   static cmpEndPoint().
//
// Test strategy:
//   - Hand-built instances for the boundary semantics and tie handling.
//   - Rows produced by IntervalSet::groupByRows (how ris_heuristic uses it).
//   - Exhaustive brute force on many small random instances with heavy ties.
// ============================================================================

namespace {

// Copy the solution returned by getSolution() into a vector so the helpers
// below (ids, sum_gains, non_overlapping) can sort and inspect it.
std::vector<Interval> to_vec(const IntervalSet& set) {
    return {set.begin(), set.end()};
}

// IntervalScheduler takes an IntervalSet (kept by reference: the set must
// outlive the scheduler).
IntervalSet to_set(const std::vector<Interval>& v) {
    IntervalSet set;
    for (auto& i : v) set.push(i);
    return set;
}

Interval make_iv(int x1, int x2, cost_t gain, int id, int y = 0) {
    return Interval{gain, x1, x2, y, false, id};
}

void sort_by_end(std::vector<Interval>& v) {
    std::stable_sort(v.begin(), v.end(),
                     [](const Interval& a, const Interval& b){ return a.x2 < b.x2; });
}

std::set<int> ids(const std::vector<Interval>& v) {
    std::set<int> out;
    for (auto& i : v) out.insert(i.id);
    return out;
}

cost_t sum_gains(const std::vector<Interval>& v) {
    cost_t total{0, 0};
    for (auto& i : v) total += i.gain;
    return total;
}

::testing::AssertionResult non_overlapping(std::vector<Interval> v) {
    sort_by_end(v);
    for (std::size_t k = 1; k < v.size(); ++k)
        if (v[k - 1].x2 >= v[k].x1)
            return ::testing::AssertionFailure()
                    << "intervals id=" << v[k - 1].id << " [" << v[k - 1].x1 << "," << v[k - 1].x2
                    << "] and id=" << v[k].id << " [" << v[k].x1 << "," << v[k].x2 << "] overlap";
    if (ids(v).size() != v.size())
        return ::testing::AssertionFailure() << "duplicate Interval in solution";
    return ::testing::AssertionSuccess();
}

cost_t brute_force(const std::vector<Interval>& v) {
    const int n = static_cast<int>(v.size());
    cost_t best{0, 0};
    for (unsigned mask = 0; mask < (1u << n); ++mask) {
        std::vector<Interval> subset;
        for (int i = 0; i < n; ++i)
            if (mask & (1u << i)) subset.push_back(v[i]);
        if (non_overlapping(subset)) best = std::max(best, sum_gains(subset));
    }
    return best;
}

std::string to_string(const std::vector<Interval>& v) {
    std::ostringstream os;
    for (auto& i : v) os << "[" << i.x1 << "," << i.x2 << "]" << i.gain << " ";
    return os.str();
}

} // anonymous namespace

// ============================================================================
// cmpEndPoint
// ============================================================================

TEST(IntervalScheduler, CmpEndPoint_StrictlyLessThanBound) {
    Interval iv = make_iv(0, 5, {1,0}, 0);
    EXPECT_TRUE(IntervalScheduler::cmpEndPoint(iv, 6));
    EXPECT_FALSE(IntervalScheduler::cmpEndPoint(iv, 5));
    EXPECT_FALSE(IntervalScheduler::cmpEndPoint(iv, 4));
}

// ============================================================================
// Degenerate ranges
// ============================================================================

TEST(IntervalScheduler, EmptyRange_SolveIsNoOp) {
    std::vector<Interval> v;
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_TRUE(to_vec(s.getSolution()).empty());
}

// getCost() on an empty range must not read stateCosts.back() of an empty
// vector. Run in a child process so a crash cannot abort the test binary.
TEST(IntervalScheduler, EmptyRange_CostIsZero) {
    EXPECT_EXIT({
        std::vector<Interval> v;
        IntervalSet iset = to_set(v);
        IntervalScheduler s(iset);
        s.solve();
        std::exit(s.getCost() == cost_t{0,0} ? 0 : 1);
    }, ::testing::ExitedWithCode(0), "");
}

TEST(IntervalScheduler, SingleInterval_Selected) {
    std::vector<Interval> v = {make_iv(2, 4, {3,1}, 7)};
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    ASSERT_EQ(to_vec(s.getSolution()).size(), 1u);
    EXPECT_EQ(to_vec(s.getSolution())[0].id, 7);
    EXPECT_EQ(s.getCost(), (cost_t{3,1}));
}

// Degenerate point interval x1 == x2.
TEST(IntervalScheduler, PointIntervals_Disjoint_AllSelected) {
    std::vector<Interval> v = {make_iv(0,0,{1,0},0), make_iv(1,1,{1,0},1), make_iv(2,2,{1,0},2)};
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_EQ(ids(to_vec(s.getSolution())), (std::set<int>{0,1,2}));
    EXPECT_EQ(s.getCost(), (cost_t{3,0}));
}

// ============================================================================
// Overlap semantics
// ============================================================================

TEST(IntervalScheduler, Disjoint_AllSelected) {
    std::vector<Interval> v = {make_iv(0,1,{5,0},0), make_iv(3,4,{5,0},1), make_iv(6,7,{5,0},2)};
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_EQ(ids(to_vec(s.getSolution())), (std::set<int>{0,1,2}));
    EXPECT_EQ(s.getCost(), (cost_t{15,0}));
}

// x2 == x1 of the next interval: closed intervals share a point → overlap.
TEST(IntervalScheduler, SharedEndPoint_IsOverlap) {
    std::vector<Interval> v = {make_iv(0,3,{5,0},0), make_iv(3,6,{4,0},1)};
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_EQ(ids(to_vec(s.getSolution())), (std::set<int>{0}));
    EXPECT_EQ(s.getCost(), (cost_t{5,0}));
}

// x2 + 1 == x1: adjacent but disjoint → both selectable.
TEST(IntervalScheduler, AdjacentIntervals_BothSelected) {
    std::vector<Interval> v = {make_iv(0,3,{5,0},0), make_iv(4,6,{4,0},1)};
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_EQ(ids(to_vec(s.getSolution())), (std::set<int>{0,1}));
    EXPECT_EQ(s.getCost(), (cost_t{9,0}));
}

TEST(IntervalScheduler, HigherGainOverlapWins) {
    std::vector<Interval> v = {make_iv(0,5,{3,0},0), make_iv(2,8,{10,0},1)};
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_EQ(ids(to_vec(s.getSolution())), (std::set<int>{1}));
    EXPECT_EQ(s.getCost(), (cost_t{10,0}));
}

// Two short compatible intervals beat a long one covering both.
TEST(IntervalScheduler, PairBeatsLong) {
    std::vector<Interval> v = {make_iv(0,2,{5,0},0), make_iv(4,6,{5,0},1), make_iv(0,6,{8,0},2)};
    sort_by_end(v);
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_EQ(ids(to_vec(s.getSolution())), (std::set<int>{0,1}));
    EXPECT_EQ(s.getCost(), (cost_t{10,0}));
}

// A long interval beats two short ones when its gain is larger.
TEST(IntervalScheduler, LongBeatsPair) {
    std::vector<Interval> v = {make_iv(0,2,{3,0},0), make_iv(4,6,{3,0},1), make_iv(0,6,{8,0},2)};
    sort_by_end(v);
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_EQ(ids(to_vec(s.getSolution())), (std::set<int>{2}));
    EXPECT_EQ(s.getCost(), (cost_t{8,0}));
}

// Best interval is the first one and has no predecessor; later intervals all
// overlap it. Backward pass must stop cleanly at index 0.
TEST(IntervalScheduler, FirstIntervalAloneIsOptimal) {
    std::vector<Interval> v = {make_iv(0,10,{9,0},0), make_iv(5,11,{1,0},1), make_iv(8,12,{1,0},2)};
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_EQ(ids(to_vec(s.getSolution())), (std::set<int>{0}));
    EXPECT_EQ(s.getCost(), (cost_t{9,0}));
}

// cost_t lexicographic ordering: turns break length ties.
TEST(IntervalScheduler, LexicographicGain_TurnsBreakTie) {
    std::vector<Interval> v = {make_iv(0,5,{4,1},0), make_iv(1,6,{4,3},1)};
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_EQ(ids(to_vec(s.getSolution())), (std::set<int>{1}));
}

// Identical overlapping intervals: exactly one is chosen.
TEST(IntervalScheduler, IdenticalIntervals_ExactlyOneSelected) {
    std::vector<Interval> v = {make_iv(0,3,{2,0},0), make_iv(0,3,{2,0},1), make_iv(0,3,{2,0},2)};
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_EQ(to_vec(s.getSolution()).size(), 1u);
    EXPECT_EQ(s.getCost(), (cost_t{2,0}));
}

// Zero gain intervals: cost zero and solution remains non-overlapping.
TEST(IntervalScheduler, ZeroGains_CostZeroAndFeasible) {
    std::vector<Interval> v = {make_iv(0,3,{0,0},0), make_iv(2,5,{0,0},1), make_iv(7,8,{0,0},2)};
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_EQ(s.getCost(), (cost_t{0,0}));
    EXPECT_TRUE(non_overlapping(to_vec(s.getSolution())));
}

// ============================================================================
// Rows produced by IntervalSet::groupByRows (as used by ris_heuristic)
// ============================================================================

// Scheduling one row taken from groupByRows() must only select intervals of
// that row.
TEST(IntervalScheduler, RowFromGroupByRows_OnlyRowElementsSelected) {
    IntervalSet all = to_set({
        make_iv(0,1,{100,0},0, 0),
        make_iv(0,1,{1,0},1, 1), make_iv(3,4,{2,0},2, 1), make_iv(3,9,{3,0},3, 1),
        make_iv(0,1,{100,0},4, 2)
    });
    GroupedIntervalSets rows = all.groupByRows();
    const IntervalSet* row1 = nullptr;
    for (auto& row : rows)
        if (!row.empty() && row.front().y == 1) row1 = &row;
    ASSERT_NE(row1, nullptr) << "groupByRows produced no row with y=1";

    IntervalScheduler s(*row1);
    s.solve();
    auto sel = ids(to_vec(s.getSolution()));
    for (int id : sel) {
        EXPECT_GE(id, 1);
        EXPECT_LE(id, 3);
    }
    EXPECT_EQ(s.getCost(), (cost_t{4,0}));  // {1,3}: gain 1 + 3
    EXPECT_EQ(sel, (std::set<int>{1,3}));
}

// ============================================================================
// Getter consistency / repeated solve
// ============================================================================

TEST(IntervalScheduler, CostEqualsSumOfSolutionGains) {
    std::vector<Interval> v = {make_iv(0,2,{2,1},0), make_iv(1,4,{3,0},1),
                               make_iv(3,5,{2,2},2), make_iv(6,9,{1,1},3)};
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    EXPECT_EQ(sum_gains(to_vec(s.getSolution())), s.getCost());
}

TEST(IntervalScheduler, SolveTwice_SolutionNotDuplicated) {
    std::vector<Interval> v = {make_iv(0,1,{5,0},0), make_iv(3,4,{5,0},1)};
    IntervalSet iset = to_set(v);
    IntervalScheduler s(iset);
    s.solve();
    auto first = to_vec(s.getSolution());
    s.solve();
    EXPECT_EQ(first.size(), to_vec(s.getSolution()).size());
    EXPECT_TRUE(non_overlapping(to_vec(s.getSolution())));
}

// ============================================================================
// Randomised comparison against brute force
// ============================================================================

TEST(IntervalScheduler, Random_MatchesBruteForce) {
    std::mt19937 gen(4242);
    std::uniform_int_distribution<int> pos(0, 12), width(0, 4), len(0, 3), turns(0, 2);
    for (int trial = 0; trial < 3000; ++trial) {
        int n = 1 + trial % 10;
        std::vector<Interval> v;
        for (int k = 0; k < n; ++k) {
            int x1 = pos(gen);
            v.push_back(make_iv(x1, x1 + width(gen), {len(gen), turns(gen)}, k));
        }
        sort_by_end(v);

        IntervalSet iset = to_set(v);

        IntervalScheduler s(iset);
        s.solve();
        const auto& sol = to_vec(s.getSolution());
        cost_t expected = brute_force(v);

        SCOPED_TRACE("intervals=" + to_string(v));
        ASSERT_EQ(s.getCost(), expected);
        ASSERT_TRUE(non_overlapping(sol));
        ASSERT_EQ(sum_gains(sol), expected);
    }
}
