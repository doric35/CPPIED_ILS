#include <gtest/gtest.h>
#include <set>
#include "../../include/heuristics/ris_heuristic.hpp"

// ============================================================================
// Interval / IntervalSet / GroupedIntervalSets unit tests
//
// What is being tested:
//   - Interval: aggregate describing one candidate segment (gain, x1, x2, y,
//     rotated, id).
//   - IntervalSet: vector-like container of Interval that keeps a running
//     total `value` (sum of gains) in sync with its content.
//   - GroupedIntervalSets: vector-like container of IntervalSet exposing the
//     per-group totals through getGroupWiseValues().
//
// Invariant checked throughout: after any mutating member call, getValue()
// equals the sum of the gains currently stored.
// ============================================================================

namespace {

Interval make_iv(int x1, int x2, cost_t gain, int id, int y = 0) {
    return Interval{gain, x1, x2, y, false, id};
}

cost_t sum_gains(const IntervalSet& set) {
    cost_t total{0, 0};
    for (auto& i : set) total += i.gain;
    return total;
}

std::vector<int> ids(const IntervalSet& set) {
    std::vector<int> out;
    for (auto& i : set) out.push_back(i.id);
    return out;
}

IntervalSet make_set(std::initializer_list<Interval> items) {
    IntervalSet set;
    for (auto& i : items) set.push(i);
    return set;
}

} // anonymous namespace

// ============================================================================
// Interval
// ============================================================================

TEST(Interval, AggregateInitialisationOrder) {
    Interval iv{{3,1}, 2, 5, 7, true, 42};
    EXPECT_EQ(iv.gain, (cost_t{3,1}));
    EXPECT_EQ(iv.x1, 2);
    EXPECT_EQ(iv.x2, 5);
    EXPECT_EQ(iv.y, 7);
    EXPECT_TRUE(iv.rotated);
    EXPECT_EQ(iv.id, 42);
}

// ============================================================================
// IntervalSet — construction and insertion
// ============================================================================

TEST(IntervalSet, DefaultConstructed_EmptyWithZeroValue) {
    IntervalSet set;
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.size(), 0u);
    EXPECT_EQ(set.getValue(), (cost_t{0,0}));
    EXPECT_EQ(set.begin(), set.end());
}

TEST(IntervalSet, PushInterval_UpdatesSizeValueAndOrder) {
    IntervalSet set;
    set.push(make_iv(0, 1, {2,1}, 0));
    set.push(make_iv(3, 4, {5,0}, 1));
    EXPECT_FALSE(set.empty());
    EXPECT_EQ(set.size(), 2u);
    EXPECT_EQ(set.getValue(), (cost_t{7,1}));
    EXPECT_EQ(ids(set), (std::vector<int>{0, 1}));
    EXPECT_EQ(set.front().id, 0);
    EXPECT_EQ(set.back().id, 1);
    EXPECT_EQ(set[1].x1, 3);
}

TEST(IntervalSet, PushBackLvalue_UpdatesValue) {
    IntervalSet set;
    const Interval iv = make_iv(0, 1, {4,2}, 9);
    set.push_back(iv);
    EXPECT_EQ(set.size(), 1u);
    EXPECT_EQ(set.getValue(), (cost_t{4,2}));
}

TEST(IntervalSet, PushBackRvalue_UpdatesValue) {
    IntervalSet set;
    set.push_back(make_iv(0, 1, {4,2}, 9));
    set.push_back(Interval{{1,1}, 3, 3, 0, false, 10});
    EXPECT_EQ(set.size(), 2u);
    EXPECT_EQ(set.getValue(), (cost_t{5,3}));
}

// push_back is what std::back_inserter uses: algorithms must keep value in sync.
TEST(IntervalSet, BackInserter_KeepsValueInSync) {
    std::vector<Interval> src = {make_iv(0,1,{1,0},0), make_iv(2,3,{2,0},1)};
    IntervalSet set;
    std::copy(src.begin(), src.end(), std::back_inserter(set));
    EXPECT_EQ(set.size(), 2u);
    EXPECT_EQ(set.getValue(), (cost_t{3,0}));
}

TEST(IntervalSet, NegativeGains_Accumulate) {
    auto set = make_set({make_iv(0,0,{-3,1},0), make_iv(1,1,{1,-2},1)});
    EXPECT_EQ(set.getValue(), (cost_t{-2,-1}));
}

// ============================================================================
// IntervalSet — merging another set
// ============================================================================

TEST(IntervalSet, PushSet_AppendsIntervalsInOrder) {
    auto a = make_set({make_iv(0,1,{1,0},0)});
    auto b = make_set({make_iv(2,3,{2,0},1), make_iv(4,5,{3,0},2)});
    a.push(b);
    EXPECT_EQ(ids(a), (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(b.size(), 2u) << "source set must be left untouched";
}

// The merged set's value must include the gains of the appended intervals.
TEST(IntervalSet, PushSet_UpdatesValue) {
    auto a = make_set({make_iv(0,1,{1,0},0)});
    auto b = make_set({make_iv(2,3,{2,1},1), make_iv(4,5,{3,0},2)});
    a.push(b);
    EXPECT_EQ(a.getValue(), (cost_t{6,1}));
    EXPECT_EQ(a.getValue(), sum_gains(a));
}

TEST(IntervalSet, PushSet_IntoEmptySet_ValueEqualsSource) {
    IntervalSet a;
    auto b = make_set({make_iv(2,3,{2,1},1)});
    a.push(b);
    EXPECT_EQ(a.size(), 1u);
    EXPECT_EQ(a.getValue(), b.getValue());
}

TEST(IntervalSet, PushEmptySet_NoChange) {
    auto a = make_set({make_iv(0,1,{1,0},0)});
    a.push(IntervalSet{});
    EXPECT_EQ(a.size(), 1u);
    EXPECT_EQ(a.getValue(), (cost_t{1,0}));
}

// ============================================================================
// IntervalSet — removal and recomputation
// ============================================================================

TEST(IntervalSet, Clear_ResetsSizeAndValue) {
    auto set = make_set({make_iv(0,1,{1,0},0), make_iv(2,3,{2,0},1)});
    set.clear();
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.getValue(), (cost_t{0,0}));
}

TEST(IntervalSet, EraseIf_RemovesMatchingAndRecomputesValue) {
    auto set = make_set({make_iv(0,1,{1,0},0), make_iv(2,3,{2,0},1), make_iv(4,5,{3,0},2)});
    set.erase_if([](const Interval& i){ return i.id == 1; });
    EXPECT_EQ(ids(set), (std::vector<int>{0, 2}));
    EXPECT_EQ(set.getValue(), (cost_t{4,0}));
}

TEST(IntervalSet, EraseIf_NoMatch_Unchanged) {
    auto set = make_set({make_iv(0,1,{1,0},0)});
    set.erase_if([](const Interval&){ return false; });
    EXPECT_EQ(set.size(), 1u);
    EXPECT_EQ(set.getValue(), (cost_t{1,0}));
}

TEST(IntervalSet, EraseIf_AllMatch_EmptyWithZeroValue) {
    auto set = make_set({make_iv(0,1,{1,0},0), make_iv(2,3,{2,0},1)});
    set.erase_if([](const Interval&){ return true; });
    EXPECT_TRUE(set.empty());
    EXPECT_EQ(set.getValue(), (cost_t{0,0}));
}

// operator[] is read-only: it must not allow writing an interval in place.
TEST(IntervalSet, IndexOperator_IsReadOnly) {
    using Ref = decltype(std::declval<IntervalSet&>()[0]);
    EXPECT_TRUE(std::is_const_v<std::remove_reference_t<Ref>>);
}

// setValue() recomputes the cached value from the content: idempotent.
TEST(IntervalSet, SetValue_RecomputesSameValue) {
    auto set = make_set({make_iv(0,1,{1,0},0), make_iv(2,3,{2,0},1)});
    set.setValue();
    EXPECT_EQ(set.getValue(), (cost_t{3,0}));
}

// begin()/end() only hand out const iterators, even on a non-const set, so
// the cached value cannot be bypassed.
TEST(IntervalSet, Iterators_AreConst) {
    using It = decltype(std::declval<IntervalSet&>().begin());
    EXPECT_TRUE((std::is_same_v<It, IntervalSet::const_iterator>));
    EXPECT_TRUE(std::is_const_v<std::remove_reference_t<decltype(*std::declval<It>())>>);
}

// front()/back() are read-only, so they cannot desynchronise the value.
TEST(IntervalSet, FrontBack_AreReadOnly) {
    using Front = decltype(std::declval<IntervalSet&>().front());
    using Back  = decltype(std::declval<IntervalSet&>().back());
    EXPECT_TRUE(std::is_const_v<std::remove_reference_t<Front>>);
    EXPECT_TRUE(std::is_const_v<std::remove_reference_t<Back>>);
}

// ============================================================================
// IntervalSet — iterators and algorithms
// ============================================================================

// groupByRows() sorts the set in place (by y, then x2) without changing value.
TEST(IntervalSet, GroupByRows_SortsInPlaceWithoutChangingValue) {
    auto set = make_set({make_iv(4,5,{3,0},2), make_iv(0,1,{1,0},0), make_iv(2,3,{2,0},1)});
    set.groupByRows();
    EXPECT_EQ(ids(set), (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(set.getValue(), (cost_t{6,0}));
}

TEST(IntervalSet, ConstIterators_UsableWithLowerBound) {
    const auto set = make_set({make_iv(0,1,{1,0},0), make_iv(2,3,{1,0},1), make_iv(4,6,{1,0},2)});
    auto it = std::lower_bound(set.begin(), set.end(), 4, IntervalScheduler::cmpEndPoint);
    ASSERT_NE(it, set.end());
    EXPECT_EQ(it->id, 2);
    EXPECT_EQ(set.size(), 3u);
}

// Read-only accessors should be callable on a const IntervalSet, e.g. on the
// `const IntervalSet&` returned by IntervalScheduler::getSolution().
TEST(IntervalSet, ConstInterface_ReadAccessorsAvailable) {
    // Template-dependent so an unsatisfied requirement yields false instead of
    // a compile error.
    auto check = []<typename S>(const S&) {
        return std::array<bool, 4>{
            requires(const S& s) { s.empty(); },
            requires(const S& s) { s[0]; },
            requires(const S& s) { s.front(); },
            requires(const S& s) { s.back(); }};
    };
    auto [has_empty, has_index, has_front, has_back] = check(IntervalSet{});
    EXPECT_TRUE(has_empty) << "IntervalSet::empty() is not const";
    EXPECT_TRUE(has_index) << "IntervalSet::operator[] has no const overload";
    EXPECT_TRUE(has_front) << "IntervalSet::front() has no const overload";
    EXPECT_TRUE(has_back)  << "IntervalSet::back() has no const overload";
}

// ============================================================================
// GroupedIntervalSets
// ============================================================================

TEST(GroupedIntervalSets, DefaultConstructed_Empty) {
    GroupedIntervalSets groups;
    EXPECT_TRUE(groups.empty());
    EXPECT_EQ(groups.size(), 0u);
    EXPECT_TRUE(groups.getGroupWiseValues().empty());
}

TEST(GroupedIntervalSets, SizedConstructor_CreatesEmptyGroups) {
    GroupedIntervalSets groups(3);
    EXPECT_EQ(groups.size(), 3u);
    for (auto& g : groups) EXPECT_TRUE(g.empty());
    EXPECT_EQ(groups.getGroupWiseValues(), (std::vector<cost_t>(3, cost_t{0,0})));
}

// operator[] and back() are read-only.
TEST(GroupedIntervalSets, IndexAndBack_AreReadOnly) {
    using Index = decltype(std::declval<GroupedIntervalSets&>()[0]);
    using Back  = decltype(std::declval<GroupedIntervalSets&>().back());
    EXPECT_TRUE(std::is_const_v<std::remove_reference_t<Index>>);
    EXPECT_TRUE(std::is_const_v<std::remove_reference_t<Back>>);
}

// ris_heuristic::setIntervalSets builds each IntervalSet then pushes it; the
// group-wise values must reflect the pushed content.
TEST(GroupedIntervalSets, FilledThroughPush_GroupWiseValuesReflectContent) {
    GroupedIntervalSets groups;
    groups.push(make_set({make_iv(0,1,{2,0},0)}));
    groups.push(make_set({make_iv(0,1,{1,1},1), make_iv(3,4,{1,1},2)}));
    EXPECT_EQ(groups.getGroupWiseValues(), (std::vector<cost_t>{{2,0},{2,2}}));
}

TEST(GroupedIntervalSets, Push_AppendsGroup) {
    GroupedIntervalSets groups;
    groups.push(make_set({make_iv(0,1,{2,0},0)}));
    groups.push(make_set({make_iv(0,1,{5,1},1)}));
    EXPECT_EQ(groups.size(), 2u);
    EXPECT_EQ(groups.back().front().id, 1);
    EXPECT_EQ(groups[0].front().id, 0);
    EXPECT_EQ(groups.getGroupWiseValues(), (std::vector<cost_t>{{2,0},{5,1}}));
}

TEST(GroupedIntervalSets, Pop_RemovesLastGroup) {
    GroupedIntervalSets groups;
    groups.push(make_set({make_iv(0,1,{2,0},0)}));
    groups.push(IntervalSet{});
    groups.pop();
    EXPECT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups.getGroupWiseValues(), (std::vector<cost_t>{{2,0}}));
}

TEST(GroupedIntervalSets, Clear_RemovesAllGroups) {
    GroupedIntervalSets groups(4);
    groups.clear();
    EXPECT_TRUE(groups.empty());
}

TEST(GroupedIntervalSets, EraseIf_RemovesEmptyGroups) {
    GroupedIntervalSets groups;
    groups.push(IntervalSet{});
    groups.push(make_set({make_iv(0,1,{2,0},7)}));
    groups.push(IntervalSet{});
    groups.erase_if([](const IntervalSet& s){ return s.size() == 0; });
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].front().id, 7);
}

TEST(GroupedIntervalSets, Reserve_DoesNotChangeSize) {
    GroupedIntervalSets groups;
    groups.reserve(10);
    EXPECT_EQ(groups.size(), 0u);
}

TEST(GroupedIntervalSets, Iteration_VisitsGroupsInOrder) {
    GroupedIntervalSets groups;
    for (int k = 0; k < 3; ++k) groups.push(make_set({make_iv(0,1,{1,0},k)}));
    std::vector<int> seen;
    for (auto& g : groups) seen.push_back(g.front().id);
    EXPECT_EQ(seen, (std::vector<int>{0, 1, 2}));
}

// GroupedIntervalSets keeps a running total `value` that must always equal the
// sum of the group-wise values.
namespace {
::testing::AssertionResult grouped_value_in_sync(GroupedIntervalSets& groups) {
    cost_t total{0, 0};
    for (auto& v : groups.getGroupWiseValues()) total += v;
    if (groups.getValue() == total) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
            << "getValue()=" << groups.getValue() << " but groups sum to " << total;
}
} // anonymous namespace

TEST(GroupedIntervalSets, Value_ZeroWhenEmpty) {
    GroupedIntervalSets groups;
    EXPECT_EQ(groups.getValue(), (cost_t{0,0}));
    GroupedIntervalSets sized(3);
    EXPECT_EQ(sized.getValue(), (cost_t{0,0}));
}

TEST(GroupedIntervalSets, Value_SumOfPushedGroups) {
    GroupedIntervalSets groups;
    groups.push(make_set({make_iv(0,1,{2,0},0)}));
    groups.push(make_set({make_iv(0,1,{3,1},1)}));
    EXPECT_EQ(groups.getValue(), (cost_t{5,1}));
}

TEST(GroupedIntervalSets, Value_InSyncAfterClearAndEraseIf) {
    GroupedIntervalSets groups;
    groups.push(make_set({make_iv(0,1,{2,0},0)}));
    groups.push(make_set({make_iv(0,1,{3,1},1)}));
    groups.erase_if([](const IntervalSet& s){ return s.front().id == 0; });
    EXPECT_TRUE(grouped_value_in_sync(groups));
    EXPECT_EQ(groups.getValue(), (cost_t{3,1}));
    groups.clear();
    EXPECT_EQ(groups.getValue(), (cost_t{0,0}));
}

TEST(GroupedIntervalSets, Value_InSyncAfterPushAndPop) {
    GroupedIntervalSets groups;
    groups.push(make_set({make_iv(0,1,{2,0},0)}));
    groups.push(make_set({make_iv(0,1,{3,1},1)}));
    groups.pop();
    EXPECT_TRUE(grouped_value_in_sync(groups));
    EXPECT_EQ(groups.getValue(), (cost_t{2,0}));
}

// begin()/end() only hand out const iterators, so groups cannot be modified
// behind the cached value.
TEST(GroupedIntervalSets, Iterators_AreConst) {
    using It = decltype(std::declval<GroupedIntervalSets&>().begin());
    EXPECT_TRUE(std::is_const_v<std::remove_reference_t<decltype(*std::declval<It>())>>);
}

// Mirrors ris_heuristic::getCoverIntervals: a schedule is built by merging row
// solutions with IntervalSet::push(const IntervalSet&) and then stored as a
// group. Its group-wise value is what softmaxSample weights groups by.
TEST(GroupedIntervalSets, GroupWiseValues_OfMergedSchedules) {
    auto row0 = make_set({make_iv(0,1,{3,0},0), make_iv(3,4,{2,0},1)});
    auto row2 = make_set({make_iv(0,4,{4,1},2)});
    IntervalSet schedule;
    schedule.push(row0);
    schedule.push(row2);

    GroupedIntervalSets groups;
    groups.push(schedule);
    EXPECT_EQ(groups.getGroupWiseValues(), (std::vector<cost_t>{{9,1}}));
}

// ============================================================================
// Interval comparators
// ============================================================================

TEST(Interval, CmpIntervalY_StrictlyLessThanRow) {
    EXPECT_TRUE(Interval::cmpIntervalY(make_iv(0,1,{1,0},0, 2), 3));
    EXPECT_FALSE(Interval::cmpIntervalY(make_iv(0,1,{1,0},0, 3), 3));
    EXPECT_FALSE(Interval::cmpIntervalY(make_iv(0,1,{1,0},0, 4), 3));
}

TEST(Interval, CmpIntervalsPosition_RowThenEndPoint) {
    auto a = make_iv(0,9,{1,0},0, 0);
    auto b = make_iv(0,1,{1,0},1, 1);
    auto c = make_iv(0,5,{1,0},2, 1);
    EXPECT_TRUE(Interval::cmpIntervalsPosition(a, b));   // lower row first
    EXPECT_FALSE(Interval::cmpIntervalsPosition(b, a));
    EXPECT_TRUE(Interval::cmpIntervalsPosition(b, c));   // same row: smaller x2
    EXPECT_FALSE(Interval::cmpIntervalsPosition(c, b));
    EXPECT_FALSE(Interval::cmpIntervalsPosition(b, b));  // strict weak order
}

// ============================================================================
// IntervalSet::push_back(begin, end)
// ============================================================================

TEST(IntervalSet, PushBackRange_AppendsAndUpdatesValue) {
    auto src = make_set({make_iv(0,1,{1,0},0), make_iv(2,3,{2,1},1), make_iv(4,5,{3,0},2)});
    auto dst = make_set({make_iv(9,9,{5,0},9)});
    dst.push_back(std::next(src.begin()), src.end());
    EXPECT_EQ(ids(dst), (std::vector<int>{9, 1, 2}));
    EXPECT_EQ(dst.getValue(), (cost_t{10,1}));
    EXPECT_EQ(dst.getValue(), sum_gains(dst));
}

TEST(IntervalSet, PushBackEmptyRange_NoChange) {
    auto src = make_set({make_iv(0,1,{1,0},0)});
    auto dst = make_set({make_iv(9,9,{5,0},9)});
    dst.push_back(src.end(), src.end());
    EXPECT_EQ(dst.size(), 1u);
    EXPECT_EQ(dst.getValue(), (cost_t{5,0}));
}

// ============================================================================
// IntervalSet::groupByRows
// ============================================================================

namespace {
std::vector<std::vector<int>> row_ids(const GroupedIntervalSets& rows) {
    std::vector<std::vector<int>> out;
    for (auto& row : rows) out.push_back(ids(row));
    return out;
}
} // anonymous namespace

TEST(IntervalSet, GroupByRows_SingleRow_OneGroupSortedByEndPoint) {
    auto set = make_set({make_iv(4,6,{1,0},2), make_iv(0,1,{1,0},0), make_iv(2,3,{1,0},1)});
    auto rows = set.groupByRows();
    EXPECT_EQ(row_ids(rows), (std::vector<std::vector<int>>{{0, 1, 2}}));
}

TEST(IntervalSet, GroupByRows_ConsecutiveRows_OneGroupPerRowInOrder) {
    auto set = make_set({make_iv(0,1,{1,0},2, 2), make_iv(0,1,{1,0},0, 0),
                         make_iv(3,4,{1,0},1, 0), make_iv(0,1,{1,0},3, 1)});
    auto rows = set.groupByRows();
    EXPECT_EQ(row_ids(rows), (std::vector<std::vector<int>>{{0, 1}, {3}, {2}}));
}

// Row index must equal y - min_y: MaximumCover applies the spacing constraint
// to group indices, so a missing row must leave an empty group in its place.
TEST(IntervalSet, GroupByRows_MissingRows_EmptyGroupsKeepRowIndex) {
    auto set = make_set({make_iv(0,1,{1,0},0, 0), make_iv(0,1,{1,0},1, 3)});
    auto rows = set.groupByRows();
    EXPECT_EQ(row_ids(rows), (std::vector<std::vector<int>>{{0}, {}, {}, {1}}));
}

TEST(IntervalSet, GroupByRows_NonZeroMinRow_IndexedFromMinRow) {
    auto set = make_set({make_iv(0,1,{1,0},0, 5), make_iv(0,1,{1,0},1, 6)});
    auto rows = set.groupByRows();
    EXPECT_EQ(row_ids(rows), (std::vector<std::vector<int>>{{0}, {1}}));
}

// Each row's value must be the sum of its gains, and the total must match the
// source set (MaximumCover is fed the group-wise values).
TEST(IntervalSet, GroupByRows_RowValuesAndTotal) {
    auto set = make_set({make_iv(0,1,{2,1},0, 0), make_iv(3,4,{3,0},1, 0),
                         make_iv(0,1,{4,2},2, 1)});
    auto rows = set.groupByRows();
    EXPECT_EQ(rows.getGroupWiseValues(), (std::vector<cost_t>{{5,1},{4,2}}));
    EXPECT_EQ(rows.getValue(), set.getValue());
}

// An empty set has no rows. Run in a child process: groupByRows reads
// front()/back() of the interval vector.
TEST(IntervalSet, GroupByRows_EmptySet_NoRows) {
    EXPECT_EXIT({
        IntervalSet set;
        auto rows = set.groupByRows();
        std::exit(rows.empty() ? 0 : 1);
    }, ::testing::ExitedWithCode(0), "");
}

// ============================================================================
// GroupedIntervalSets — bulk insertion and per-set row grouping
// ============================================================================

TEST(GroupedIntervalSets, PushGroups_AppendsAndSumsValue) {
    GroupedIntervalSets a, b;
    a.push(make_set({make_iv(0,1,{2,0},0)}));
    b.push(make_set({make_iv(0,1,{3,1},1)}));
    b.push(make_set({make_iv(0,1,{1,0},2)}));
    a.push(b);
    ASSERT_EQ(a.size(), 3u);
    EXPECT_EQ(a[2].front().id, 2);
    EXPECT_EQ(a.getValue(), (cost_t{6,1}));
    EXPECT_TRUE(grouped_value_in_sync(a));
}

TEST(GroupedIntervalSets, PushBackRange_NewGroupWithValue) {
    auto src = make_set({make_iv(0,1,{2,0},0), make_iv(3,4,{3,1},1)});
    GroupedIntervalSets groups;
    groups.push_back(src.begin(), src.end());
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(ids(groups[0]), (std::vector<int>{0, 1}));
    EXPECT_EQ(groups.getGroupWiseValues(), (std::vector<cost_t>{{5,1}}));
    EXPECT_TRUE(grouped_value_in_sync(groups));
}

TEST(GroupedIntervalSets, GroupSetsByRows_OneEntryPerSet) {
    GroupedIntervalSets groups;
    groups.push(make_set({make_iv(0,1,{1,0},0, 0), make_iv(0,1,{1,0},1, 1)}));
    groups.push(make_set({make_iv(0,1,{1,0},2, 4)}));
    auto per_set = groups.groupSetsByRows();
    ASSERT_EQ(per_set.size(), 2u);
    EXPECT_EQ(row_ids(per_set[0]), (std::vector<std::vector<int>>{{0}, {1}}));
    EXPECT_EQ(row_ids(per_set[1]), (std::vector<std::vector<int>>{{2}}));
}
