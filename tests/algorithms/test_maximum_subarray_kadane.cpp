#include <gtest/gtest.h>
#include <random>
#include "../../include/algorithms/MaximumSubarrayKadane.hpp"

// ============================================================================
// Tests of MaximumSubarrayKadane: the maximum sum contiguous subarray
// costs[start..end] (inclusive bounds, non-empty).
// The result is checked against a brute force over every (start, end) pair.
// ============================================================================

namespace {

double rangeSum(const std::vector<double>& costs, int start, int end) {
    double sum = 0.0;
    for (int i = start; i <= end; ++i) sum += costs[i];
    return sum;
}

double bruteForceMaximum(const std::vector<double>& costs) {
    double best = -std::numeric_limits<double>::infinity();
    for (int s = 0; s < static_cast<int>(costs.size()); ++s)
        for (int e = s; e < static_cast<int>(costs.size()); ++e)
            best = std::max(best, rangeSum(costs, s, e));
    return best;
}

void expectValidOptimum(const std::vector<double>& costs, const SubArray& result) {
    ASSERT_LE(0, result.start);
    ASSERT_LE(result.start, result.end);
    ASSERT_LT(result.end, static_cast<int>(costs.size()));
    EXPECT_DOUBLE_EQ(result.value, rangeSum(costs, result.start, result.end));
    EXPECT_DOUBLE_EQ(result.value, bruteForceMaximum(costs));
}

// Integer valued costs so that sums are exact.
std::vector<double> randomCosts(std::mt19937& rng, int n, int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    std::vector<double> costs(n);
    for (auto& c : costs) c = dist(rng);
    return costs;
}

}  // namespace

TEST(MaximumSubarrayKadane, SingleElement) {
    std::vector<double> costs{3.5};
    SubArray r = MaximumSubarrayKadane(costs).solve();
    EXPECT_EQ(r.start, 0);
    EXPECT_EQ(r.end, 0);
    EXPECT_DOUBLE_EQ(r.value, 3.5);
}

TEST(MaximumSubarrayKadane, SingleNegativeElement) {
    std::vector<double> costs{-2.0};
    SubArray r = MaximumSubarrayKadane(costs).solve();
    EXPECT_EQ(r.start, 0);
    EXPECT_EQ(r.end, 0);
    EXPECT_DOUBLE_EQ(r.value, -2.0);
}

TEST(MaximumSubarrayKadane, AllPositiveTakesWholeArray) {
    std::vector<double> costs{1.0, 2.0, 3.0, 4.0};
    SubArray r = MaximumSubarrayKadane(costs).solve();
    EXPECT_EQ(r.start, 0);
    EXPECT_EQ(r.end, 3);
    EXPECT_DOUBLE_EQ(r.value, 10.0);
}

TEST(MaximumSubarrayKadane, AllNegativeTakesMaximumElement) {
    std::vector<double> costs{-5.0, -1.0, -3.0, -4.0};
    SubArray r = MaximumSubarrayKadane(costs).solve();
    EXPECT_EQ(r.start, 1);
    EXPECT_EQ(r.end, 1);
    EXPECT_DOUBLE_EQ(r.value, -1.0);
}

TEST(MaximumSubarrayKadane, ClassicExample) {
    std::vector<double> costs{-2.0, 1.0, -3.0, 4.0, -1.0, 2.0, 1.0, -5.0, 4.0};
    SubArray r = MaximumSubarrayKadane(costs).solve();
    EXPECT_EQ(r.start, 3);
    EXPECT_EQ(r.end, 6);
    EXPECT_DOUBLE_EQ(r.value, 6.0);
}

TEST(MaximumSubarrayKadane, OptimumAtLastElement) {
    std::vector<double> costs{1.0, -10.0, 2.0, -10.0, 7.0};
    SubArray r = MaximumSubarrayKadane(costs).solve();
    EXPECT_EQ(r.start, 4);
    EXPECT_EQ(r.end, 4);
    EXPECT_DOUBLE_EQ(r.value, 7.0);
}

TEST(MaximumSubarrayKadane, OptimumAtFirstElement) {
    std::vector<double> costs{7.0, -10.0, 2.0, -10.0, 1.0};
    SubArray r = MaximumSubarrayKadane(costs).solve();
    EXPECT_EQ(r.start, 0);
    EXPECT_EQ(r.end, 0);
    EXPECT_DOUBLE_EQ(r.value, 7.0);
}

// A negative prefix must be dropped, a small negative inside a profitable run kept.
TEST(MaximumSubarrayKadane, KeepsInnerNegativeDropsNegativePrefix) {
    std::vector<double> costs{-3.0, 4.0, -1.0, 4.0, -9.0};
    SubArray r = MaximumSubarrayKadane(costs).solve();
    EXPECT_EQ(r.start, 1);
    EXPECT_EQ(r.end, 3);
    EXPECT_DOUBLE_EQ(r.value, 7.0);
}

// DpSweeper marks satisfied vertices with a very negative sentinel.
TEST(MaximumSubarrayKadane, LargeNegativeSentinelsSplitRuns) {
    const double sentinel = -1e9;
    std::vector<double> costs{2.0, sentinel, 1.0, 1.0, 1.0, sentinel, 2.5};
    SubArray r = MaximumSubarrayKadane(costs).solve();
    EXPECT_EQ(r.start, 2);
    EXPECT_EQ(r.end, 4);
    EXPECT_DOUBLE_EQ(r.value, 3.0);
}

TEST(MaximumSubarrayKadane, MatchesBruteForceOnRandomMixedArrays) {
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 500; ++trial) {
        int n = 1 + trial % 15;
        auto costs = randomCosts(rng, n, -6, 6);
        SubArray r = MaximumSubarrayKadane(costs).solve();
        SCOPED_TRACE("trial " + std::to_string(trial));
        expectValidOptimum(costs, r);
    }
}

TEST(MaximumSubarrayKadane, MatchesBruteForceOnRandomNegativeArrays) {
    std::mt19937 rng(54321);
    for (int trial = 0; trial < 200; ++trial) {
        int n = 1 + trial % 10;
        auto costs = randomCosts(rng, n, -9, -1);
        SubArray r = MaximumSubarrayKadane(costs).solve();
        SCOPED_TRACE("trial " + std::to_string(trial));
        expectValidOptimum(costs, r);
        EXPECT_EQ(r.start, r.end);
    }
}

// Calling solve twice on the same instance must return the same subarray.
TEST(MaximumSubarrayKadane, SolveIsIdempotent) {
    std::vector<double> costs{-2.0, 1.0, -3.0, 4.0, -1.0, 2.0, 1.0, -5.0, 4.0};
    MaximumSubarrayKadane kadane(costs);
    SubArray first = kadane.solve();
    SubArray second = kadane.solve();
    EXPECT_EQ(first.start, second.start);
    EXPECT_EQ(first.end, second.end);
    EXPECT_DOUBLE_EQ(first.value, second.value);
}

TEST(SubArray, OrderedByValueOnly) {
    SubArray a{0, 5, 1.0}, b{3, 3, 2.0}, c{1, 1, 1.0};
    EXPECT_TRUE(a < b);
    EXPECT_FALSE(b < a);
    EXPECT_FALSE(a < c);
    EXPECT_FALSE(c < a);
}
