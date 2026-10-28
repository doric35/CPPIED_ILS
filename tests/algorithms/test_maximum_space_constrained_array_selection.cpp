#include <gtest/gtest.h>
#include <random>
#include "../../include/algorithms/MaximumSpaceConstrainedArraySelection.hpp"

// ============================================================================
// Tests of MaximumSpaceConstrainedArraySelection: select indexes of costs of
// maximum total cost such that any two selected indexes i < j satisfy
// j - i >= spacingConstraint. The empty selection is feasible (value 0).
// The result is checked against a brute force over every subset.
// ============================================================================

namespace {

double selectionValue(const std::vector<double>& costs, const std::vector<int>& selection) {
    double sum = 0.0;
    for (int i : selection) sum += costs[i];
    return sum;
}

double bruteForceMaximum(int spacing, const std::vector<double>& costs) {
    const int n = static_cast<int>(costs.size());
    double best = 0.0;
    for (int mask = 0; mask < (1 << n); ++mask) {
        double sum = 0.0;
        int last = -1;
        bool feasible = true;
        for (int i = 0; i < n && feasible; ++i) {
            if (!(mask & (1 << i))) continue;
            if (last >= 0 && i - last < spacing) feasible = false;
            sum += costs[i];
            last = i;
        }
        if (feasible) best = std::max(best, sum);
    }
    return best;
}

void expectFeasible(int spacing, const std::vector<double>& costs, const std::vector<int>& selection) {
    for (size_t k = 0; k < selection.size(); ++k) {
        ASSERT_GE(selection[k], 0);
        ASSERT_LT(selection[k], static_cast<int>(costs.size()));
        if (k > 0)
            EXPECT_GE(selection[k] - selection[k-1], spacing)
                << "indexes " << selection[k-1] << " and " << selection[k];
    }
}

void expectOptimal(int spacing, const std::vector<double>& costs, const std::vector<int>& selection) {
    expectFeasible(spacing, costs, selection);
    EXPECT_DOUBLE_EQ(selectionValue(costs, selection), bruteForceMaximum(spacing, costs));
}

std::string describe(int spacing, const std::vector<double>& costs, const std::vector<int>& selection) {
    std::ostringstream os;
    os << "spacing " << spacing << " costs [";
    for (double c : costs) os << c << " ";
    os << "] selection [";
    for (int i : selection) os << i << " ";
    os << "]";
    return os.str();
}

// Integer valued costs so that sums are exact.
std::vector<double> randomCosts(std::mt19937& rng, int n, int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    std::vector<double> costs(n);
    for (auto& c : costs) c = dist(rng);
    return costs;
}

}  // namespace

TEST(MaximumSpaceConstrainedArraySelection, SingleElementPositive) {
    std::vector<double> costs{5.0};
    MaximumSpaceConstrainedArraySelection algorithm(1, costs);
    EXPECT_EQ(algorithm.solve(), (std::vector<int>{0}));
}

TEST(MaximumSpaceConstrainedArraySelection, SingleElementNegative) {
    std::vector<double> costs{-5.0};
    MaximumSpaceConstrainedArraySelection algorithm(1, costs);
    EXPECT_TRUE(algorithm.solve().empty());
}

TEST(MaximumSpaceConstrainedArraySelection, UnitSpacingSelectsAllPositive) {
    std::vector<double> costs{1.0, -2.0, 3.0, 0.5, -1.0, 4.0};
    MaximumSpaceConstrainedArraySelection algorithm(1, costs);
    EXPECT_EQ(algorithm.solve(), (std::vector<int>{0, 2, 3, 5}));
}

TEST(MaximumSpaceConstrainedArraySelection, AllNegativeSelectsNothing) {
    std::vector<double> costs{-1.0, -2.0, -3.0, -4.0, -5.0};
    MaximumSpaceConstrainedArraySelection algorithm(2, costs);
    EXPECT_TRUE(algorithm.solve().empty());
}

TEST(MaximumSpaceConstrainedArraySelection, AdjacentPairKeepsBest) {
    std::vector<double> costs{3.0, 5.0};
    MaximumSpaceConstrainedArraySelection algorithm(2, costs);
    EXPECT_EQ(algorithm.solve(), (std::vector<int>{1}));
}

TEST(MaximumSpaceConstrainedArraySelection, AlternatingSelection) {
    std::vector<double> costs{3.0, 1.0, 4.0};
    MaximumSpaceConstrainedArraySelection algorithm(2, costs);
    EXPECT_EQ(algorithm.solve(), (std::vector<int>{0, 2}));
}

// Every value increases on [2, 4] but 2, 3, 4 are not pairwise compatible.
TEST(MaximumSpaceConstrainedArraySelection, IncreasingValuesDoNotImplySelection) {
    std::vector<double> costs{0.0, 0.0, 2.0, 3.0, 2.0};
    MaximumSpaceConstrainedArraySelection algorithm(2, costs);
    const auto& selection = algorithm.solve();
    SCOPED_TRACE(describe(2, costs, selection));
    expectOptimal(2, costs, selection);
    EXPECT_EQ(selection, (std::vector<int>{2, 4}));
}

// With spacing equal to the array size at most one index can be selected.
TEST(MaximumSpaceConstrainedArraySelection, SpacingEqualToSizeSelectsMaximum) {
    std::vector<double> costs{1.0, 2.0, 7.0, 3.0};
    MaximumSpaceConstrainedArraySelection algorithm(4, costs);
    EXPECT_EQ(algorithm.solve(), (std::vector<int>{2}));
}

// The best isolated element sits after the last strictly increasing value.
TEST(MaximumSpaceConstrainedArraySelection, RootAfterLastIncrease) {
    std::vector<double> costs{1.0, 0.0, 0.0, 6.0};
    MaximumSpaceConstrainedArraySelection algorithm(3, costs);
    const auto& selection = algorithm.solve();
    SCOPED_TRACE(describe(3, costs, selection));
    expectOptimal(3, costs, selection);
}

TEST(MaximumSpaceConstrainedArraySelection, SelectionIsSortedAndFeasible) {
    std::vector<double> costs{2.0, 9.0, 1.0, 1.0, 8.0, 1.0, 1.0, 7.0};
    MaximumSpaceConstrainedArraySelection algorithm(3, costs);
    const auto& selection = algorithm.solve();
    SCOPED_TRACE(describe(3, costs, selection));
    EXPECT_TRUE(std::ranges::is_sorted(selection));
    expectOptimal(3, costs, selection);
    EXPECT_EQ(selection, (std::vector<int>{1, 4, 7}));
}

TEST(MaximumSpaceConstrainedArraySelection, MatchesBruteForceOnRandomArrays) {
    std::mt19937 rng(2024);
    for (int spacing = 1; spacing <= 4; ++spacing) {
        for (int trial = 0; trial < 300; ++trial) {
            int n = spacing + trial % (13 - spacing);
            auto costs = randomCosts(rng, n, -5, 9);
            MaximumSpaceConstrainedArraySelection algorithm(spacing, costs);
            const auto& selection = algorithm.solve();
            SCOPED_TRACE(describe(spacing, costs, selection));
            expectOptimal(spacing, costs, selection);
        }
    }
}

// DpSweeper feeds gains where satisfied candidates carry a very negative sentinel.
TEST(MaximumSpaceConstrainedArraySelection, MatchesBruteForceWithNegativeSentinels) {
    std::mt19937 rng(77);
    std::bernoulli_distribution sentinel(0.3);
    for (int trial = 0; trial < 300; ++trial) {
        int spacing = 2 + trial % 3;
        int n = spacing + trial % 9;
        auto costs = randomCosts(rng, n, 1, 9);
        for (auto& c : costs) if (sentinel(rng)) c = -1e9;
        MaximumSpaceConstrainedArraySelection algorithm(spacing, costs);
        const auto& selection = algorithm.solve();
        SCOPED_TRACE(describe(spacing, costs, selection));
        expectOptimal(spacing, costs, selection);
    }
}

// Calling solve twice on the same instance must return the same selection.
TEST(MaximumSpaceConstrainedArraySelection, SolveIsIdempotent) {
    std::vector<double> costs{3.0, 1.0, 4.0};
    MaximumSpaceConstrainedArraySelection algorithm(2, costs);
    std::vector<int> first = algorithm.solve();
    std::vector<int> second = algorithm.solve();
    EXPECT_EQ(first, second);
}

// Arrays shorter than the spacing (and empty arrays) occur for DpSweeper
// candidate sets near the grid border. Run in a subprocess: the check exits
// with 0 on the expected result, so a crash is reported as a failure.
TEST(MaximumSpaceConstrainedArraySelectionDeathTest, ArrayShorterThanSpacing) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    auto check = [] {
        std::vector<double> costs{1.0, 4.0, 2.0};
        MaximumSpaceConstrainedArraySelection algorithm(5, costs);
        return algorithm.solve() == std::vector<int>{1};
    };
    EXPECT_EXIT(std::exit(check() ? 0 : 1), ::testing::ExitedWithCode(0), "");
}

TEST(MaximumSpaceConstrainedArraySelectionDeathTest, EmptyArray) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    auto check = [] {
        std::vector<double> costs;
        MaximumSpaceConstrainedArraySelection algorithm(2, costs);
        return algorithm.solve().empty();
    };
    EXPECT_EXIT(std::exit(check() ? 0 : 1), ::testing::ExitedWithCode(0), "");
}
