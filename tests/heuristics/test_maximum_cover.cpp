#include <gtest/gtest.h>
#include <random>
#include <set>
#include "../../include/heuristics/MaximumCover.hpp"

// ============================================================================
// MaximumCover unit tests
//
// What is being tested:
//   MaximumCover solves a 1D "spaced selection" problem: given costs c[0..n-1]
//   and a spacing m >= 1, select indices S such that any two selected indices
//   differ by at least m, maximising sum_{i in S} c[i] (cost_t is compared
//   lexicographically: length first, then turns).
//
//   Public API: constructor, solve(), getSolution(), getCost().
//
// Test strategy:
//   - Hand-built instances for the base cases (n < m, n == m, m == 1, ...).
//   - Exhaustive brute force on many small random instances with heavy ties,
//     checking optimality of getCost() and feasibility/consistency of
//     getSolution().
// ============================================================================

namespace {

cost_t sum_costs(const std::vector<cost_t>& costs, const std::vector<int>& sel) {
    cost_t total{0, 0};
    for (int i : sel) total += costs[i];
    return total;
}

// True if every index is in range, no duplicates, and pairwise gaps >= spacing.
::testing::AssertionResult is_feasible(const std::vector<cost_t>& costs,
                                       const std::vector<int>& sel,
                                       int spacing) {
    std::vector<int> sorted = sel;
    std::sort(sorted.begin(), sorted.end());
    for (std::size_t k = 0; k < sorted.size(); ++k) {
        if (sorted[k] < 0 || sorted[k] >= static_cast<int>(costs.size()))
            return ::testing::AssertionFailure() << "index " << sorted[k] << " out of range";
        if (k > 0 && sorted[k] - sorted[k - 1] < spacing)
            return ::testing::AssertionFailure()
                    << "indices " << sorted[k - 1] << " and " << sorted[k]
                    << " are closer than spacing=" << spacing;
    }
    return ::testing::AssertionSuccess();
}

// Exhaustive optimum over all feasible subsets (n <= ~16).
cost_t brute_force(const std::vector<cost_t>& costs, int spacing) {
    const int n = static_cast<int>(costs.size());
    cost_t best{0, 0};
    for (unsigned mask = 0; mask < (1u << n); ++mask) {
        int last = -spacing;
        bool ok = true;
        cost_t total{0, 0};
        for (int i = 0; i < n && ok; ++i) {
            if (!(mask & (1u << i))) continue;
            if (i - last < spacing) ok = false;
            last = i;
            total += costs[i];
        }
        if (ok) best = std::max(best, total);
    }
    return best;
}

std::string to_string(const std::vector<cost_t>& costs) {
    std::ostringstream os;
    for (auto& c : costs) os << c << " ";
    return os.str();
}

} // anonymous namespace

// ============================================================================
// Getters before solve()
// ============================================================================

TEST(MaximumCover, BeforeSolve_SolutionEmptyAndCostZero) {
    std::vector<cost_t> costs = {{3,0},{4,0},{5,0}};
    MaximumCover mc(costs, 2);
    EXPECT_TRUE(mc.getSolution().empty());
    EXPECT_EQ(mc.getCost(), (cost_t{0,0}));
}

// ============================================================================
// Base cases
// ============================================================================

TEST(MaximumCover, SingleElement_Selected) {
    std::vector<cost_t> costs = {{7,2}};
    MaximumCover mc(costs, 1);
    mc.solve();
    EXPECT_EQ(mc.getSolution(), (std::vector<int>{0}));
    EXPECT_EQ(mc.getCost(), (cost_t{7,2}));
}

// spacing == 1: no conflict at all, every element (non-negative) can be taken.
TEST(MaximumCover, SpacingOne_AllPositiveSelected) {
    std::vector<cost_t> costs = {{1,0},{2,0},{3,0},{4,0}};
    MaximumCover mc(costs, 1);
    mc.solve();
    auto sol = mc.getSolution();
    EXPECT_EQ(std::set<int>(sol.begin(), sol.end()), (std::set<int>{0,1,2,3}));
    EXPECT_EQ(mc.getCost(), (cost_t{10,0}));
}

// n == spacing: only one element can be picked, it must be the best one.
TEST(MaximumCover, SizeEqualsSpacing_BestSingleElement) {
    std::vector<cost_t> costs = {{2,0},{9,0},{4,0}};
    MaximumCover mc(costs, 3);
    mc.solve();
    EXPECT_EQ(mc.getSolution(), (std::vector<int>{1}));
    EXPECT_EQ(mc.getCost(), (cost_t{9,0}));
}

// n < spacing: forward pass must not run past the end of the cost vector.
TEST(MaximumCover, SizeSmallerThanSpacing_BestSingleElement) {
    std::vector<cost_t> costs = {{2,0},{9,0}};
    MaximumCover mc(costs, 5);
    mc.solve();
    EXPECT_EQ(mc.getSolution(), (std::vector<int>{1}));
    EXPECT_EQ(mc.getCost(), (cost_t{9,0}));
}

// The maximum among the first `spacing` elements is NOT the last of them.
// The optimal value at index spacing-1 is the prefix maximum, not costs[m-1].
TEST(MaximumCover, BaseWindowMaxNotLast_CostIsPrefixMax) {
    std::vector<cost_t> costs = {{5,0},{1,0}};
    MaximumCover mc(costs, 2);
    mc.solve();
    EXPECT_EQ(mc.getCost(), (cost_t{5,0}));
    EXPECT_EQ(mc.getSolution(), (std::vector<int>{0}));
}

// Same issue, but propagated past the base window: optimum is {0, 4}.
TEST(MaximumCover, BaseWindowMaxNotLast_PropagatesToOptimum) {
    std::vector<cost_t> costs = {{5,0},{1,0},{0,0},{0,0},{4,0}};
    MaximumCover mc(costs, 3);
    mc.solve();
    EXPECT_EQ(mc.getCost(), (cost_t{9,0}));
    auto sol = mc.getSolution();
    EXPECT_EQ(std::set<int>(sol.begin(), sol.end()), (std::set<int>{0,4}));
}

// Root of the backward pass must be allowed to be the last index of the base
// window (index spacing-1).
TEST(MaximumCover, RootCanBeLastIndexOfBaseWindow) {
    std::vector<cost_t> costs = {{1,0},{6,0},{0,0},{3,0}};
    MaximumCover mc(costs, 2);
    mc.solve();
    EXPECT_EQ(mc.getCost(), (cost_t{9,0}));
    auto sol = mc.getSolution();
    EXPECT_EQ(std::set<int>(sol.begin(), sol.end()), (std::set<int>{1,3}));
}

// ============================================================================
// Structural cases
// ============================================================================

// Every other element, spacing 2: picks even indices when they dominate.
TEST(MaximumCover, AlternatingCosts_PicksHighParity) {
    std::vector<cost_t> costs = {{5,0},{1,0},{5,0},{1,0},{5,0}};
    MaximumCover mc(costs, 2);
    mc.solve();
    auto sol = mc.getSolution();
    EXPECT_EQ(std::set<int>(sol.begin(), sol.end()), (std::set<int>{0,2,4}));
    EXPECT_EQ(mc.getCost(), (cost_t{15,0}));
}

// A single large element beats two smaller compatible ones.
TEST(MaximumCover, SingleLargeBeatsTwoSmall) {
    std::vector<cost_t> costs = {{3,0},{10,0},{3,0}};
    MaximumCover mc(costs, 2);
    mc.solve();
    EXPECT_EQ(mc.getSolution(), (std::vector<int>{1}));
    EXPECT_EQ(mc.getCost(), (cost_t{10,0}));
}

// cost_t is lexicographic: equal length, turns break the tie.
TEST(MaximumCover, LexicographicCost_TurnsBreakLengthTie) {
    std::vector<cost_t> costs = {{4,1},{4,5},{4,2}};
    MaximumCover mc(costs, 3);
    mc.solve();
    EXPECT_EQ(mc.getSolution(), (std::vector<int>{1}));
    EXPECT_EQ(mc.getCost(), (cost_t{4,5}));
}

// Length dominates turns: {5,0} beats {4,100}.
TEST(MaximumCover, LexicographicCost_LengthDominatesTurns) {
    std::vector<cost_t> costs = {{4,100},{5,0}};
    MaximumCover mc(costs, 2);
    mc.solve();
    EXPECT_EQ(mc.getSolution(), (std::vector<int>{1}));
}

// All zero costs: cost is zero and whatever is returned stays feasible.
TEST(MaximumCover, AllZeroCosts_CostZeroAndFeasible) {
    std::vector<cost_t> costs(6, cost_t{0,0});
    MaximumCover mc(costs, 2);
    mc.solve();
    EXPECT_EQ(mc.getCost(), (cost_t{0,0}));
    EXPECT_TRUE(is_feasible(costs, mc.getSolution(), 2));
}

// All equal costs with many optimal solutions: result must still be optimal.
TEST(MaximumCover, AllEqualCosts_OptimalCardinality) {
    std::vector<cost_t> costs(7, cost_t{1,0});
    MaximumCover mc(costs, 3);
    mc.solve();
    auto sol = mc.getSolution();
    EXPECT_TRUE(is_feasible(costs, sol, 3));
    EXPECT_EQ(sol.size(), 3u);  // e.g. {0,3,6}
    EXPECT_EQ(mc.getCost(), (cost_t{3,0}));
}

// Calling solve() twice must not duplicate the solution.
TEST(MaximumCover, SolveTwice_SolutionNotDuplicated) {
    std::vector<cost_t> costs = {{5,0},{1,0},{5,0}};
    MaximumCover mc(costs, 2);
    mc.solve();
    auto first = mc.getSolution();
    mc.solve();
    auto second = mc.getSolution();
    EXPECT_EQ(first.size(), second.size());
    EXPECT_TRUE(is_feasible(costs, second, 2));
}

// Same input, two fresh solvers: identical result (default-seeded RNG).
TEST(MaximumCover, Deterministic_SameInputSameSolution) {
    std::vector<cost_t> costs = {{1,0},{1,0},{1,0},{1,0},{1,0},{1,0}};
    MaximumCover a(costs, 2), b(costs, 2);
    a.solve();
    b.solve();
    EXPECT_EQ(a.getSolution(), b.getSolution());
}

// ============================================================================
// Randomised comparison against brute force
// ============================================================================

// For many small random instances (lots of ties), getCost() must equal the
// brute-force optimum, getSolution() must be feasible, and the selected costs
// must add up to getCost().
TEST(MaximumCover, Random_MatchesBruteForce) {
    std::mt19937 gen(2026);
    std::uniform_int_distribution<int> len(0, 4), turns(0, 2);
    for (int trial = 0; trial < 2000; ++trial) {
        int n = 1 + trial % 12;
        int spacing = 1 + (trial / 12) % 5;
        std::vector<cost_t> costs(n);
        for (auto& c : costs) c = {len(gen), turns(gen)};

        MaximumCover mc(costs, spacing);
        mc.solve();
        auto sol = mc.getSolution();
        cost_t expected = brute_force(costs, spacing);

        SCOPED_TRACE("n=" + std::to_string(n) + " spacing=" + std::to_string(spacing)
                     + " costs=" + to_string(costs));
        ASSERT_EQ(mc.getCost(), expected);
        ASSERT_TRUE(is_feasible(costs, sol, spacing));
        ASSERT_EQ(sum_costs(costs, sol), expected);
    }
}
