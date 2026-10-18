#include <gtest/gtest.h>
#include "../../include/heuristics/luby.hpp"

// ============================================================================
// luby unit tests
//
// What is being tested:
//   luby::operator()(i) returns the i-th term (1-based) of the Luby restart
//   sequence 1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,... multiplied by scale = 4.
//   ILS calls it with iterations = 1, 2, 3, ...
//
// Test strategy: compare the first terms with the reference sequence, check
// the closed form at the end of each block (i = 2^k - 1) and the recursive
// structure of the sequence.
// ============================================================================

namespace {

// Reference Luby term (unscaled), straightforward recursive definition.
int luby_reference(int i) {
    int k = 1;
    while ((1 << k) - 1 < i) ++k;
    if (i == (1 << k) - 1) return 1 << (k - 1);
    return luby_reference(i - (1 << (k - 1)) + 1);
}

} // anonymous namespace

TEST(Luby, FirstTerms_MatchScaledSequence) {
    luby L;
    const std::vector<int> expected = {1,1,2,1,1,2,4,1,1,2,1,1,2,4,8,1};
    for (int i = 1; i <= static_cast<int>(expected.size()); ++i)
        EXPECT_EQ(L(i), 4 * expected[i - 1]) << "i=" << i;
}

// First call made by ILS (iterations starts at 1).
TEST(Luby, FirstIndex_ReturnsScale) {
    luby L;
    EXPECT_EQ(L(1), 4);
}

// End of block k: i = 2^k - 1 gives scale * 2^(k-1).
TEST(Luby, EndOfBlock_ClosedForm) {
    luby L;
    for (int k = 1; k <= 27; ++k)
        EXPECT_EQ(L((1 << k) - 1), 4 * (1 << (k - 1))) << "k=" << k;
}

// Start of block k + 1: i = 2^k restarts the sequence at 1.
TEST(Luby, StartOfBlock_RestartsAtScale) {
    luby L;
    for (int k = 1; k <= 27; ++k)
        EXPECT_EQ(L(1 << k), 4) << "i=2^" << k;
}

// Matches the reference definition over a long prefix.
TEST(Luby, LongPrefix_MatchesReference) {
    luby L;
    for (int i = 1; i <= 5000; ++i)
        ASSERT_EQ(L(i), 4 * luby_reference(i)) << "i=" << i;
}

// Values are always scale times a power of two.
TEST(Luby, Values_AreScaledPowersOfTwo) {
    luby L;
    for (int i = 1; i <= 2000; ++i) {
        int v = L(i);
        ASSERT_EQ(v % 4, 0) << "i=" << i;
        int p = v / 4;
        ASSERT_TRUE(p > 0 && (p & (p - 1)) == 0) << "i=" << i << " value=" << v;
    }
}

// Stateless: calling twice with the same index gives the same value, and the
// order of calls has no influence.
TEST(Luby, Stateless_SameInputSameOutput) {
    luby L;
    int a = L(13);
    L(1000);
    EXPECT_EQ(L(13), a);
}
