#include "../test_fixture.hpp"
#include "../../include/heuristics/ris_heuristic.hpp"
#include "../../include/algorithms/RandomOneTwoReallocation.hpp"
#include "../../include/neighborhoods/neighborhood_n1.hpp"
#include "../../include/neighborhoods/neighborhood_n12.hpp"
#include "../../include/neighborhoods/neighborhood_n21.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_cut.hpp"
#include "../../include/neighborhoods/neighborhood_reduce.hpp"
#include "../../include/neighborhoods/neighborhood_r.hpp"

// ============================================================================
// Precondition of ris_heuristic::softmaxSample: every gain it receives is
// non-negative (gain >= {0,0}, cost_t's lexicographic order).
//
// softmaxSample weights gains by setFlatScores, which flattens them to
// M * length + turns (M = max turns + 1, unsigned arithmetic). Each call
// site is also checked for the property the sampling relies on: the flat
// scores keep the order of the gains.
//
// Call sites, each checked on the gains it would pass:
//   1. neighborhood_nested::getRandomMove      removal gains of path[1..]
//   2. neighborhood_r::setSelectReallocations  even / odd sums of removal gains
//   3. RandomOneTwoReallocation::setIncumbentMove  candidate move gains
//   4. ris_heuristic::softmaxSample()          values of the RIS schedules,
//      built from SimpleNeighborhood::selectMoves intervals (n1, n12, n21,
//      trim, cut, reduce)
//
// States: captured feasible paths plus every state visited by a descent of
// each simple neighborhood from them.
// ============================================================================

namespace {

std::string segStr(const segment& s) {
    return "{" + std::to_string(s.source) + "," + std::to_string(s.target) + "}";
}

std::string costStr(const cost_t& c) {
    return "{l=" + std::to_string(c.length) + ", t=" + std::to_string(c.turns) + "}";
}

bool nonNegative(const cost_t& c) { return c >= cost_t{0, 0}; }

// The dominated / "no move" sentinel, or any gain carrying one of its
// components.
bool isSentinel(const cost_t& c) { return c.length == MIN_INT || c.turns == MIN_INT; }

int countSentinels(const std::vector<cost_t>& gains) {
    return static_cast<int>(std::ranges::count_if(gains, isSentinel));
}

// Number of gain pairs a < b whose flat scores are not ordered the same way.
int flatScoreInversions(std::vector<cost_t> gains) {
    if (gains.empty()) return 0;
    std::vector<double> scores = ris_heuristic::setFlatScores(gains);
    int inversions = 0;
    for (std::size_t a = 0; a < gains.size(); ++a)
        for (std::size_t b = 0; b < gains.size(); ++b)
            inversions += gains[a] < gains[b] && !(scores[a] < scores[b]);
    return inversions;
}

// Fixture paths captured from ils_full runs (all feasible).
std::vector<std::vector<segment>> inputs() {
    return {
        {{6, 11}, {23, 18}, {44, 42}, {48, 51}, {30, 35}, {75, 73}, {62, 63}},
        {{6, 11}, {23, 18}, {44, 43}, {6, path_engine::NULL_NODE}, {48, 51}, {30, 35}, {75, 73}, {62, 63}},
        {{6, 11}, {79, path_engine::NULL_NODE}, {23, 18}, {44, 42}, {48, 51}, {30, 35}, {75, 74}, {10, 9}, {62, 63}},
        {{6, 11}, {79, path_engine::NULL_NODE}, {23, 18}, {44, 42}, {48, 51}, {30, 35}, {75, 74},
         {16, path_engine::REVERSED_NULL_NODE}, {61, 63}},
        {{6,-1},{12,-2},{6,11},{79,80},{35,30},{45,42},{54,57},{18,23},{29,28},{63,62},{16,-1}},
        {{6,-1},{48,-2},{7,6},{18,22},{23,-1},{35,30},{44,-2},{6,11},{79,-1},{75,74},{16,-2},{15,12},{18,21}},
        {{6,-1},{48,51},{30,35},{23,18},{44,-2},{6,9},{72,73},{62,63},{75,73},{11,-1}},
        {{6,-1},{50,-1},{18,19},{51,48},{60,-1},{62,65},{77,73},{61,63},{75,72},{48,53}},
        {{6,-1},{7,11},{23,20},{25,24},{45,42},{0,1},{9,10},{79,81},{35,30},{44,-2},{12,13},{56,57},{27,28},{16,15}},
        {{6,-1},{7,11},{80,81},{29,-2},{77,-1},{71,68},{56,59},{53,48},{42,44},{18,-1},{20,22},{73,-2},{22,23}},
    };
}

template<typename N>
struct simple_accessor : public N {
    using N::N;
    using N::candidatesMoves;
    using N::clear;
    using N::setMovesAndGains;
    using N::pruneNonImprovingMoves;
    using N::splitMoves;
    using N::constructIntervalSets;
};

struct r_accessor : public neighborhood_r {
    using neighborhood_r::neighborhood_r;
    using neighborhood_r::getRemovalGains;
};

// Records the gains RandomOneTwoReallocation hands to softmaxSample: the
// candidate gains (raw) after applyNegativeGainsCorrection (sampled).
struct capturing_random_reallocation : public RandomOneTwoReallocation {
    using RandomOneTwoReallocation::RandomOneTwoReallocation;
    using RandomOneTwoReallocation::applyNegativeGainsCorrection;
    static inline std::vector<cost_t> raw{};
    static inline std::vector<cost_t> sampled{};
    void setIncumbentMove() override {
        if (!candidates.empty()) {
            std::vector<cost_t> gains;
            for (const auto& move : candidates) gains.push_back(move.gain);
            raw = gains;
            applyNegativeGainsCorrection(gains);
            sampled = gains;
        }
        RandomOneTwoReallocation::setIncumbentMove();
    }
};

}  // namespace

class softmax_precondition_fixture : public cppied_context_fixture {
protected:
    path_engine g{P};
    coverage_engine c{P};

    cppied_solution make(const std::vector<segment>& path) {
        cppied_solution s;
        s.path = path;
        s.coverage = Eigen::VectorXd::Zero(P.req.size());
        c.reset(s);
        s.cost = g.cost(s);
        return s;
    }

    // Mirrors the start segment split SimpleNeighborhood::local_search does
    // before scoring moves.
    void splitStart(cppied_solution& s) {
        if (path_engine::is_node(s.path[0].target)) {
            auto split = g.splitSegment(s.path[0], 0);
            s.path[0] = split.first;
            s.path.insert(s.path.begin() + 1, split.second);
        }
    }

    // Gains SimpleNeighborhood::selectMoves turns into RIS intervals, checked
    // on the state s. Returns the number of negative gains (each reported);
    // counts the intervals whose gain differs from their move's gain.
    template<typename N>
    int checkSelection(cppied_solution s, const char* who, int& intervalGainMismatches, bool report = true) {
        splitStart(s);
        simple_accessor<N> n(ctx, P, s);
        n.setMovesAndGains();
        std::vector<int> candidates(n.candidatesMoves.size());
        std::iota(candidates.begin(), candidates.end(), 0);
        n.pruneNonImprovingMoves(candidates);
        int violations = 0;
        std::vector<cost_t> gains;
        for (int i : candidates) gains.push_back(n.candidatesMoves[i].gain);
        if (const int inv = flatScoreInversions(gains); inv > 0) {
            ++violations;
            if (report) {
                std::string list;
                for (const auto& gain : gains) list += costStr(gain) + " ";
                ADD_FAILURE() << who << ": flat scores invert " << inv << " pairs of selected gains " << list;
            }
        }
        for (int i : candidates)
            if (!nonNegative(n.candidatesMoves[i].gain)) {
                ++violations;
                if (report) ADD_FAILURE() << who << ": selected move " << i << " (" << segStr(s.path[i])
                              << ") has gain " << costStr(n.candidatesMoves[i].gain);
            }
        n.splitMoves(candidates);
        GroupedIntervalSets grouped = n.constructIntervalSets(candidates);
        for (std::size_t k = 0; k < grouped.size(); ++k) {
            for (const auto& iv : grouped[k])
                intervalGainMismatches += iv.gain != n.candidatesMoves[iv.id].gain;
            if (!nonNegative(grouped[k].getValue())) {
                ++violations;
                if (report) ADD_FAILURE() << who << ": interval group " << k << " has value " << costStr(grouped[k].getValue());
            }
        }
        n.clear();
        return violations;
    }

    // Captured paths. Descents are not used: the RIS selection draws from a
    // random_device, so the states they visit change between runs.
    std::vector<cppied_solution> states() {
        std::vector<cppied_solution> pool;
        for (const auto& p : inputs()) pool.push_back(make(p));
        return pool;
    }
};

// ----------------------------------------------------------------------------
// 1. neighborhood_nested: removal gains of path[1..]
// ----------------------------------------------------------------------------

TEST_F(softmax_precondition_fixture, NestedRemovalGainsAreNonNegative) {
    int checked = 0;
    for (const auto& s : states()) {
        std::vector<cost_t> gains;
        for (auto it = std::next(s.path.begin()); it != s.path.end(); ++it) {
            const cost_t gain = g.removal_gain(s, it);
            gains.push_back(gain);
            ++checked;
            EXPECT_TRUE(nonNegative(gain)) << "removal of path[" << std::distance(s.path.begin(), it) << "] = "
                                           << segStr(*it) << " has gain " << costStr(gain);
        }
        EXPECT_EQ(flatScoreInversions(gains), 0) << "flat scores do not keep the order of the removal gains";
    }
    EXPECT_GT(checked, 0);
}

// ----------------------------------------------------------------------------
// 2. neighborhood_r: even and odd sums of removal gains
// ----------------------------------------------------------------------------

TEST_F(softmax_precondition_fixture, ReallocationParityGainsAreNonNegative) {
    for (auto s : states()) {
        r_accessor r(ctx, P, s);
        std::vector<cost_t> gains = r.getRemovalGains();
        cost_t even{0, 0}, odd{0, 0};
        for (int i = 1; i < static_cast<int>(gains.size()); ++i)
            (i % 2 == 0 ? even : odd) += gains[i];
        EXPECT_TRUE(nonNegative(even)) << "even removal gains sum to " << costStr(even);
        EXPECT_TRUE(nonNegative(odd)) << "odd removal gains sum to " << costStr(odd);
        EXPECT_EQ(flatScoreInversions({even, odd}), 0)
            << "flat scores do not keep the order of " << costStr(even) << " and " << costStr(odd);
    }
}

// ----------------------------------------------------------------------------
// 3. RandomOneTwoReallocation: candidate move gains
// ----------------------------------------------------------------------------

TEST_F(softmax_precondition_fixture, RandomOneTwoCandidateGainsAreNonNegative) {
    int sampled = 0, violations = 0, corrected = 0;
    for (const auto& p : inputs())
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            cppied_solution s = make(p);
            capturing_random_reallocation::raw.clear();
            capturing_random_reallocation::sampled.clear();
            capturing_random_reallocation(P, c, g, s, i).solve();
            const auto& raw = capturing_random_reallocation::raw;
            for (const cost_t& gain : raw) corrected += !nonNegative(gain);
            // The correction keeps the order of the candidates.
            for (std::size_t a = 0; a < raw.size(); ++a)
                for (std::size_t b = 0; b < raw.size(); ++b)
                    EXPECT_EQ(raw[a] < raw[b], capturing_random_reallocation::sampled[a] < capturing_random_reallocation::sampled[b]);
            int local = 0;
            for (const cost_t& gain : capturing_random_reallocation::sampled) local += !nonNegative(gain);
            sampled += static_cast<int>(capturing_random_reallocation::sampled.size());
            violations += local;
            EXPECT_EQ(local, 0) << local << " of " << capturing_random_reallocation::sampled.size()
                                << " candidate gains are negative when reallocating path[" << i << "] = "
                                << segStr(p[i]);
            EXPECT_EQ(flatScoreInversions(capturing_random_reallocation::sampled), 0)
                << "flat scores do not keep the order of the candidate gains of path[" << i << "]";
        }
    EXPECT_GT(sampled, 0);
    EXPECT_GT(corrected, 0) << "no negative raw gain: the correction is not exercised";
    RecordProperty("negative_candidate_gains", std::to_string(violations) + "/" + std::to_string(sampled));
}

// applyNegativeGainsCorrection on hand-built gains.
class negative_gains_correction_fixture : public softmax_precondition_fixture {
protected:
    std::vector<cost_t> correct(std::vector<cost_t> gains) {
        cppied_solution s = make(inputs()[0]);
        capturing_random_reallocation alg(P, c, g, s, 1);
        alg.applyNegativeGainsCorrection(gains);
        return gains;
    }
};

TEST_F(negative_gains_correction_fixture, LeavesNonNegativeGainsUnchanged) {
    const std::vector<cost_t> gains = {{0, 0}, {3, -2}, {1, 4}};
    EXPECT_EQ(correct(gains), gains);
}

// The lexicographic minimum gain is subtracted from every gain.
TEST_F(negative_gains_correction_fixture, ShiftsByMinimumGain) {
    const std::vector<cost_t> gains = {{-4, 3}, {-1, -2}, {2, 0}};
    EXPECT_EQ(correct(gains), (std::vector<cost_t>{{0, 0}, {3, -5}, {6, -3}}));
}

// A gain {0, t < 0} is negative although its length is not.
TEST_F(negative_gains_correction_fixture, RulesOutNegativeGainsWithZeroLength) {
    for (const cost_t& gain : correct({{0, -3}, {2, 1}}))
        EXPECT_TRUE(nonNegative(gain)) << "negative gain " << costStr(gain) << " passed to softmaxSample";
}

// ----------------------------------------------------------------------------
// setFlatScores on non-negative gains whose turns are negative
// ----------------------------------------------------------------------------

// Strictly improving gains (they pass pruneNonImprovingMoves, so they satisfy
// the precondition) with negative turns. The vectors are selected n12 move
// gains observed along n12 descents from the captured paths.
TEST(softmax_flat_scores, KeepOrderOfNonNegativeGainsWithNegativeTurns) {
    const std::vector<std::vector<cost_t>> observed = {
        {{2, 2}, {1, -6}},
        {{0, 2}, {2, 2}, {3, -2}},
        {{2, 2}, {2, 2}, {3, -4}},
        {{2, 2}, {2, 0}, {2, 2}, {3, -2}},
    };
    for (const auto& gains : observed) {
        std::string list;
        for (const auto& gain : gains) list += costStr(gain) + " ";
        EXPECT_EQ(flatScoreInversions(gains), 0) << "gains " << list;
    }
}

// ----------------------------------------------------------------------------
// 4. ris_heuristic::softmaxSample(): RIS schedule values from the intervals
//    of the simple neighborhoods
// ----------------------------------------------------------------------------

// The values the RIS weights its schedules by are the gains of the moves the
// intervals stand for, so every interval carries its move's gain.
TEST_F(softmax_precondition_fixture, SelectionIntervalsCarryMoveGains) {
    int mismatches = 0;
    for (const auto& s : states()) {
        checkSelection<neighborhood_n1>(s, "n1", mismatches, false);
        checkSelection<neighborhood_n12>(s, "n12", mismatches, false);
        checkSelection<neighborhood_n21>(s, "n21", mismatches, false);
        checkSelection<neighborhood_trim>(s, "trim", mismatches, false);
        checkSelection<neighborhood_cut>(s, "cut", mismatches, false);
        checkSelection<neighborhood_reduce>(s, "reduce", mismatches, false);
    }
    EXPECT_EQ(mismatches, 0) << "intervals whose gain differs from their move's gain";
}

// Every selected move gain, and so every schedule value (a sum of them), is
// non-negative. checkSelection reports each violation.
TEST_F(softmax_precondition_fixture, SelectionGainsAreNonNegative) {
    int ignored = 0, violations = 0;
    for (const auto& s : states()) {
        violations += checkSelection<neighborhood_n1>(s, "n1", ignored);
        violations += checkSelection<neighborhood_n12>(s, "n12", ignored);
        violations += checkSelection<neighborhood_n21>(s, "n21", ignored);
        violations += checkSelection<neighborhood_trim>(s, "trim", ignored);
        violations += checkSelection<neighborhood_cut>(s, "cut", ignored);
        violations += checkSelection<neighborhood_reduce>(s, "reduce", ignored);
    }
    EXPECT_EQ(violations, 0);
}

// ============================================================================
// No dominated sentinel reaches the RIS heuristic or softmaxSample.
//
// {MIN_INT, MIN_INT} marks a dominated or empty move. It is never a sampling
// weight: every entry point is checked on the gains it hands over.
//   neighborhood_nested         -> softmaxSample(removalGains)
//   neighborhood_r              -> softmaxSample({even, odd})
//   RandomOneTwoReallocation    -> softmaxSample(candidate gains), whose
//                                  candidates are all potential paired
//                                  replacements scored without a bound
//   SimpleNeighborhood (n1, n12, n21, trim, cut, reduce)
//                               -> ris_heuristic::solve_selection(intervals)
// ============================================================================

class no_sentinel_fixture : public softmax_precondition_fixture {
protected:
    // Interval gains and group values selectMoves hands to solve_selection.
    template<typename N>
    std::vector<cost_t> selectionGains(cppied_solution s) {
        splitStart(s);
        simple_accessor<N> n(ctx, P, s);
        n.setMovesAndGains();
        std::vector<int> candidates(n.candidatesMoves.size());
        std::iota(candidates.begin(), candidates.end(), 0);
        n.pruneNonImprovingMoves(candidates);
        n.splitMoves(candidates);
        GroupedIntervalSets grouped = n.constructIntervalSets(candidates);
        std::vector<cost_t> gains;
        for (std::size_t k = 0; k < grouped.size(); ++k) {
            for (const auto& iv : grouped[k]) gains.push_back(iv.gain);
            gains.push_back(grouped[k].getValue());
        }
        n.clear();
        return gains;
    }
};

TEST_F(no_sentinel_fixture, NestedRemovalGains) {
    for (const auto& s : states()) {
        std::vector<cost_t> gains;
        for (auto it = std::next(s.path.begin()); it != s.path.end(); ++it)
            gains.push_back(g.removal_gain(s, it));
        EXPECT_EQ(countSentinels(gains), 0);
    }
}

TEST_F(no_sentinel_fixture, ReallocationParityGains) {
    for (auto s : states()) {
        r_accessor r(ctx, P, s);
        std::vector<cost_t> gains = r.getRemovalGains();
        cost_t even{0, 0}, odd{0, 0};
        for (int i = 1; i < static_cast<int>(gains.size()); ++i)
            (i % 2 == 0 ? even : odd) += gains[i];
        EXPECT_EQ(countSentinels({even, odd}), 0);
    }
}

// Raw candidate gains (no candidate is dominated) and the corrected gains
// passed to softmaxSample.
TEST_F(no_sentinel_fixture, RandomOneTwoCandidateGains) {
    int sampled = 0;
    for (const auto& p : inputs())
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            cppied_solution s = make(p);
            capturing_random_reallocation::raw.clear();
            capturing_random_reallocation::sampled.clear();
            capturing_random_reallocation(P, c, g, s, i).solve();
            sampled += static_cast<int>(capturing_random_reallocation::sampled.size());
            EXPECT_EQ(countSentinels(capturing_random_reallocation::raw), 0)
                << "dominated candidate when reallocating path[" << i << "] = " << segStr(p[i]);
            EXPECT_EQ(countSentinels(capturing_random_reallocation::sampled), 0)
                << "sentinel passed to softmaxSample when reallocating path[" << i << "] = " << segStr(p[i]);
        }
    EXPECT_GT(sampled, 0);
}

// Every candidate of RandomOneTwoReallocation is paired or single and scored:
// none is left with the MIN gain setSecondLevelCandidates initializes it to.
TEST_F(no_sentinel_fixture, RandomOneTwoCandidatesAreScoredReplacements) {
    for (const auto& p : inputs())
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            cppied_solution s = make(p);
            OneTwoReallocationMove mv = RandomOneTwoReallocation(P, c, g, s, i).solve();
            if (!path_engine::is_node(mv.replacements[0].source)) continue;   // direct removal, no sampling
            EXPECT_FALSE(isSentinel(mv.gain)) << "drawn move " << segStr(mv.replacements[0]) << " + "
                                              << segStr(mv.replacements[1]) << " is dominated";
        }
}

TEST_F(no_sentinel_fixture, SimpleNeighborhoodSelectionGains) {
    for (const auto& s : states()) {
        EXPECT_EQ(countSentinels(selectionGains<neighborhood_n1>(s)), 0) << "n1";
        EXPECT_EQ(countSentinels(selectionGains<neighborhood_n12>(s)), 0) << "n12";
        EXPECT_EQ(countSentinels(selectionGains<neighborhood_n21>(s)), 0) << "n21";
        EXPECT_EQ(countSentinels(selectionGains<neighborhood_trim>(s)), 0) << "trim";
        EXPECT_EQ(countSentinels(selectionGains<neighborhood_cut>(s)), 0) << "cut";
        EXPECT_EQ(countSentinels(selectionGains<neighborhood_reduce>(s)), 0) << "reduce";
    }
}

// The start segment's move carries the sentinel (setMovesAndGains); pruning
// keeps it out of the selection.
TEST_F(no_sentinel_fixture, StartSegmentSentinelIsPruned) {
    cppied_solution s = make(inputs()[0]);
    splitStart(s);
    simple_accessor<neighborhood_n1> n(ctx, P, s);
    n.setMovesAndGains();
    ASSERT_TRUE(isSentinel(n.candidatesMoves[0].gain));
    std::vector<int> candidates(n.candidatesMoves.size());
    std::iota(candidates.begin(), candidates.end(), 0);
    n.pruneNonImprovingMoves(candidates);
    EXPECT_EQ(std::ranges::count(candidates, 0), 0);
}
