#include "../test_fixture.hpp"
#include "../../include/algorithms/MinimumTopTwoInsertion.hpp"

// ============================================================================
// Tests of MinimumTopTwoInsertion: the two cheapest insertions of a segment
// at two different indexes, each in its cheaper orientation.
//
// Index convention (get_top_k_insertion, neighborhoods' apply): an insertion
// index k means "insert before path[k]", k in [1, size]; k == size appends.
//
// Oracle: for every index k, the cheaper of both orientations, measured with
// geometry.cost on the path with the segment inserted. bestInsertion is the
// minimum over all indexes, backupInsertion the minimum over the other ones.
// ============================================================================

namespace {

std::string segStr(const segment& s) {
    return "{" + std::to_string(s.source) + "," + std::to_string(s.target) + "}";
}

// Fixture paths captured from ils_full runs (all feasible).
std::vector<std::vector<segment>> inputs() {
    return {
        {{6, 11}, {23, 18}, {44, 42}, {48, 51}, {30, 35}, {75, 73}, {62, 63}},
        {{6, 11}, {79, path_engine::NULL_NODE}, {23, 18}, {44, 42}, {48, 51}, {30, 35}, {75, 74}, {10, 9}, {62, 63}},
        {{6,-1},{12,-2},{6,11},{79,80},{35,30},{45,42},{54,57},{18,23},{29,28},{63,62},{16,-1}},
        {{6,-1},{48,51},{30,35},{23,18},{44,-2},{6,9},{72,73},{62,63},{75,73},{11,-1}},
        {{6,-1},{7,11},{80,81},{29,-2},{77,-1},{71,68},{56,59},{53,48},{42,44},{18,-1},{20,22},{73,-2},{22,23}},
    };
}

// Horizontal and vertical, multi-vertex and single vertex, both orientations.
std::vector<segment> segmentsToInsert() {
    return {
        {7, 10}, {10, 7}, {0, 4}, {4, 0}, {12, 13}, {13, 12},
        {48, 52}, {52, 48}, {72, 75}, {75, 72}, {56, 59}, {59, 56},
        {20, path_engine::NULL_NODE}, {20, path_engine::REVERSED_NULL_NODE},
        {77, path_engine::NULL_NODE}, {77, path_engine::REVERSED_NULL_NODE},
    };
}

const cost_t kMax = {MAX_INT, MAX_INT};

}  // namespace

class minimum_top_two_insertion_fixture : public cppied_context_fixture {
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
    cost_t costOf(const std::vector<segment>& path) {
        cppied_solution s; s.path = path;
        return g.cost(s);
    }
    // Real cost difference of inserting s before path[index].
    cost_t insertionDelta(std::vector<segment> path, const segment& s, int index) {
        const cost_t before = costOf(path);
        path.insert(path.begin() + index, s);
        return costOf(path) - before;
    }
    // Cheapest orientation of s at every index in [1, size] (entry 0 unused).
    std::vector<cost_t> perIndexMinimum(const std::vector<segment>& path, const segment& s) {
        const segment o[2] = {g.getNonOriented(s), g.flip_segment(g.getNonOriented(s))};
        std::vector<cost_t> best(path.size() + 1, kMax);
        for (int k = 1; k <= static_cast<int>(path.size()); ++k)
            best[k] = std::min(insertionDelta(path, o[0], k), insertionDelta(path, o[1], k));
        return best;
    }

    // Checks one SegmentInsertion of the result.
    void checkInsertion(const std::vector<segment>& path, const segment& s, const SegmentInsertion& ins,
                        const char* which) {
        SCOPED_TRACE(which);
        EXPECT_EQ(g.getNonOriented(ins.s), g.getNonOriented(s))
            << "returned segment " << segStr(ins.s) << " is not an orientation of " << segStr(s);
        ASSERT_GE(ins.index, 1);
        ASSERT_LE(ins.index, static_cast<int>(path.size()));
        EXPECT_EQ(ins.insertion_cost, insertionDelta(path, ins.s, ins.index))
            << "reported cost does not match inserting " << segStr(ins.s) << " at " << ins.index;
        EXPECT_EQ(ins.insertion_cost, perIndexMinimum(path, s)[ins.index])
            << "not the cheaper orientation at index " << ins.index;
    }

    void checkTopTwo(const std::vector<segment>& path, const segment& s, const TopTwoInsertion& top) {
        SCOPED_TRACE("insert " + segStr(s) + " in path of size " + std::to_string(path.size()));
        std::vector<cost_t> per_index = perIndexMinimum(path, s);
        auto first = std::min_element(per_index.begin() + 1, per_index.end());
        const int best_index = static_cast<int>(std::distance(per_index.begin(), first));
        cost_t backup = kMax;
        for (int k = 1; k < static_cast<int>(per_index.size()); ++k)
            if (k != best_index) backup = std::min(backup, per_index[k]);

        checkInsertion(path, s, top.bestInsertion, "bestInsertion");
        EXPECT_EQ(top.bestInsertion.insertion_cost, *first) << "best is not the minimum over all indexes";
        if (path.size() < 2) {
            EXPECT_EQ(top.backupInsertion.insertion_cost, kMax) << "backup without a second index";
            return;
        }
        checkInsertion(path, s, top.backupInsertion, "backupInsertion");
        EXPECT_NE(top.bestInsertion.index, top.backupInsertion.index) << "both insertions at the same index";
        EXPECT_LE(top.bestInsertion.insertion_cost, top.backupInsertion.insertion_cost);
        EXPECT_EQ(top.backupInsertion.insertion_cost, backup) << "backup is not the minimum over the other indexes";
    }
};

// ----------------------------------------------------------------------------
// Contract against the brute force
// ----------------------------------------------------------------------------

TEST_F(minimum_top_two_insertion_fixture, MatchesBruteForceOnCapturedPaths) {
    for (const auto& p : inputs())
        for (const segment& s : segmentsToInsert()) {
            cppied_solution sol = make(p);
            const cppied_solution before = sol;
            TopTwoInsertion top = MinimumTopTwoInsertion(P, c, g, sol, s).solve();
            EXPECT_EQ(sol.path, before.path) << "path modified";
            EXPECT_TRUE(sol.coverage.isApprox(before.coverage)) << "tracked coverage modified";
            checkTopTwo(p, s, top);
        }
}

TEST_F(minimum_top_two_insertion_fixture, ReinsertRemovedSegment) {
    for (const auto& p : inputs())
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            auto reduced = p; reduced.erase(reduced.begin() + i);
            cppied_solution sol = make(reduced);
            checkTopTwo(reduced, p[i], MinimumTopTwoInsertion(P, c, g, sol, p[i]).solve());
        }
}

// Both orientations of the input give the same costs.
TEST_F(minimum_top_two_insertion_fixture, OrientationOfInputIsIrrelevant) {
    for (const auto& p : inputs())
        for (const segment& s : segmentsToInsert()) {
            SCOPED_TRACE("insert " + segStr(s));
            cppied_solution a = make(p), b = make(p);
            TopTwoInsertion ta = MinimumTopTwoInsertion(P, c, g, a, s).solve();
            TopTwoInsertion tb = MinimumTopTwoInsertion(P, c, g, b, g.flip_segment(s)).solve();
            EXPECT_EQ(ta.bestInsertion.insertion_cost, tb.bestInsertion.insertion_cost);
            EXPECT_EQ(ta.backupInsertion.insertion_cost, tb.backupInsertion.insertion_cost);
        }
}

// ----------------------------------------------------------------------------
// Positions at the end of the path
// ----------------------------------------------------------------------------

// Insertion before the last segment (index size - 1) is a candidate: a
// segment placed right before the last one is found there.
TEST_F(minimum_top_two_insertion_fixture, InsertionBeforeLastSegmentIsConsidered) {
    int checked = 0;
    for (const auto& p : inputs())
        for (const segment& s : segmentsToInsert()) {
            std::vector<cost_t> per_index = perIndexMinimum(p, s);
            const int n = static_cast<int>(p.size());
            if (!(per_index[n - 1] < *std::min_element(per_index.begin() + 1, per_index.end() - 2)) ||
                !(per_index[n - 1] < per_index[n]))
                continue;
            ++checked;
            SCOPED_TRACE("insert " + segStr(s));
            cppied_solution sol = make(p);
            TopTwoInsertion top = MinimumTopTwoInsertion(P, c, g, sol, s).solve();
            EXPECT_EQ(top.bestInsertion.index, n - 1);
            EXPECT_EQ(top.bestInsertion.insertion_cost, per_index[n - 1]);
        }
    // Constructed case: reinsert the second to last segment.
    for (const auto& p : inputs()) {
        const int n = static_cast<int>(p.size());
        auto reduced = p; reduced.erase(reduced.begin() + n - 2);
        cppied_solution sol = make(reduced);
        TopTwoInsertion top = MinimumTopTwoInsertion(P, c, g, sol, p[n - 2]).solve();
        const std::vector<cost_t> per_index = perIndexMinimum(reduced, p[n - 2]);
        EXPECT_LE(top.bestInsertion.insertion_cost, per_index[n - 2])
            << "reinserting " << segStr(p[n - 2]) << " before the last segment is cheaper";
    }
    RecordProperty("before_last_cases", checked);
}

// Single segment path: the tail is the only index; no backup.
TEST_F(minimum_top_two_insertion_fixture, SingleSegmentPathHasOnlyTail) {
    for (const std::vector<segment>& p : std::vector<std::vector<segment>>{{{6, -1}}, {{6, 11}}, {{23, 18}}})
        for (const segment& s : segmentsToInsert()) {
            cppied_solution sol = make(p);
            TopTwoInsertion top = MinimumTopTwoInsertion(P, c, g, sol, s).solve();
            checkTopTwo(p, s, top);
            EXPECT_EQ(top.bestInsertion.index, 1);
        }
}

// Two segment path: both indexes 1 and 2 are returned.
TEST_F(minimum_top_two_insertion_fixture, TwoSegmentPathReturnsBothIndexes) {
    for (const std::vector<segment>& p : std::vector<std::vector<segment>>{{{6, -1}, {23, 18}}, {{6, 11}, {44, 42}}})
        for (const segment& s : segmentsToInsert()) {
            cppied_solution sol = make(p);
            TopTwoInsertion top = MinimumTopTwoInsertion(P, c, g, sol, s).solve();
            checkTopTwo(p, s, top);
            EXPECT_EQ((std::set<int>{top.bestInsertion.index, top.backupInsertion.index}), (std::set<int>{1, 2}));
        }
}

// ----------------------------------------------------------------------------
// resetSegment: reusing the solver gives the result of a fresh one.
// ----------------------------------------------------------------------------

TEST_F(minimum_top_two_insertion_fixture, ResetSegmentMatchesFreshSolver) {
    for (const auto& p : inputs()) {
        cppied_solution sol = make(p);
        const auto segments = segmentsToInsert();
        segment first = segments.front();
        MinimumTopTwoInsertion reused(P, c, g, sol, first);
        reused.solve();
        for (segment s : segments) {
            SCOPED_TRACE("insert " + segStr(s));
            reused.resetSegment(s);
            TopTwoInsertion r = reused.solve();
            TopTwoInsertion f = MinimumTopTwoInsertion(P, c, g, sol, s).solve();
            EXPECT_EQ(r.bestInsertion.insertion_cost, f.bestInsertion.insertion_cost);
            EXPECT_EQ(r.backupInsertion.insertion_cost, f.backupInsertion.insertion_cost);
            EXPECT_EQ(r.bestInsertion.index, f.bestInsertion.index);
            EXPECT_EQ(r.backupInsertion.index, f.backupInsertion.index);
        }
    }
}
