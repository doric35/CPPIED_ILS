#include "../test_fixture.hpp"
#include "../../include/algorithms/MinimumSegmentInsertion.hpp"

// ============================================================================
// Tests of MinimumSegmentInsertion: the cheapest insertion of a segment in a
// path over every position and both orientations of the segment.
//
// Index convention (get_top_k_insertion, neighborhoods' apply): an insertion
// index k means "insert before path[k]", k in [1, size]; k == size appends.
//
// The returned move is checked against a brute force that inserts every
// orientation at every index and measures the real cost difference with
// geometry.cost(solution). solve() must leave the solution unchanged.
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

// Segments to insert, horizontal and vertical, multi-vertex and single vertex,
// given in both orientations.
std::vector<segment> segmentsToInsert() {
    return {
        {7, 10}, {10, 7}, {0, 4}, {4, 0}, {12, 13}, {13, 12},
        {48, 52}, {52, 48}, {72, 75}, {75, 72}, {56, 59}, {59, 56},
        {20, path_engine::NULL_NODE}, {20, path_engine::REVERSED_NULL_NODE},
        {77, path_engine::NULL_NODE}, {77, path_engine::REVERSED_NULL_NODE},
    };
}

}  // namespace

class minimum_segment_insertion_fixture : public cppied_context_fixture {
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
    static std::vector<segment> inserted(std::vector<segment> path, const segment& s, int index) {
        path.insert(path.begin() + index, s);
        return path;
    }
    // Real cost difference of inserting s before path[index].
    cost_t insertionDelta(const std::vector<segment>& path, const segment& s, int index) {
        return costOf(inserted(path, s, index)) - costOf(path);
    }
    // Brute force minimum over every index in [1, size] and both orientations.
    cost_t bruteMinimum(const std::vector<segment>& path, const segment& s) {
        const segment orientations[2] = {g.getNonOriented(s), g.flip_segment(g.getNonOriented(s))};
        cost_t best{std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
        for (int k = 1; k <= static_cast<int>(path.size()); ++k)
            for (const segment& o : orientations)
                best = std::min(best, insertionDelta(path, o, k));
        return best;
    }

    // Checks one solve() call against the oracle.
    void checkInsertion(const std::vector<segment>& path, const segment& s) {
        SCOPED_TRACE("insert " + segStr(s) + " in path of size " + std::to_string(path.size()));
        cppied_solution sol = make(path);
        const cppied_solution before = sol;
        SegmentInsertion ins = MinimumSegmentInsertion(P, c, g, sol, s).solve();

        EXPECT_EQ(sol.path, before.path) << "path modified";
        EXPECT_TRUE(sol.coverage.isApprox(before.coverage)) << "tracked coverage modified";

        // The returned segment is one of the two orientations of s.
        EXPECT_EQ(g.getNonOriented(ins.s), g.getNonOriented(s))
            << "returned segment " << segStr(ins.s) << " is not an orientation of " << segStr(s);

        // The index lies in [1, size].
        ASSERT_GE(ins.index, 1);
        ASSERT_LE(ins.index, static_cast<int>(path.size()));

        // The reported cost is the real cost of applying the returned move.
        EXPECT_EQ(ins.insertion_cost, insertionDelta(path, ins.s, ins.index))
            << "reported cost does not match inserting " << segStr(ins.s) << " at " << ins.index;

        // The move is the cheapest one over all positions and orientations.
        EXPECT_EQ(ins.insertion_cost, bruteMinimum(path, s)) << "not the minimum insertion";
    }
};

// ----------------------------------------------------------------------------
// Contract against the brute force on captured paths
// ----------------------------------------------------------------------------

TEST_F(minimum_segment_insertion_fixture, MatchesBruteForceOnCapturedPaths) {
    for (const auto& p : inputs())
        for (const segment& s : segmentsToInsert())
            checkInsertion(p, s);
}

// Insertion of a segment already present in the path (removed then reinserted).
TEST_F(minimum_segment_insertion_fixture, ReinsertRemovedSegment) {
    for (const auto& p : inputs())
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            auto reduced = p; reduced.erase(reduced.begin() + i);
            checkInsertion(reduced, p[i]);
        }
}

// ----------------------------------------------------------------------------
// Orientation invariance: both orientations of s give the same move cost.
// ----------------------------------------------------------------------------

TEST_F(minimum_segment_insertion_fixture, OrientationOfInputIsIrrelevant) {
    for (const auto& p : inputs())
        for (const segment& s : segmentsToInsert()) {
            SCOPED_TRACE("insert " + segStr(s));
            cppied_solution a = make(p), b = make(p);
            SegmentInsertion ia = MinimumSegmentInsertion(P, c, g, a, s).solve();
            SegmentInsertion ib = MinimumSegmentInsertion(P, c, g, b, g.flip_segment(s)).solve();
            EXPECT_EQ(ia.insertion_cost, ib.insertion_cost);
        }
}

// ----------------------------------------------------------------------------
// Terminal insertion: on a single-segment path only the tail is available.
// ----------------------------------------------------------------------------

TEST_F(minimum_segment_insertion_fixture, SingleSegmentPathAppends) {
    const std::vector<std::vector<segment>> paths = {
        {{6, path_engine::NULL_NODE}},
        {{6, 11}},
        {{23, 18}},
    };
    for (const auto& p : paths)
        for (const segment& s : segmentsToInsert()) {
            checkInsertion(p, s);
            cppied_solution sol = make(p);
            SegmentInsertion ins = MinimumSegmentInsertion(P, c, g, sol, s).solve();
            EXPECT_EQ(ins.index, 1) << "insert " << segStr(s);
        }
}

// Empty path: no insertion position; solve() does not touch the path and
// returns the unbounded cost.
TEST_F(minimum_segment_insertion_fixture, EmptyPathHasNoInsertion) {
    cppied_solution sol;
    sol.coverage = Eigen::VectorXd::Zero(P.req.size());
    SegmentInsertion ins = MinimumSegmentInsertion(P, c, g, sol, segment{7, 10}).solve();
    EXPECT_TRUE(sol.path.empty());
    EXPECT_EQ(ins.insertion_cost, (cost_t{MAX_INT, MAX_INT}));
}

// ----------------------------------------------------------------------------
// Hand-built cases where the optimum is known.
// ----------------------------------------------------------------------------

// The tail is strictly the best position: s lies right after the last segment.
TEST_F(minimum_segment_insertion_fixture, TailIsChosenWhenStrictlyBest) {
    for (const auto& p : inputs()) {
        const segment last = p.back();
        // Pick, among the candidates, a segment whose tail insertion is
        // strictly cheaper than every non-terminal insertion.
        for (const segment& s : segmentsToInsert()) {
            const segment o[2] = {g.getNonOriented(s), g.flip_segment(g.getNonOriented(s))};
            const int n = static_cast<int>(p.size());
            cost_t tail = std::min(insertionDelta(p, o[0], n), insertionDelta(p, o[1], n));
            cost_t inner{std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
            for (int k = 1; k < n; ++k)
                for (const segment& x : o) inner = std::min(inner, insertionDelta(p, x, k));
            if (!(tail < inner)) continue;
            SCOPED_TRACE("insert " + segStr(s) + " after " + segStr(last));
            cppied_solution sol = make(p);
            SegmentInsertion ins = MinimumSegmentInsertion(P, c, g, sol, s).solve();
            EXPECT_EQ(ins.index, n);
            EXPECT_EQ(ins.insertion_cost, tail);
            EXPECT_EQ(insertionDelta(p, ins.s, ins.index), tail);
        }
    }
}

// The flipped orientation at the tail is strictly the best move: the reported
// cost must be the cost of the flipped orientation.
TEST_F(minimum_segment_insertion_fixture, FlippedTailCostIsReported) {
    int checked = 0;
    for (const auto& p : inputs())
        for (const segment& s : segmentsToInsert()) {
            const segment no = g.getNonOriented(s);
            const segment fl = g.flip_segment(no);
            const int n = static_cast<int>(p.size());
            const cost_t best = bruteMinimum(p, s);
            const cost_t tailFlipped = insertionDelta(p, fl, n);
            if (!(tailFlipped == best && insertionDelta(p, no, n) > best)) continue;
            ++checked;
            SCOPED_TRACE("insert " + segStr(s));
            cppied_solution sol = make(p);
            SegmentInsertion ins = MinimumSegmentInsertion(P, c, g, sol, s).solve();
            EXPECT_EQ(ins.insertion_cost, tailFlipped);
        }
    // Also a constructed case: append right after the last segment, coming back.
    std::vector<segment> p = {{6, 11}};
    const segment s = {17, 12};  // next row, W orientation is natural after {6,11} ending at 11
    const cost_t best = bruteMinimum(p, s);
    cppied_solution sol = make(p);
    SegmentInsertion ins = MinimumSegmentInsertion(P, c, g, sol, s).solve();
    EXPECT_EQ(ins.insertion_cost, best);
    EXPECT_EQ(insertionDelta(p, ins.s, ins.index), best);
    RecordProperty("flipped_tail_cases", checked);
}
