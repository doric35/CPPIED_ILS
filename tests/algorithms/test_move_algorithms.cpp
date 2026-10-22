#include "../test_fixture.hpp"
#include "../../include/algorithms/MaximumReplacement.hpp"
#include "../../include/algorithms/MaximumSegmentTrim.hpp"
#include "../../include/algorithms/MaximumSegmentCut.hpp"
#include "../../include/algorithms/SatisfySegmentReduce.hpp"
#include "../../include/algorithms/MinimumSegmentAllocation.hpp"
#include "../../include/algorithms/MinimumSegmentPairInsertion.hpp"
#include "../../include/algorithms/MaximumOneTwoReallocation.hpp"
#include "../../include/algorithms/MaximumTwoOneReallocation.hpp"
#include "../../include/algorithms/RandomOneTwoReallocation.hpp"

// ============================================================================
// Contract tests of the move-search algorithms used by the neighborhoods.
//
// Every returned move is checked against an independent oracle: the real
// cost difference of applying it (geometry.cost before/after), coverage
// feasibility after applying it, and, where the algorithm is exact over a
// candidate set, a brute force over that set. solve() must leave the
// solution (path and tracked coverage) unchanged.
//
// Index convention (get_top_k_insertion, neighborhoods' apply): an insertion
// index k means "insert before path[k]", k in [1, size]; k == size appends.
// ============================================================================

namespace {

std::string segStr(const segment& s) {
    return "{" + std::to_string(s.source) + "," + std::to_string(s.target) + "}";
}

const segment kNull = {path_engine::NULL_NODE, path_engine::NULL_NODE};

// Fixture paths and n21 inputs captured from ils_full runs (all feasible).
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

}  // namespace

class move_algorithms_fixture : public cppied_context_fixture {
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
    bool feasible(std::vector<segment> path) {
        cppied_solution s = make(path);
        return (s.coverage.array() >= P.req.array()).all();
    }
    cost_t costOf(const std::vector<segment>& path) {
        cppied_solution s; s.path = path;
        return g.cost(s);
    }
    // Insert segments before path[index] (in the given path's indexing; a
    // stable order for equal indexes), then drop the removed indexes.
    static std::vector<segment> apply(const std::vector<segment>& path, std::set<int> removed,
                                      std::vector<std::pair<segment, int>> inserts) {
        std::erase_if(inserts, [](const auto& p){ return !path_engine::is_node(p.first.source); });
        std::stable_sort(inserts.begin(), inserts.end(), [](auto& a, auto& b){ return a.second < b.second; });
        std::vector<segment> out;
        auto it = inserts.begin();
        for (int k = 0; k < static_cast<int>(path.size()); ++k) {
            while (it != inserts.end() && it->second == k) out.push_back((it++)->first);
            if (!removed.count(k)) out.push_back(path[k]);
        }
        for (; it != inserts.end(); ++it) out.push_back(it->first);
        return out;
    }
    // solve() must not leave side effects on the solution.
    void expectUnchanged(const cppied_solution& before, const cppied_solution& after, const std::string& what) {
        EXPECT_EQ(after.path, before.path) << what << ": path modified";
        EXPECT_TRUE(after.coverage.isApprox(before.coverage)) << what << ": tracked coverage modified";
    }
    std::vector<int> unsatisfiedAfterRemoving(cppied_solution s, int i) {
        c.remove(s, s.path[i]);
        std::vector<int> cells;
        c.unsatisfied_cells(s, s.path[i], cells);
        return cells;
    }
    std::set<int> vertices(const segment& s) {
        std::set<int> vs;
        if (!path_engine::is_node(s.source)) return vs;
        segment u = g.getNonOriented(s);
        int last = path_engine::is_node(u.target) ? u.target : u.source;
        for (int v = u.source; v <= last; ++v) vs.insert(v);
        return vs;
    }
    bool sameLine(const segment& a, const segment& b) {
        if (g.is_horizontal(a) != g.is_horizontal(b)) return false;
        return g.is_horizontal(a) ? g.get_row(a) == g.get_row(b) : g.get_column(a) == g.get_column(b);
    }
};

// ----------------------------------------------------------------------------
// MaximumReplacement (n1)
// ----------------------------------------------------------------------------

TEST_F(move_algorithms_fixture, MaximumReplacementContract) {
    const int R = P.max_range;
    int checked = 0;
    for (const auto& p : inputs()) {
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            cppied_solution s = make(p);
            const cppied_solution before = s;
            ReplacementMove mv = MaximumReplacement(P, c, g, s, i).solve();
            SCOPED_TRACE("path[" + std::to_string(i) + "] = " + segStr(p[i]));
            expectUnchanged(before, s, "MaximumReplacement");
            std::vector<int> unsat = unsatisfiedAfterRemoving(s, i);
            if (unsat.empty()) {
                EXPECT_FALSE(path_engine::is_node(mv.s.source)) << "removable segment must be dropped";
                EXPECT_EQ(mv.gain, g.removal_gain(s, s.path.begin() + i));
                continue;
            }
            // Brute force over the algorithm's candidate set: complete-coverage
            // lines spanning the unsatisfied box, both orientations.
            iRectangle box;
            c.unsatisfied_rectangle(unsat, box);
            cost_t best = {MIN_INT, MIN_INT};
            std::vector<segment> cands;
            for (int row = std::max(0, box.lr.y - R); row <= std::min(g.n_rows, box.ul.y + R); ++row) {
                segment h{row * g.n_cols + box.ul.x, row * g.n_cols + box.lr.x - 1};
                if (h.source == h.target) h.target = path_engine::NULL_NODE;
                cands.push_back(h);
            }
            for (int col = std::max(0, box.lr.x - R); col <= std::min(g.n_cols, box.ul.x + R); ++col) {
                segment v{g.horizontal_bound + col * g.n_rows + box.ul.y, g.horizontal_bound + col * g.n_rows + box.lr.y - 1};
                if (v.source == v.target) v.target = path_engine::NULL_NODE;
                cands.push_back(v);
            }
            const size_t n = cands.size();
            for (size_t k = 0; k < n; ++k) cands.push_back(g.flip_segment(cands[k]));
            for (const auto& r : cands) {
                auto q = p; q[i] = r;
                if (!feasible(q)) continue;
                cost_t real = costOf(p) - costOf(q);
                if (real > cost_t{0, 0}) best = std::max(best, real);
            }
            if (best > cost_t{MIN_INT, MIN_INT}) {
                ASSERT_TRUE(path_engine::is_node(mv.s.source)) << "an improving replacement exists";
                auto q = p; q[i] = mv.s;
                EXPECT_TRUE(feasible(q)) << "replacement " << segStr(mv.s) << " breaks coverage";
                EXPECT_EQ(mv.gain, costOf(p) - costOf(q)) << "predicted gain differs from the real one";
                EXPECT_EQ(mv.gain, best) << "not the best complete-coverage replacement";
            } else {
                EXPECT_FALSE((mv.gain > cost_t{0, 0})) << "improving move " << segStr(mv.s) << " where none exists";
            }
            ++checked;
        }
    }
    EXPECT_GT(checked, 0);
}

// ----------------------------------------------------------------------------
// MaximumSegmentTrim (trim)
// ----------------------------------------------------------------------------

TEST_F(move_algorithms_fixture, MaximumSegmentTrimContract) {
    for (const auto& p : inputs()) {
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            cppied_solution s = make(p);
            const cppied_solution before = s;
            ReplacementMove mv = MaximumSegmentTrim(P, c, g, s, i).solve();
            SCOPED_TRACE("path[" + std::to_string(i) + "] = " + segStr(p[i]));
            expectUnchanged(before, s, "MaximumSegmentTrim");
            auto q = p;
            if (path_engine::is_node(mv.s.source)) {
                // A sub-segment of the original, on its line, same direction.
                EXPECT_TRUE(sameLine(mv.s, p[i])) << segStr(mv.s);
                auto vs = vertices(mv.s), vp = vertices(p[i]);
                EXPECT_TRUE(std::includes(vp.begin(), vp.end(), vs.begin(), vs.end())) << segStr(mv.s) << " not inside";
                EXPECT_EQ(g.get_direction(mv.s), g.get_direction(p[i])) << segStr(mv.s) << " changed direction";
                q[i] = mv.s;
                // Maximally trimmed: both end vertices are needed.
                cppied_solution t = make(q);
                segment u = g.getNonOriented(mv.s);
                int last = path_engine::is_node(u.target) ? u.target : u.source;
                EXPECT_FALSE(c.can_remove_vertex(t, u.source)) << segStr(mv.s) << " head still removable";
                EXPECT_FALSE(c.can_remove_vertex(t, last)) << segStr(mv.s) << " tail still removable";
            } else {
                q.erase(q.begin() + i);
            }
            EXPECT_TRUE(feasible(q)) << "trim " << segStr(mv.s) << " breaks coverage";
            EXPECT_EQ(mv.gain, costOf(p) - costOf(q)) << "predicted gain differs from the real one";
        }
    }
}

// ----------------------------------------------------------------------------
// MaximumSegmentCut (cut)
// ----------------------------------------------------------------------------

TEST_F(move_algorithms_fixture, MaximumSegmentCutContract) {
    for (const auto& p : inputs()) {
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            cppied_solution s = make(p);
            const cppied_solution before = s;
            CutMove mv = MaximumSegmentCut(P, c, g, s, i).solve();
            SCOPED_TRACE("path[" + std::to_string(i) + "] = " + segStr(p[i]));
            expectUnchanged(before, s, "MaximumSegmentCut");
            // Longest run of individually removable vertices of path[i].
            segment u = g.getNonOriented(p[i]);
            int last = path_engine::is_node(u.target) ? u.target : u.source;
            int longest = 0, run = 0;
            for (int v = u.source; v <= last; ++v) {
                run = c.can_remove_vertex(s, v) ? run + 1 : 0;
                longest = std::max(longest, run);
            }
            if (longest == 0) {
                EXPECT_FALSE((mv.gain > cost_t{0, 0})) << "cut proposed without removable vertex";
                continue;
            }
            ASSERT_TRUE(path_engine::is_node(mv.s.source));
            auto cut = vertices(mv.s), seg = vertices(p[i]);
            EXPECT_TRUE(std::includes(seg.begin(), seg.end(), cut.begin(), cut.end())) << segStr(mv.s) << " not inside";
            EXPECT_EQ(static_cast<int>(cut.size()), longest) << "not the longest removable run";
            EXPECT_EQ(mv.gain, (cost_t{longest, 0}));
            // Removing the run keeps coverage.
            std::vector<segment> pieces;
            std::vector<int> left, right;
            for (int v : seg) (v < *cut.begin() ? left : right).push_back(v);
            std::erase_if(right, [&](int v){ return cut.count(v); });
            auto q = p; q.erase(q.begin() + i);
            for (auto* part : {&left, &right})
                if (!part->empty())
                    q.push_back(part->size() == 1 ? segment{part->front(), path_engine::NULL_NODE}
                                                  : segment{part->front(), part->back()});
            EXPECT_TRUE(feasible(q)) << "removing " << segStr(mv.s) << " breaks coverage";
        }
    }
}

// ----------------------------------------------------------------------------
// SatisfySegmentReduce (reduce)
// ----------------------------------------------------------------------------

TEST_F(move_algorithms_fixture, SatisfySegmentReduceContract) {
    for (const auto& p : inputs()) {
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            cppied_solution s = make(p);
            const cppied_solution before = s;
            MoveReduce mv = SatisfySegmentReduce(P, c, g, s, i).solve();
            SCOPED_TRACE("path[" + std::to_string(i) + "] = " + segStr(p[i]));
            expectUnchanged(before, s, "SatisfySegmentReduce");
            auto q = p; q.erase(q.begin() + i);
            if (feasible(q)) {
                cost_t real = costOf(p) - costOf(q);
                EXPECT_EQ(mv.s, p[i]);
                EXPECT_EQ(mv.gain, (real <= cost_t{0, 0} ? real + cost_t{0, 1} : real));
            } else {
                EXPECT_FALSE((mv.gain > cost_t{0, 0})) << "non-removable segment proposed for removal";
            }
        }
    }
}

// ----------------------------------------------------------------------------
// MinimumSegmentAllocation (used by n21)
// ----------------------------------------------------------------------------

TEST_F(move_algorithms_fixture, MinimumSegmentAllocationContract) {
    const int R = P.max_range;
    int allocated = 0;
    for (const auto& p : inputs()) {
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            auto reduced = p; reduced.erase(reduced.begin() + i);
            cppied_solution s = make(reduced);
            const cppied_solution before = s;
            Allocation a = MinimumSegmentAllocation(P, c, g, s).solve();
            SCOPED_TRACE("without path[" + std::to_string(i) + "] = " + segStr(p[i]));
            expectUnchanged(before, s, "MinimumSegmentAllocation");
            std::vector<int> unsat;
            c.unsatisfied_cells(s, unsat);
            if (unsat.empty()) {
                EXPECT_FALSE(path_engine::is_node(a.s.source)) << "allocation on a feasible solution";
                continue;
            }
            // Brute force: complete-coverage lines x orientations x positions.
            iRectangle box;
            c.unsatisfied_rectangle(unsat, box);
            std::vector<segment> cands;
            for (int row = std::max(0, box.lr.y - R); row <= std::min(g.n_rows, box.ul.y + R); ++row) {
                segment h{row * g.n_cols + box.ul.x, row * g.n_cols + box.lr.x - 1};
                if (h.source == h.target) h.target = path_engine::NULL_NODE;
                cands.push_back(h);
            }
            for (int col = std::max(0, box.lr.x - R); col <= std::min(g.n_cols, box.ul.x + R); ++col) {
                segment v{g.horizontal_bound + col * g.n_rows + box.ul.y, g.horizontal_bound + col * g.n_rows + box.lr.y - 1};
                if (v.source == v.target) v.target = path_engine::NULL_NODE;
                cands.push_back(v);
            }
            cost_t best = {MAX_INT, MAX_INT};
            for (const auto& base : cands)
                for (const segment& r : {base, g.flip_segment(base)}) {
                    if (!feasible(apply(reduced, {}, {{r, static_cast<int>(reduced.size())}}))) break;  // coverage is order-free
                    for (int k = 1; k <= static_cast<int>(reduced.size()); ++k)
                        best = std::min(best, costOf(apply(reduced, {}, {{r, k}})) - costOf(reduced));
                }
            if (best == cost_t{MAX_INT, MAX_INT}) {
                EXPECT_FALSE(path_engine::is_node(a.s.source)) << "allocation " << segStr(a.s) << " where none satisfies";
                continue;
            }
            ASSERT_TRUE(path_engine::is_node(a.s.source)) << "a satisfying allocation exists";
            ASSERT_GE(a.index, 1);
            ASSERT_LE(a.index, static_cast<int>(reduced.size()));
            auto q = apply(reduced, {}, {{a.s, a.index}});
            EXPECT_TRUE(feasible(q)) << "allocation " << segStr(a.s) << " does not restore coverage";
            EXPECT_EQ(a.cost, costOf(q) - costOf(reduced)) << "predicted cost differs from the real one";
            EXPECT_EQ(a.cost, best) << "not the cheapest complete-coverage allocation";
            ++allocated;
        }
    }
    EXPECT_GT(allocated, 0);
}

// ----------------------------------------------------------------------------
// MinimumSegmentPairInsertion (used by n12 and nested)
//
// Precondition: both segments are valid (is_node(source)). Single segment
// insertions go through MinimumSegmentInsertion
// (test_minimum_segment_insertion.cpp).
// ----------------------------------------------------------------------------

// The pair insertion is exact. Every configuration is either
//   disjoint: the segments in two different gaps, whose costs add up, so the
//     cheapest one combines the top two insertions of each segment
//     (MinimumTopTwoInsertion): best + best when their indexes differ, else
//     the cheaper of best + backup and backup + best;
//   paired: both segments in the same gap, in one of the 8 orders and
//     orientations (indexes {k, k}).
// The returned plan is consistent (ordered indexes, cost equal to the real
// cost of inserting both) and equals the brute force optimum over all
// positions, orders and orientations.
class pair_insertion_fixture : public move_algorithms_fixture {
protected:
    cost_t bruteForce(const std::vector<segment>& path, const segment& first, const segment& second) {
        cost_t best = {MAX_INT, MAX_INT};
        const int n = static_cast<int>(path.size());
        for (const segment& a : {first, g.flip_segment(first)})
            for (const segment& b : {second, g.flip_segment(second)})
                for (int x = 1; x <= n; ++x)
                    for (int y = 1; y <= n; ++y) {
                        best = std::min(best, costOf(apply(path, {}, {{a, x}, {b, y}})) - costOf(path));
                        best = std::min(best, costOf(apply(path, {}, {{b, x}, {a, y}})) - costOf(path));
                    }
        return best;
    }
    // Returns true when the optimum is a paired configuration.
    bool checkPair(const std::vector<segment>& path, const segment& first, const segment& second) {
        SCOPED_TRACE("insert " + segStr(first) + " + " + segStr(second) + " in path of size " +
                     std::to_string(path.size()));
        cppied_solution s = make(path);
        const cppied_solution before = s;
        MinimumSegmentPairInsertion alg(P, c, g, s, cost_t{MAX_INT, MAX_INT});
        segment a = first, b = second;
        alg.setSegmentPair(a, b);
        Insertion ins = alg.solve();
        expectUnchanged(before, s, "MinimumSegmentPairInsertion");
        EXPECT_TRUE(path_engine::is_node(ins.segments.first.source));
        EXPECT_TRUE(path_engine::is_node(ins.segments.second.source));
        EXPECT_EQ(std::set({g.getNonOriented(ins.segments.first), g.getNonOriented(ins.segments.second)}),
                  std::set({g.getNonOriented(first), g.getNonOriented(second)}))
            << "returned segments are not orientations of the inserted ones";
        EXPECT_GE(ins.indexes.first, 1);
        EXPECT_LE(ins.indexes.first, ins.indexes.second) << "unordered indexes";
        EXPECT_LE(ins.indexes.second, static_cast<int>(path.size()));
        auto q = apply(path, {}, {{ins.segments.first, ins.indexes.first}, {ins.segments.second, ins.indexes.second}});
        EXPECT_EQ(ins.cost, costOf(q) - costOf(path)) << "predicted cost differs from the real one ("
                                                       << segStr(ins.segments.first) << "@" << ins.indexes.first << ", "
                                                       << segStr(ins.segments.second) << "@" << ins.indexes.second << ")";
        EXPECT_EQ(ins.cost, bruteForce(path, first, second)) << "not the cheapest pair insertion";
        return ins.indexes.first == ins.indexes.second;
    }
};

// Halves of a removed segment, reinserted in the reduced path.
TEST_F(pair_insertion_fixture, PairInsertionOfSplitSegmentIsOptimal) {
    int checked = 0, paired = 0;
    for (const auto& p : inputs()) {
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            if (g.cost(p[i]).length < 1) continue;
            auto [first, second] = g.splitSegment(p[i], g.cost(p[i]).length / 2);
            if (!path_engine::is_node(second.source)) continue;
            auto reduced = p; reduced.erase(reduced.begin() + i);
            paired += checkPair(reduced, first, second);
            ++checked;
        }
    }
    EXPECT_GT(checked, 0);
    RecordProperty("paired_optima", std::to_string(paired) + "/" + std::to_string(checked));
}

// Unrelated segments: mostly disjoint optima, horizontal and vertical.
TEST_F(pair_insertion_fixture, PairInsertionOfUnrelatedSegmentsIsOptimal) {
    const std::vector<std::pair<segment, segment>> pairs = {
        {{7, 10}, {48, 52}}, {{0, 4}, {72, 75}}, {{12, 13}, {56, 59}},
        {{20, path_engine::NULL_NODE}, {77, path_engine::NULL_NODE}}, {{7, 10}, {12, 13}}, {{48, 52}, {56, 59}},
    };
    int checked = 0, disjoint = 0;
    for (const auto& p : inputs())
        for (const auto& [a, b] : pairs) {
            disjoint += !checkPair(p, a, b);
            ++checked;
        }
    EXPECT_GT(disjoint, 0) << "no disjoint optimum exercised";
    RecordProperty("disjoint_optima", std::to_string(disjoint) + "/" + std::to_string(checked));
}

// Both segments have the same best index: the disjoint optimum uses one
// backup, the cheaper of best + backup and backup + best.
TEST_F(pair_insertion_fixture, SharedBestIndexUsesBackup) {
    int shared = 0;
    for (const auto& p : inputs())
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            // Two copies of the removed segment share every insertion index.
            auto reduced = p; reduced.erase(reduced.begin() + i);
            cppied_solution s = make(reduced);
            TopTwoInsertion a = MinimumTopTwoInsertion(P, c, g, s, p[i]).solve();
            TopTwoInsertion b = MinimumTopTwoInsertion(P, c, g, s, g.flip_segment(p[i])).solve();
            if (a.bestInsertion.index != b.bestInsertion.index) continue;
            ++shared;
            checkPair(reduced, p[i], g.flip_segment(p[i]));
        }
    EXPECT_GT(shared, 0);
}

// ----------------------------------------------------------------------------
// Reuse of one MinimumSegmentPairInsertion over several pairs (the instance
// MaximumOneTwoReallocation builds once and memoizes insertions in).
// ----------------------------------------------------------------------------

namespace {
std::vector<std::pair<segment, segment>> reusePairs(path_engine& g) {
    std::vector<std::pair<segment, segment>> pairs = {
        {{7, 10}, {48, 52}}, {{0, 4}, {72, 75}}, {{12, 13}, {56, 59}},
        {{20, path_engine::NULL_NODE}, {77, path_engine::NULL_NODE}}, {{7, 10}, {12, 13}},
    };
    auto [first, second] = g.splitSegment({30, 35}, 2);
    pairs.emplace_back(first, second);
    return pairs;
}
}  // namespace

// Every pair solved on a reused instance gives the result of a fresh one,
// whatever the pairs solved before (cheaper or not).
TEST_F(pair_insertion_fixture, ReusedInstanceMatchesFreshInstance) {
    const auto pairs = reusePairs(g);
    for (const auto& p : inputs()) {
        cppied_solution s = make(p);
        MinimumSegmentPairInsertion reused(P, c, g, s, cost_t{MAX_INT, MAX_INT});
        for (int order = 0; order < 2; ++order)
            for (std::size_t k = 0; k < pairs.size(); ++k) {
                auto [a, b] = pairs[order == 0 ? k : pairs.size() - 1 - k];
                SCOPED_TRACE("insert " + segStr(a) + " + " + segStr(b));
                reused.setSegmentPair(a, b);
                Insertion r = reused.solve();
                MinimumSegmentPairInsertion fresh(P, c, g, s, cost_t{MAX_INT, MAX_INT});
                fresh.setSegmentPair(a, b);
                Insertion f = fresh.solve();
                EXPECT_EQ(r.cost, f.cost);
                EXPECT_EQ(r.indexes, f.indexes);
                EXPECT_EQ(r.segments, f.segments) << "returned segments of another pair";
                EXPECT_EQ(r.cost, bruteForce(p, a, b));
            }
    }
}

// A memoized pair solved again returns the same insertion.
TEST_F(pair_insertion_fixture, RepeatedPairReturnsMemoizedInsertion) {
    const auto pairs = reusePairs(g);
    for (const auto& p : inputs()) {
        cppied_solution s = make(p);
        MinimumSegmentPairInsertion alg(P, c, g, s, cost_t{MAX_INT, MAX_INT});
        std::vector<Insertion> first_pass;
        for (auto [a, b] : pairs) { alg.setSegmentPair(a, b); first_pass.push_back(alg.solve()); }
        for (std::size_t k = 0; k < pairs.size(); ++k) {
            auto [a, b] = pairs[k];
            SCOPED_TRACE("insert " + segStr(a) + " + " + segStr(b));
            alg.setSegmentPair(a, b);
            Insertion again = alg.solve();
            EXPECT_EQ(again.cost, first_pass[k].cost);
            EXPECT_EQ(again.indexes, first_pass[k].indexes);
            EXPECT_EQ(again.segments, first_pass[k].segments);
        }
    }
}

// The memo key is unordered: {a, b} and {b, a} share one insertion cost.
TEST_F(pair_insertion_fixture, SwappedPairSharesMemoizedCost) {
    const auto pairs = reusePairs(g);
    for (const auto& p : inputs()) {
        cppied_solution s = make(p);
        MinimumSegmentPairInsertion alg(P, c, g, s, cost_t{MAX_INT, MAX_INT});
        for (auto [a, b] : pairs) {
            SCOPED_TRACE("insert " + segStr(a) + " + " + segStr(b));
            alg.setSegmentPair(a, b);
            Insertion ab = alg.solve();
            alg.setSegmentPair(b, a);
            Insertion ba = alg.solve();
            EXPECT_EQ(ab.cost, ba.cost);
            EXPECT_EQ(ab.cost, bruteForce(p, a, b));
        }
    }
}

// ----------------------------------------------------------------------------
// Content of the caches of MinimumSegmentPairInsertion
//   memoizedDisjointTopInsertions: encodeSegment(s) -> top two insertions of s
//   memoizedMinimalInsertions:     {a, b} (unordered) -> cheapest insertion
// A poisoned entry shows the cache is read: solve() returns what it holds.
// ----------------------------------------------------------------------------

namespace {
struct pair_insertion_accessor : public MinimumSegmentPairInsertion {
    using MinimumSegmentPairInsertion::MinimumSegmentPairInsertion;
    using MinimumSegmentPairInsertion::memoizedMinimalInsertions;
    using MinimumSegmentPairInsertion::memoizedDisjointTopInsertions;
};
// The top-two cache is keyed on the non-oriented segment.
std::uint64_t topTwoKey(path_engine& g, const segment& s) {
    const segment u = g.getNonOriented(s);
    return encodeSegment(u.source, u.target);
}
}  // namespace

// After a solve, the top two insertions of both segments are cached and equal
// those of a fresh MinimumTopTwoInsertion.
TEST_F(pair_insertion_fixture, TopTwoCacheHoldsTopTwoInsertionsOfBothSegments) {
    for (const auto& p : inputs())
        for (auto [a, b] : reusePairs(g)) {
            SCOPED_TRACE("insert " + segStr(a) + " + " + segStr(b));
            cppied_solution s = make(p);
            pair_insertion_accessor alg(P, c, g, s, cost_t{MAX_INT, MAX_INT});
            alg.setSegmentPair(a, b);
            alg.solve();
            EXPECT_EQ(alg.memoizedDisjointTopInsertions.size(), 2u);
            for (const segment& x : {a, b}) {
                ASSERT_TRUE(alg.memoizedDisjointTopInsertions.contains(topTwoKey(g, x))) << segStr(x) << " not cached";
                const TopTwoInsertion& cached = alg.memoizedDisjointTopInsertions.at(topTwoKey(g, x));
                const TopTwoInsertion fresh = MinimumTopTwoInsertion(P, c, g, s, x).solve();
                EXPECT_EQ(cached.bestInsertion.insertion_cost, fresh.bestInsertion.insertion_cost);
                EXPECT_EQ(cached.bestInsertion.index, fresh.bestInsertion.index);
                EXPECT_EQ(cached.bestInsertion.s, fresh.bestInsertion.s);
                EXPECT_EQ(cached.backupInsertion.insertion_cost, fresh.backupInsertion.insertion_cost);
                EXPECT_EQ(cached.backupInsertion.index, fresh.backupInsertion.index);
                EXPECT_EQ(cached.backupInsertion.s, fresh.backupInsertion.s);
            }
        }
}

// After a solve, the pair cache holds the returned insertion: the optimum
// over disjoint and paired configurations.
TEST_F(pair_insertion_fixture, MinimalInsertionCacheHoldsOptimum) {
    for (const auto& p : inputs())
        for (auto [a, b] : reusePairs(g)) {
            SCOPED_TRACE("insert " + segStr(a) + " + " + segStr(b));
            cppied_solution s = make(p);
            pair_insertion_accessor alg(P, c, g, s, cost_t{MAX_INT, MAX_INT});
            alg.setSegmentPair(a, b);
            Insertion ins = alg.solve();
            ASSERT_EQ(alg.memoizedMinimalInsertions.size(), 1u);
            ASSERT_TRUE(alg.memoizedMinimalInsertions.contains(SegmentPair(a, b)));
            const Insertion& cached = alg.memoizedMinimalInsertions.at(SegmentPair(a, b));
            EXPECT_EQ(cached.cost, ins.cost);
            EXPECT_EQ(cached.cost, bruteForce(p, a, b)) << "cached insertion is not the optimum";
            // The cached plan is a real one (indexes may be unordered).
            auto first = std::make_pair(cached.segments.first, cached.indexes.first);
            auto second = std::make_pair(cached.segments.second, cached.indexes.second);
            if (first.second > second.second) std::swap(first, second);
            EXPECT_EQ(cached.cost, costOf(apply(p, {}, {first, second})) - costOf(p));
        }
}

// Pairs sharing segments: one top-two entry per distinct segment and one
// minimal insertion entry per distinct unordered pair.
TEST_F(pair_insertion_fixture, CacheSizesCountDistinctSegmentsAndPairs) {
    const auto pairs = reusePairs(g);
    for (const auto& p : inputs()) {
        cppied_solution s = make(p);
        pair_insertion_accessor alg(P, c, g, s, cost_t{MAX_INT, MAX_INT});
        std::set<std::uint64_t> segments;
        std::set<SegmentPair> unordered;
        for (int pass = 0; pass < 2; ++pass)
            for (auto [a, b] : pairs) {
                alg.setSegmentPair(a, b);
                alg.solve();
                alg.setSegmentPair(b, a);
                alg.solve();
                segments.insert(topTwoKey(g, a));
                segments.insert(topTwoKey(g, b));
                unordered.insert(SegmentPair(a, b));
            }
        EXPECT_EQ(alg.memoizedDisjointTopInsertions.size(), segments.size());
        EXPECT_EQ(alg.memoizedMinimalInsertions.size(), unordered.size());
    }
}

// Both orientations of a segment share one top-two entry: a pair using the
// flipped segment adds no entry for it.
TEST_F(pair_insertion_fixture, TopTwoCacheIsSharedByBothOrientations) {
    for (const auto& p : inputs())
        for (auto [a, b] : reusePairs(g)) {
            SCOPED_TRACE("insert " + segStr(a) + " + " + segStr(b));
            cppied_solution s = make(p);
            pair_insertion_accessor alg(P, c, g, s, cost_t{MAX_INT, MAX_INT});
            alg.setSegmentPair(a, b);
            alg.solve();
            const std::size_t entries = alg.memoizedDisjointTopInsertions.size();
            segment fa = g.flip_segment(a), fb = g.flip_segment(b);
            alg.setSegmentPair(fa, fb);
            alg.solve();
            EXPECT_EQ(alg.memoizedDisjointTopInsertions.size(), entries);
        }
}

// A segment already cached is not recomputed: poisoning its entry with
// insertions above any bound makes the next pair using it dominated.
TEST_F(pair_insertion_fixture, TopTwoCacheIsReadForKnownSegment) {
    auto pairs = reusePairs(g);
    auto [a, b] = pairs[0];
    segment other = pairs[1].second;
    for (const auto& p : inputs()) {
        cppied_solution s = make(p);
        pair_insertion_accessor alg(P, c, g, s, cost_t{MAX_INT, MAX_INT});
        alg.setSegmentPair(a, b);
        alg.solve();
        TopTwoInsertion& entry = alg.memoizedDisjointTopInsertions.at(topTwoKey(g, a));
        entry.bestInsertion.insertion_cost = {MAX_INT, MAX_INT};
        entry.backupInsertion.insertion_cost = {MAX_INT, MAX_INT};
        alg.setSegmentPair(a, other);
        EXPECT_THROW(alg.solve(), DominatedInsertion) << "the cached top two insertions of " << segStr(a)
                                                      << " were not used";
    }
}

// A pair already cached is not recomputed: solve() returns the cached
// insertion, for both orders of the pair.
TEST_F(pair_insertion_fixture, MinimalInsertionCacheIsReadForKnownPair) {
    for (const auto& p : inputs())
        for (auto [a, b] : reusePairs(g)) {
            SCOPED_TRACE("insert " + segStr(a) + " + " + segStr(b));
            cppied_solution s = make(p);
            pair_insertion_accessor alg(P, c, g, s, cost_t{MAX_INT, MAX_INT});
            alg.setSegmentPair(a, b);
            alg.solve();
            const Insertion marker{{1, 1}, {-7, -7}, {a, b}};
            alg.memoizedMinimalInsertions.at(SegmentPair(a, b)) = marker;
            for (int order = 0; order < 2; ++order) {
                order == 0 ? alg.setSegmentPair(a, b) : alg.setSegmentPair(b, a);
                Insertion hit = alg.solve();
                EXPECT_EQ(hit.cost, marker.cost) << "the cached insertion was not used (order " << order << ")";
                EXPECT_EQ(hit.indexes, marker.indexes);
            }
        }
}

// The bound is the one of the caller: RandomOneTwoReallocation inserts
// without a bound, so none of its paired candidates is dominated.
TEST_F(pair_insertion_fixture, RandomOneTwoPairedCandidatesAreNeverDominated) {
    int paired = 0;
    for (const auto& p : inputs())
        for (int i = 1; i < static_cast<int>(p.size()); ++i)
            for (int d = 0; d < 5; ++d) {
                cppied_solution s = make(p);
                OneTwoReallocationMove mv = RandomOneTwoReallocation(P, c, g, s, i).solve();
                if (!path_engine::is_node(mv.replacements[1].source)) continue;
                ++paired;
                EXPECT_GT(mv.gain, (cost_t{MIN_INT, MIN_INT}))
                    << "dominated paired move drawn for path[" << i << "] = " << segStr(p[i]);
            }
    EXPECT_GT(paired, 0);
}

// Two segment path: the disjoint configurations use indexes 1 and 2 only.
TEST_F(pair_insertion_fixture, PairInsertionIntoTwoSegmentPath) {
    const std::vector<std::vector<segment>> paths = {{{6, -1}, {23, 18}}, {{6, 11}, {44, 42}}};
    for (const auto& p : paths) {
        checkPair(p, {7, 10}, {48, 52});
        checkPair(p, {0, 4}, {72, 75});
        auto [first, second] = g.splitSegment({30, 35}, 2);
        checkPair(p, first, second);
    }
}

TEST_F(move_algorithms_fixture, PairInsertionThrowsWhenDominated) {
    const auto p = inputs()[0];
    auto reduced = p; reduced.erase(reduced.begin() + 4);
    cppied_solution s = make(reduced);
    auto [first, second] = g.splitSegment(p[4], 2);
    MinimumSegmentPairInsertion pair(P, c, g, s, cost_t{0, 0});
    pair.setSegmentPair(first, second);
    EXPECT_THROW(pair.solve(), DominatedInsertion);
}

// Edge case: the path holds only the start segment, so the only insertion
// position is 1 (append). solveUnitPathEdgeCase handles it: both segments
// appended consecutively, in the cheapest of the 8 orders/orientations, with
// the paired-insertion convention {k, k} ("both before path[k]", here k = 1)
// used by the callers (nested inserts the second segment right after the
// first: at index k + 1).
// The returned cost is the real cost of the insertion, and it never throws:
// there is no alternative position to dominate it.
TEST_F(move_algorithms_fixture, PairInsertionIntoStartOnlyPath) {
    const std::vector<segment> start = {{6, path_engine::NULL_NODE}};
    cppied_solution s = make(start);
    const cppied_solution before = s;
    const segment a = {23, 18};

    auto [first, second] = g.splitSegment(a, 2);
    MinimumSegmentPairInsertion pair(P, c, g, s, cost_t{MAX_INT, MAX_INT});
    pair.setSegmentPair(first, second);
    Insertion pr = pair.solve();
    expectUnchanged(before, s, "MinimumSegmentPairInsertion (unit path, pair)");
    ASSERT_TRUE(path_engine::is_node(pr.segments.first.source));
    ASSERT_TRUE(path_engine::is_node(pr.segments.second.source));
    EXPECT_EQ(pr.indexes.first, 1);
    EXPECT_EQ(pr.indexes.second, 1) << "paired insertion uses {k, k}: both before path[k]";
    const std::vector<segment> result = {start[0], pr.segments.first, pr.segments.second};
    EXPECT_EQ(pr.cost, costOf(result) - costOf(start)) << "predicted cost differs from the real one";
    cost_t best = {MAX_INT, MAX_INT};
    for (const segment& x : {first, g.flip_segment(first)})
        for (const segment& y : {second, g.flip_segment(second)}) {
            best = std::min(best, costOf({start[0], x, y}) - costOf(start));
            best = std::min(best, costOf({start[0], y, x}) - costOf(start));
        }
    EXPECT_EQ(pr.cost, best) << "not the cheapest order/orientation";
}

// ----------------------------------------------------------------------------
// MaximumOneTwoReallocation (n12), RandomOneTwoReallocation (nested)
// ----------------------------------------------------------------------------

class reallocation_fixture : public move_algorithms_fixture {
protected:
    // Apply a 1-2 move: indexes refer to the full path (path[i] present).
    std::vector<segment> applyOneTwo(const std::vector<segment>& p, int i, const OneTwoReallocationMove& mv) {
        return apply(p, {i}, {{mv.replacements[0], mv.reallocationIndexes[0]},
                              {mv.replacements[1], mv.reallocationIndexes[1]}});
    }
    // minGain: the smallest gain the caller applies (n12 applies improving
    // moves only; nested applies any non-dominated random move).
    void checkOneTwo(const std::vector<segment>& p, int i, const OneTwoReallocationMove& mv, const char* who,
                     cost_t minGain) {
        SCOPED_TRACE(std::string(who) + " path[" + std::to_string(i) + "] = " + segStr(p[i]));
        if (!path_engine::is_node(mv.replacements[0].source)) {
            if (mv.gain > cost_t{MIN_INT, MIN_INT} && !(mv.gain < minGain)) {
                auto q = p; q.erase(q.begin() + i);
                EXPECT_TRUE(feasible(q)) << "direct removal breaks coverage";
                EXPECT_EQ(mv.gain, costOf(p) - costOf(q));
            }
            return;
        }
        if (!(mv.gain > cost_t{MIN_INT, MIN_INT}) || mv.gain < minGain) return;   // not applied by the caller
        for (int k : mv.reallocationIndexes)
            if (k >= 0) {
                EXPECT_GE(k, 1) << "insertion before the start segment";
                EXPECT_LE(k, static_cast<int>(p.size()));
            }
        auto q = applyOneTwo(p, i, mv);
        EXPECT_EQ(q[0], p[0]) << "start segment displaced";
        EXPECT_TRUE(feasible(q)) << "move " << segStr(mv.replacements[0]) << " + " << segStr(mv.replacements[1])
                                 << " breaks coverage";
        EXPECT_EQ(mv.gain, costOf(p) - costOf(q)) << "predicted gain differs from the real one ("
                                                  << segStr(mv.replacements[0]) << "@" << mv.reallocationIndexes[0] << ", "
                                                  << segStr(mv.replacements[1]) << "@" << mv.reallocationIndexes[1] << ")";
    }
};

TEST_F(reallocation_fixture, MaximumOneTwoReallocationContract) {
    for (const auto& p : inputs())
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            cppied_solution s = make(p);
            const cppied_solution before = s;
            OneTwoReallocationMove mv = MaximumOneTwoReallocation(P, c, g, s, i).solve();
            expectUnchanged(before, s, "MaximumOneTwoReallocation");
            checkOneTwo(p, i, mv, "MaximumOneTwoReallocation", cost_t{0, 1});
        }
}

// A single replacement (second replacement NULL) is routed to
// MinimumSegmentInsertion: no second index, and its gain is the removal gain
// minus the cheapest single insertion of the replacement in the reduced path.
// Like a paired replacement, it is dominated (gain MIN) when that insertion
// costs at least the bound, which is the removal gain for n12 and unbounded
// for nested (RandomOneTwoReallocation).
class unitary_reallocation_fixture : public reallocation_fixture {
protected:
    template<typename Algorithm>
    // bound: the insertion cost bound; none means the removal gain of path[i].
    void checkUnitary(std::optional<cost_t> bound, const char* who, int& unitary, int& dominated) {
        for (const auto& p : inputs())
            for (int i = 1; i < static_cast<int>(p.size()); ++i) {
                cppied_solution s = make(p);
                OneTwoReallocationMove mv = Algorithm(P, c, g, s, i).solve();
                if (!path_engine::is_node(mv.replacements[0].source) || path_engine::is_node(mv.replacements[1].source))
                    continue;
                ++unitary;
                SCOPED_TRACE(std::string(who) + " path[" + std::to_string(i) + "] = " + segStr(p[i]) +
                             " -> " + segStr(mv.replacements[0]));
                EXPECT_EQ(mv.replacements[1], kNull);
                EXPECT_EQ(mv.reallocationIndexes[1], -1) << "single replacement with a second index";
                auto reduced = p; reduced.erase(reduced.begin() + i);
                cppied_solution r = make(reduced);
                SegmentInsertion ins = MinimumSegmentInsertion(P, c, g, r, mv.replacements[0]).solve();
                const cost_t removal = costOf(p) - costOf(reduced);
                if (ins.insertion_cost >= bound.value_or(removal)) {
                    ++dominated;
                    EXPECT_EQ(mv.gain, (cost_t{MIN_INT, MIN_INT})) << "insertion above the bound not dominated";
                } else
                    EXPECT_EQ(mv.gain, removal - ins.insertion_cost);
            }
    }
};

TEST_F(unitary_reallocation_fixture, MaximumOneTwoUnitaryMoveUsesMinimumInsertion) {
    int unitary = 0, dominated = 0;
    checkUnitary<MaximumOneTwoReallocation>(std::nullopt, "MaximumOneTwoReallocation", unitary, dominated);
    EXPECT_GT(unitary, 0) << "no single replacement move exercised";
    RecordProperty("dominated_unitary", std::to_string(dominated) + "/" + std::to_string(unitary));
}

TEST_F(unitary_reallocation_fixture, RandomOneTwoUnitaryMoveIsNeverDominated) {
    int unitary = 0, dominated = 0;
    for (int d = 0; d < 5; ++d)
        checkUnitary<RandomOneTwoReallocation>(cost_t{MAX_INT, MAX_INT}, "RandomOneTwoReallocation", unitary, dominated);
    EXPECT_GT(unitary, 0) << "no single replacement move exercised";
    EXPECT_EQ(dominated, 0);
}

// With the bound at the removal gain, a single replacement returned by n12 is
// either dominated (gain MIN) or strictly improving.
TEST_F(reallocation_fixture, MaximumOneTwoUnitaryMoveIsDominatedOrImproving) {
    for (const auto& p : inputs())
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            cppied_solution s = make(p);
            OneTwoReallocationMove mv = MaximumOneTwoReallocation(P, c, g, s, i).solve();
            if (!path_engine::is_node(mv.replacements[0].source)) continue;
            SCOPED_TRACE("path[" + std::to_string(i) + "] = " + segStr(p[i]));
            // A move that is not dominated must strictly beat the bound: positive gain.
            if (mv.gain > cost_t{MIN_INT, MIN_INT} && !path_engine::is_node(mv.replacements[1].source))
                EXPECT_GT(mv.gain, (cost_t{0, 0})) << "non-dominated unitary move with non-positive gain";
        }
}

TEST_F(reallocation_fixture, RandomOneTwoReallocationContract) {
    int draws = 0, dominated = 0, dominatedWithAlternative = 0;
    for (const auto& p : inputs())
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            cppied_solution s0 = make(p);
            const OneTwoReallocationMove best = MaximumOneTwoReallocation(P, c, g, s0, i).solve();
            for (int d = 0; d < 10; ++d) {
                cppied_solution s = make(p);
                const cppied_solution before = s;
                OneTwoReallocationMove mv = RandomOneTwoReallocation(P, c, g, s, i).solve();
                expectUnchanged(before, s, "RandomOneTwoReallocation");
                checkOneTwo(p, i, mv, "RandomOneTwoReallocation", cost_t{MIN_INT + 1, MIN_INT});
                ++draws;
                if (path_engine::is_node(mv.replacements[0].source) && !(mv.gain > cost_t{MIN_INT, MIN_INT})) {
                    ++dominated;
                    dominatedWithAlternative += best.gain > cost_t{MIN_INT, MIN_INT};
                }
            }
        }
    // A random move is drawn from the scored candidates; a dominated
    // candidate (no insertion within the bound) carries no usable move and
    // must not be drawn when a scored one exists.
    EXPECT_EQ(dominatedWithAlternative, 0) << dominated << " of " << draws << " draws returned a dominated move";
}

// ----------------------------------------------------------------------------
// MaximumTwoOneReallocation (n21)
// ----------------------------------------------------------------------------

TEST_F(reallocation_fixture, MaximumTwoOneReallocationContract) {
    for (const auto& p : inputs())
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            cppied_solution s = make(p);
            const cppied_solution before = s;
            TwoOneReallocationMove mv = MaximumTwoOneReallocation(P, c, g, s, i).solve();
            SCOPED_TRACE("path[" + std::to_string(i) + "] = " + segStr(p[i]));
            expectUnchanged(before, s, "MaximumTwoOneReallocation");
            // A paired removal without allocation is "no move": gain MIN.
            if (mv.removalIndex >= 0 && !path_engine::is_node(mv.replacement.source))
                EXPECT_EQ(mv.gain, (cost_t{MIN_INT, MIN_INT})) << "paired removal without allocation must be no move";
            // n21 applies improving moves only.
            if (!(mv.gain > cost_t{0, 0})) continue;
            if (mv.removalIndex < 0) {
                EXPECT_FALSE(path_engine::is_node(mv.replacement.source)) << "direct removal with a replacement";
                auto q = p; q.erase(q.begin() + i);
                EXPECT_TRUE(feasible(q)) << "direct removal breaks coverage";
                EXPECT_EQ(mv.gain, costOf(p) - costOf(q));
                continue;
            }
            EXPECT_NE(mv.removalIndex, 0) << "start segment removed";
            EXPECT_NE(mv.removalIndex, i);
            ASSERT_LT(mv.removalIndex, static_cast<int>(p.size()));
            if (path_engine::is_node(mv.replacement.source)) {
                EXPECT_GE(mv.reallocationIndex, 1);
                EXPECT_LE(mv.reallocationIndex, static_cast<int>(p.size()));
            }
            auto q = apply(p, {i, mv.removalIndex}, {{mv.replacement, mv.reallocationIndex}});
            EXPECT_EQ(q[0], p[0]) << "start segment displaced";
            EXPECT_TRUE(feasible(q)) << "move breaks coverage (removing " << segStr(p[mv.removalIndex])
                                     << ", inserting " << segStr(mv.replacement) << "@" << mv.reallocationIndex << ")";
            EXPECT_EQ(mv.gain, costOf(p) - costOf(q)) << "predicted gain differs from the real one";
        }
}
