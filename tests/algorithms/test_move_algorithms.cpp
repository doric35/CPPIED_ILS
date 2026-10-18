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
// ----------------------------------------------------------------------------

// Unitary insertion (second segment NULL) is exact: cheapest position and
// orientation of the segment.
TEST_F(move_algorithms_fixture, PairInsertionUnitaryIsCheapest) {
    for (const auto& p : inputs()) {
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            auto reduced = p; reduced.erase(reduced.begin() + i);
            cppied_solution s = make(reduced);
            const cppied_solution before = s;
            MinimumSegmentPairInsertion alg(P, c, g, s, p[i], kNull, cost_t{MAX_INT, MAX_INT});
            Insertion ins = alg.solve();
            SCOPED_TRACE("insert " + segStr(p[i]));
            expectUnchanged(before, s, "MinimumSegmentPairInsertion");
            cost_t best = {MAX_INT, MAX_INT};
            for (const segment& r : {p[i], g.flip_segment(p[i])})
                for (int k = 1; k <= static_cast<int>(reduced.size()); ++k)
                    best = std::min(best, costOf(apply(reduced, {}, {{r, k}})) - costOf(reduced));
            ASSERT_FALSE(path_engine::is_node(ins.segments.second.source));
            auto q = apply(reduced, {}, {{ins.segments.first, ins.indexes.first}});
            EXPECT_EQ(ins.cost, costOf(q) - costOf(reduced)) << "predicted cost differs from the real one";
            EXPECT_EQ(ins.cost, best) << "not the cheapest insertion";
        }
    }
}

// Pair insertion: the returned plan is consistent (ordered indexes, cost
// equal to the real cost of inserting both) and never cheaper than the brute
// force optimum over all positions, orders and orientations.
TEST_F(move_algorithms_fixture, PairInsertionCostIsRealAndBounded) {
    int checked = 0, optimal = 0;
    for (const auto& p : inputs()) {
        for (int i = 1; i < static_cast<int>(p.size()); ++i) {
            if (g.cost(p[i]).length < 1) continue;
            auto [first, second] = g.splitSegment(p[i], g.cost(p[i]).length / 2);
            if (!path_engine::is_node(second.source)) continue;
            auto reduced = p; reduced.erase(reduced.begin() + i);
            cppied_solution s = make(reduced);
            const cppied_solution before = s;
            MinimumSegmentPairInsertion alg(P, c, g, s, first, second, cost_t{MAX_INT, MAX_INT});
            Insertion ins = alg.solve();
            SCOPED_TRACE("insert " + segStr(first) + " + " + segStr(second));
            expectUnchanged(before, s, "MinimumSegmentPairInsertion");
            ASSERT_TRUE(path_engine::is_node(ins.segments.first.source));
            ASSERT_TRUE(path_engine::is_node(ins.segments.second.source));
            EXPECT_LE(ins.indexes.first, ins.indexes.second) << "unordered indexes";
            auto q = apply(reduced, {}, {{ins.segments.first, ins.indexes.first}, {ins.segments.second, ins.indexes.second}});
            EXPECT_EQ(ins.cost, costOf(q) - costOf(reduced)) << "predicted cost differs from the real one";
            cost_t best = {MAX_INT, MAX_INT};
            const int n = static_cast<int>(reduced.size());
            for (const segment& a : {first, g.flip_segment(first)})
                for (const segment& b : {second, g.flip_segment(second)})
                    for (int x = 1; x <= n; ++x)
                        for (int y = 1; y <= n; ++y) {
                            best = std::min(best, costOf(apply(reduced, {}, {{a, x}, {b, y}})) - costOf(reduced));
                            best = std::min(best, costOf(apply(reduced, {}, {{b, x}, {a, y}})) - costOf(reduced));
                        }
            EXPECT_GE(ins.cost, best) << "cost below the optimum (oracle mismatch)";
            optimal += ins.cost == best;
            ++checked;
        }
    }
    EXPECT_GT(checked, 0);
    RecordProperty("optimal_pair_insertions", std::to_string(optimal) + "/" + std::to_string(checked));
}

TEST_F(move_algorithms_fixture, PairInsertionThrowsWhenDominated) {
    const auto p = inputs()[0];
    auto reduced = p; reduced.erase(reduced.begin() + 4);
    cppied_solution s = make(reduced);
    MinimumSegmentPairInsertion unitary(P, c, g, s, p[4], kNull, cost_t{0, 0});
    EXPECT_THROW(unitary.solve(), DominatedInsertion);
    auto [first, second] = g.splitSegment(p[4], 2);
    MinimumSegmentPairInsertion pair(P, c, g, s, first, second, cost_t{0, 0});
    EXPECT_THROW(pair.solve(), DominatedInsertion);
}

// Edge case: the path holds only the start segment, so the only insertion
// position is 1 (append). solveUnitPathEdgeCase handles it:
//   unitary: the cheaper orientation, appended, indexes {1, -1};
//   pair: both segments appended consecutively, in the cheapest of the 8
//   orders/orientations, with the paired-insertion convention {k, k}
//   ("both before path[k]", here k = 1) used by the callers (nested inserts
//   the second segment right after the first: at index k + 1).
// The returned cost is the real cost of the insertion, and it never throws:
// there is no alternative position to dominate it.
TEST_F(move_algorithms_fixture, PairInsertionIntoStartOnlyPath) {
    const std::vector<segment> start = {{6, path_engine::NULL_NODE}};
    cppied_solution s = make(start);
    const cppied_solution before = s;
    const segment a = {23, 18};

    MinimumSegmentPairInsertion unitary(P, c, g, s, a, kNull, cost_t{MAX_INT, MAX_INT});
    Insertion u = unitary.solve();
    expectUnchanged(before, s, "MinimumSegmentPairInsertion (unit path, unitary)");
    EXPECT_EQ(u.indexes.first, 1);
    EXPECT_FALSE(path_engine::is_node(u.segments.second.source));
    const cost_t forward = costOf(apply(start, {}, {{a, 1}})) - costOf(start);
    const cost_t reversed = costOf(apply(start, {}, {{g.flip_segment(a), 1}})) - costOf(start);
    EXPECT_EQ(u.cost, std::min(forward, reversed));
    EXPECT_EQ(u.cost, costOf(apply(start, {}, {{u.segments.first, u.indexes.first}})) - costOf(start));

    auto [first, second] = g.splitSegment(a, 2);
    MinimumSegmentPairInsertion pair(P, c, g, s, first, second, cost_t{MAX_INT, MAX_INT});
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
