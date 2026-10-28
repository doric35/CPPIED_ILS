#include "../test_fixture.hpp"
#include "../../include/construction/dp_sweeper.hpp"
#include "../../include/neighborhoods/neighborhood_n1.hpp"
#include "../../include/neighborhoods/neighborhood_n12.hpp"
#include "../../include/neighborhoods/neighborhood_n21.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_cut.hpp"
#include "../../include/neighborhoods/neighborhood_reduce.hpp"

// ============================================================================
// Unit tests of the SimpleNeighborhood<Move, Algorithm> machinery and of the
// move application of each simple neighborhood.
// ============================================================================

template<typename N>
struct units_accessor : public N {
    using N::N;
    using N::candidatesMoves;
    using N::splitIndices;
    using N::spacingConstraint;
    using N::clear;
    using N::setMovesAndGains;
    using N::selectMoves;
    using N::applyMoves;
    using N::pruneNonImprovingMoves;
    using N::splitOddEvenMoves;
    using N::splitHorizontalVerticalMoves;
    using N::constructIntervalSets;
    using N::transformToInterval;
    using N::geometry;
    using N::coverage;
};

using trim_acc = units_accessor<neighborhood_trim>;

static std::string segStr(const segment& s) {
    return "{" + std::to_string(s.source) + "," + std::to_string(s.target) + "}";
}

class simple_units_fixture : public cppied_context_fixture {
protected:
    cppied_solution sol;
    void SetUp() override {
        // Orientations (6x6: ids >= 42 are vertical): H, H, V, V, H, V, V
        sol.path = {{6, 11}, {23, 18}, {44, 42}, {48, 51}, {30, 35}, {75, 73}, {62, 63}};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        path_engine g(P);
        coverage_engine c(P);
        c.reset(sol);
        sol.cost = g.cost(sol);
    }
    bool feasible(coverage_engine& c, const cppied_solution& s) {
        cppied_solution f = s; c.reset(f);
        return (f.coverage.array() >= P.req.array()).all();
    }
};

// ----------------------------------------------------------------------------
// Candidate pruning and grouping
// ----------------------------------------------------------------------------

TEST_F(simple_units_fixture, PruneKeepsOnlyStrictlyImprovingMoves) {
    trim_acc m(ctx, P, sol);
    m.candidatesMoves.resize(6);
    m.candidatesMoves[0].gain = {MIN_INT, MIN_INT};
    m.candidatesMoves[1].gain = {0, 0};
    m.candidatesMoves[2].gain = {0, 1};
    m.candidatesMoves[3].gain = {-1, 5};
    m.candidatesMoves[4].gain = {1, -3};
    m.candidatesMoves[5].gain = {0, -1};
    std::vector<int> candidates = {0, 1, 2, 3, 4, 5};
    m.pruneNonImprovingMoves(candidates);
    EXPECT_EQ(candidates, (std::vector<int>{2, 4}));
}

TEST_F(simple_units_fixture, SplitOddEvenGroupsByParityStably) {
    trim_acc m(ctx, P, sol);
    std::vector<int> candidates = {1, 2, 3, 4, 5, 6};
    m.splitOddEvenMoves(candidates);
    EXPECT_EQ(candidates, (std::vector<int>{2, 4, 6, 1, 3, 5}));
    EXPECT_EQ(m.splitIndices, (std::vector<int>{3}));
}

// Every group delimited by splitIndices holds a single orientation, the
// split points are non-decreasing and inside the candidate list, and no
// candidate is lost or duplicated.
static void expectHomogeneousGroups(trim_acc& m, const std::vector<int>& before, const std::vector<int>& after,
                                    const cppied_solution& s) {
    std::vector<int> a = before, b = after;
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    EXPECT_EQ(a, b) << "candidates lost or duplicated";
    int begin = 0;
    std::vector<int> bounds = m.splitIndices;
    bounds.push_back(static_cast<int>(after.size()));
    for (int end : bounds) {
        ASSERT_LE(begin, end) << "split indices decrease";
        ASSERT_LE(end, static_cast<int>(after.size())) << "split index past the candidates";
        for (int k = begin + 1; k < end; ++k)
            EXPECT_EQ(m.geometry.is_horizontal(s.path[after[k]]), m.geometry.is_horizontal(s.path[after[begin]]))
                << "group [" << begin << "," << end << ") mixes orientations";
        begin = end;
    }
}

TEST_F(simple_units_fixture, SplitHorizontalVerticalWithoutPriorSplit) {
    trim_acc m(ctx, P, sol);
    std::vector<int> candidates = {1, 2, 3, 4, 5, 6}, before = candidates;
    m.splitHorizontalVerticalMoves(candidates);
    ASSERT_EQ(m.splitIndices.size(), 1u);
    expectHomogeneousGroups(m, before, candidates, sol);
    // Horizontal group ({23,18}, {30,35}) first, stable order.
    EXPECT_EQ(candidates, (std::vector<int>{1, 4, 2, 3, 5, 6}));
}

TEST_F(simple_units_fixture, SplitHorizontalVerticalAfterOddEvenSplit) {
    trim_acc m(ctx, P, sol);
    std::vector<int> candidates = {1, 2, 3, 4, 5, 6}, before = candidates;
    m.splitOddEvenMoves(candidates);
    m.splitHorizontalVerticalMoves(candidates);
    EXPECT_EQ(m.splitIndices.size(), 3u) << "odd/even x horizontal/vertical = 4 groups";
    expectHomogeneousGroups(m, before, candidates, sol);
}

TEST_F(simple_units_fixture, SplitHorizontalVerticalEdgeCases) {
    trim_acc m(ctx, P, sol);
    for (std::vector<int> candidates : {std::vector<int>{}, std::vector<int>{1, 3, 5}, std::vector<int>{2, 4, 6},
                                        std::vector<int>{4}}) {
        m.splitIndices.clear();
        auto before = candidates;
        m.splitHorizontalVerticalMoves(candidates);
        expectHomogeneousGroups(m, before, candidates, sol);
        m.splitIndices.clear();
        candidates = before;
        m.splitOddEvenMoves(candidates);
        m.splitHorizontalVerticalMoves(candidates);
        expectHomogeneousGroups(m, before, candidates, sol);
    }
}

TEST_F(simple_units_fixture, ConstructIntervalSetsFollowsSplits) {
    trim_acc m(ctx, P, sol);
    m.candidatesMoves.resize(sol.path.size());
    for (auto& mv : m.candidatesMoves) mv.gain = {1, 0};
    std::vector<int> candidates = {1, 2, 3, 4, 5, 6};
    m.splitOddEvenMoves(candidates);
    m.splitHorizontalVerticalMoves(candidates);
    GroupedIntervalSets grouped = m.constructIntervalSets(candidates);
    ASSERT_EQ(grouped.size(), m.splitIndices.size() + 1);
    int begin = 0, g = 0;
    std::vector<int> bounds = m.splitIndices;
    bounds.push_back(static_cast<int>(candidates.size()));
    for (int end : bounds) {
        std::vector<int> ids;
        for (const auto& iv : grouped[g]) ids.push_back(iv.id);
        std::vector<int> expected(candidates.begin() + begin, candidates.begin() + end);
        std::sort(ids.begin(), ids.end()); std::sort(expected.begin(), expected.end());
        EXPECT_EQ(ids, expected) << "group " << g;
        begin = end; ++g;
    }
}

TEST_F(simple_units_fixture, ClearResetsState) {
    trim_acc m(ctx, P, sol);
    m.setMovesAndGains();
    std::vector<int> c = {1, 2, 3};
    m.splitOddEvenMoves(c);
    m.clear();
    EXPECT_TRUE(m.candidatesMoves.empty());
    EXPECT_TRUE(m.splitIndices.empty());
}

// setMovesAndGains: one move per segment, the start segment never moves.
TEST_F(simple_units_fixture, SetMovesAndGainsProtectsStartSegment) {
    trim_acc m(ctx, P, sol);
    m.setMovesAndGains();
    ASSERT_EQ(m.candidatesMoves.size(), sol.path.size());
    EXPECT_EQ(m.candidatesMoves[0].gain, (cost_t{MIN_INT, MIN_INT}));
    std::vector<int> c(sol.path.size());
    std::iota(c.begin(), c.end(), 0);
    m.pruneNonImprovingMoves(c);
    EXPECT_EQ(std::count(c.begin(), c.end(), 0), 0);
}

// local_search isolates the start vertex when path[0] spans several nodes,
// keeping the same start and the same geometric cost.
TEST_F(simple_units_fixture, LocalSearchIsolatesStartVertex) {
    ASSERT_TRUE(path_engine::is_node(sol.path[0].target));
    const int start = sol.path[0].source;
    trim_acc m(ctx, P, sol);
    m.local_search();
    EXPECT_EQ(sol.path[0].source, start);
    EXPECT_FALSE(path_engine::is_node(sol.path[0].target)) << "start segment " << segStr(sol.path[0]);
    EXPECT_EQ(sol.cost, m.geometry.cost(sol));
}

// ----------------------------------------------------------------------------
// Neighborhood-specific move application with hand-made moves
// ----------------------------------------------------------------------------

using reduce_acc = units_accessor<neighborhood_reduce>;
using n1_acc = units_accessor<neighborhood_n1>;
using cut_acc_base = units_accessor<neighborhood_cut>;
struct cut_acc : public cut_acc_base {
    using cut_acc_base::cut_acc_base;
    using neighborhood_cut::cutDiff;
    using neighborhood_cut::cutSelectedMoves;
};

TEST_F(simple_units_fixture, ReduceRemovesExactlySelectedSegments) {
    reduce_acc m(ctx, P, sol);
    m.candidatesMoves.resize(sol.path.size());
    std::vector<int> sel = {5, 2};
    auto expected = sol.path;
    expected.erase(expected.begin() + 5);
    expected.erase(expected.begin() + 2);
    m.applyMoves(sel);
    EXPECT_EQ(sol.path, expected);
    EXPECT_EQ(sol.cost, m.geometry.cost(sol));
    cppied_solution fresh = sol; m.coverage.reset(fresh);
    EXPECT_TRUE(fresh.coverage.isApprox(sol.coverage));
}

TEST_F(simple_units_fixture, TrimReplacesInPlaceAndDropsEmptiedSegments) {
    trim_acc m(ctx, P, sol);
    m.candidatesMoves.resize(sol.path.size());
    m.candidatesMoves[1] = {{22, 18}, {1, 0}};                                  // shrink {23,18}
    m.candidatesMoves[4] = {{path_engine::NULL_NODE, path_engine::NULL_NODE}, {6, 2}};  // drop {30,35}
    std::vector<int> sel = {4, 1};
    const cost_t before = sol.cost;
    auto expected = sol.path;
    expected[1] = {22, 18};
    expected.erase(expected.begin() + 4);
    m.applyMoves(sel);
    EXPECT_EQ(sol.path, expected);
    EXPECT_EQ(sol.cost, (before - cost_t{1, 0} - cost_t{6, 2})) << "tracked cost is decremented by the gains";
    cppied_solution fresh = sol; m.coverage.reset(fresh);
    EXPECT_TRUE(fresh.coverage.isApprox(sol.coverage));
}

TEST_F(simple_units_fixture, N1ReplacesInPlace) {
    n1_acc m(ctx, P, sol);
    m.candidatesMoves.resize(sol.path.size());
    m.candidatesMoves[3] = {{49, 51}, {1, 0}};
    std::vector<int> sel = {3};
    auto expected = sol.path;
    expected[3] = {49, 51};
    const cost_t before = sol.cost;
    m.applyMoves(sel);
    EXPECT_EQ(sol.path, expected);
    EXPECT_EQ(sol.cost, (before - cost_t{1, 0}));
    cppied_solution fresh = sol; m.coverage.reset(fresh);
    EXPECT_TRUE(fresh.coverage.isApprox(sol.coverage));
}

// cutDiff returns the parts of path[i] before and after the cut, oriented
// like path[i], in travel order: a non-empty piece comes first, and the first
// piece starts at path[i].source when it is not removed. Checked for every
// contiguous cut of horizontal and vertical segments in both orientations.
TEST_F(simple_units_fixture, CutDiffPartitionsSegmentForEveryCut) {
    const std::vector<segment> segments = {{6, 11}, {11, 6}, {42, 47}, {47, 42}};
    for (const segment& s : segments) {
        cut_acc m(ctx, P, sol);
        const int i = 1;
        sol.path[i] = s;
        const segment u = m.geometry.getNonOriented(s);
        const direction d = m.geometry.get_direction(s);
        for (int a = u.source; a <= u.target; ++a) {
            for (int b = a; b <= u.target; ++b) {
                segment cut = a == b ? segment{a, path_engine::NULL_NODE} : segment{a, b};
                m.geometry.correct_segment_direction(cut, d);
                auto pieces = m.cutDiff(cut, i);
                SCOPED_TRACE("segment " + segStr(s) + " cut " + segStr(cut));
                std::set<int> left, right;
                for (int v = u.source; v < a; ++v) left.insert(v);
                for (int v = b + 1; v <= u.target; ++v) right.insert(v);
                auto vertices = [&](const segment& p){
                    std::set<int> vs;
                    if (!path_engine::is_node(p.source)) return vs;
                    segment q = m.geometry.getNonOriented(p);
                    int last = path_engine::is_node(q.target) ? q.target : q.source;
                    for (int v = q.source; v <= last; ++v) vs.insert(v);
                    return vs;
                };
                // Non-empty pieces in travel order (pushSegments skips the
                // empty ones): the piece next to s.source comes first.
                const bool forward = u.source == s.source;
                std::vector<std::set<int>> expected;
                for (const auto& part : forward ? std::vector<std::set<int>>{left, right}
                                                : std::vector<std::set<int>>{right, left})
                    if (!part.empty()) expected.push_back(part);
                std::vector<std::set<int>> actual;
                for (const segment& p : pieces)
                    if (path_engine::is_node(p.source)) actual.push_back(vertices(p));
                EXPECT_EQ(actual, expected);
                for (const segment& p : pieces)
                    if (path_engine::is_node(p.source))
                        EXPECT_EQ(m.geometry.get_direction(p), d);
            }
        }
        sol.path[i] = {23, 18};
    }
}

// Applying a middle cut on a reversed segment keeps the remaining pieces in
// travel order: the piece containing the segment's source comes first.
TEST_F(simple_units_fixture, CutKeepsTravelOrderForReversedSegment) {
    sol.path[1] = {11, 6};   // reversed (W) segment as path[1]
    cut_acc m(ctx, P, sol);
    m.candidatesMoves.resize(sol.path.size());
    segment cut = {9, 8};
    m.candidatesMoves[1] = {cut, {2, 0}};
    std::vector<int> sel = {1};
    m.cutSelectedMoves(sel);
    ASSERT_GE(sol.path.size(), 3u);
    EXPECT_EQ(sol.path[1].source, 11) << "first piece " << segStr(sol.path[1]) << " should start at the segment source";
    EXPECT_EQ(sol.path[2], (segment{7, 6})) << "second piece " << segStr(sol.path[2]);
}

// ----------------------------------------------------------------------------
// Base neighborhood candidate generation
// ----------------------------------------------------------------------------

// "Complete coverage" candidates: every generated segment, inserted alone,
// covers every cell of the box, and every line whose coverage spans the box
// is generated. A line at row k covers cell rows [k-R, k+R-1] (column c
// covers cell columns [c-R, c+R-1]), so the rows are [lr.y-R, ul.y+R] and the
// columns [lr.x-R, ul.x+R], clamped to the grid; boxes taller (wider) than
// 2R have no horizontal (vertical) candidate.
TEST_F(simple_units_fixture, CompleteCoverageSegmentsCoverTheWholeBox) {
    trim_acc m(ctx, P, sol);
    const int R = P.max_range, rows = m.geometry.n_rows, cols = m.geometry.n_cols;
    auto box = [](int x0, int y0, int x1, int y1){ iRectangle b; b.ul = {x0, y0}; b.lr = {x1, y1}; return b; };
    std::vector<iRectangle> boxes = {box(0, 0, 1, 1),                    // single top-left cell
                                     box(cols - 1, rows - 1, cols, rows),// single bottom-right cell
                                     box(2, 3, 4, 5),                    // 2x2 interior (= 2R)
                                     box(1, 2, 4, 5),                    // 3x3: too tall/wide for R = 1
                                     box(0, 0, cols, 1),                 // whole top row
                                     box(0, 0, 1, rows)};                // whole left column
    auto coveredCells = [&](const segment& s){
        std::set<int> cells;
        segment u = m.geometry.getNonOriented(s);
        int last = path_engine::is_node(u.target) ? u.target : u.source;
        for (int v = u.source; v <= last; ++v)
            for (Eigen::SparseMatrix<double>::InnerIterator it(P.s_pod, v); it; ++it)
                if (it.value() > 0) cells.insert(static_cast<int>(it.row()));
        return cells;
    };
    for (const auto& b : boxes) {
        SCOPED_TRACE("box (" + std::to_string(b.ul.x) + "," + std::to_string(b.ul.y) + ")-(" +
                     std::to_string(b.lr.x) + "," + std::to_string(b.lr.y) + ")");
        std::set<int> boxCells;
        for (int y = b.ul.y; y < b.lr.y; ++y)
            for (int x = b.ul.x; x < b.lr.x; ++x) boxCells.insert(y * cols + x);
        auto hs = m.getCompleteCoverageHorizontalSegments(b);
        auto vs = m.getCompleteCoverageVerticalSegments(b);
        EXPECT_EQ(static_cast<int>(hs.size()), std::max(0, std::min(rows, b.ul.y + R) - std::max(0, b.lr.y - R) + 1));
        EXPECT_EQ(static_cast<int>(vs.size()), std::max(0, std::min(cols, b.ul.x + R) - std::max(0, b.lr.x - R) + 1));
        for (const auto& s : hs) {
            ASSERT_TRUE(m.geometry.is_horizontal(s)) << segStr(s);
            auto covered = coveredCells(s);
            EXPECT_TRUE(std::includes(covered.begin(), covered.end(), boxCells.begin(), boxCells.end()))
                << "horizontal " << segStr(s) << " does not cover the whole box";
        }
        for (const auto& s : vs) {
            ASSERT_FALSE(m.geometry.is_horizontal(s)) << segStr(s);
            auto covered = coveredCells(s);
            EXPECT_TRUE(std::includes(covered.begin(), covered.end(), boxCells.begin(), boxCells.end()))
                << "vertical " << segStr(s) << " does not cover the whole box";
        }
        EXPECT_EQ(m.getCompleteCoverageSegments(b).size(), hs.size() + vs.size());
    }
}

// An empty box (default iRectangle, no unsatisfied cell) yields no segment.
TEST_F(simple_units_fixture, CompleteCoverageSegmentsOfEmptyBoxIsEmpty) {
    trim_acc m(ctx, P, sol);
    iRectangle empty;
    EXPECT_TRUE(m.getCompleteCoverageSegments(empty).empty());
}

// ============================================================================
// Selection and application invariants, for every simple neighborhood, on
// the 6x6 fixture paths and on a dp_sweeper solution of the 16x16 instance.
//
// For each draw the selection must be applicable together:
//   - only strictly improving moves, never the start segment;
//   - all from one orientation group (same interval orientation);
//   - on a shared line, pairwise disjoint intervals; different lines at
//     least spacingConstraint*max_range apart (MaximumCover);
//   - n12/n21: referenced path indices of distinct moves at least 2 apart
//     (RandomKeyPacking);
// and applying it keeps coverage feasible, the tracked coverage in sync and
// the tracked cost equal to the geometric cost.
// ============================================================================

template<typename N> struct MoveIndices;
template<> struct MoveIndices<neighborhood_n12> {
    static std::vector<int> of(const OneTwoReallocationMove& m, int i) {
        return {i, m.reallocationIndexes[0], m.reallocationIndexes[1]};
    }
};
template<> struct MoveIndices<neighborhood_n21> {
    static std::vector<int> of(const TwoOneReallocationMove& m, int i) {
        return {i, m.removalIndex, m.reallocationIndex};
    }
};
template<typename N> constexpr bool kIsReallocation =
        std::is_same_v<N, neighborhood_n12> || std::is_same_v<N, neighborhood_n21>;

template<typename N>
int insertedReplacements(const units_accessor<N>& m, const std::vector<int>& sel) {
    int count = 0;
    if constexpr (std::is_same_v<N, neighborhood_n21>) {
        for (int i : sel) count += path_engine::is_node(m.candidatesMoves[i].replacement.source);
    } else if constexpr (std::is_same_v<N, neighborhood_n12>) {
        for (int i : sel) count += path_engine::is_node(m.candidatesMoves[i].replacements[0].source)
                                 + path_engine::is_node(m.candidatesMoves[i].replacements[1].source);
    }
    return count;
}

struct SelectionStats {
    int draws = 0, multi = 0, multiInsert = 0, maxSize = 0, maxImproving = 0, multiCompatible = 0;
};

template<typename N>
void checkDraw(cppied_context& ctx, cppied_instance& P, cppied_solution& s, SelectionStats& stats) {
    units_accessor<N> m(ctx, P, s);
    if (path_engine::is_node(s.path[0].target)) {
        auto split = m.geometry.splitSegment(s.path[0], 0);
        s.path[0] = split.first;
        s.path.insert(s.path.begin() + 1, split.second);
    }
    m.setMovesAndGains();
    std::vector<int> improving;
    for (int i = 1; i < static_cast<int>(m.candidatesMoves.size()); ++i)
        if (m.candidatesMoves[i].gain > cost_t{0, 0}) improving.push_back(i);
    stats.maxImproving = std::max<int>(stats.maxImproving, static_cast<int>(improving.size()));
    // Two improving moves of the same orientation on lines at least the
    // spacing apart are always co-selectable by the interval scheduling.
    {
        const int sp = m.spacingConstraint * P.max_range;
        bool compatible = false;
        for (size_t a = 0; a < improving.size() && !compatible; ++a)
            for (size_t b = a + 1; b < improving.size() && !compatible; ++b) {
                Interval u = m.transformToInterval(improving[a]), v = m.transformToInterval(improving[b]);
                bool sameOddEven = (improving[a] % 2) == (improving[b] % 2);
                compatible = u.rotated == v.rotated && std::abs(u.y - v.y) >= sp &&
                             (sameOddEven || kIsReallocation<N> || std::is_same_v<N, neighborhood_cut>);
            }
        stats.multiCompatible += compatible;
    }
    std::vector<int> sel = m.selectMoves();
    stats.draws++;
    stats.maxSize = std::max<int>(stats.maxSize, static_cast<int>(sel.size()));
    stats.multi += sel.size() >= 2;
    stats.multiInsert += insertedReplacements<N>(m, sel) >= 2;

    const int spacing = m.spacingConstraint * P.max_range;
    std::vector<Interval> ivs;
    for (int i : sel) {
        ASSERT_GT(i, 0) << "start segment selected";
        ASSERT_LT(i, static_cast<int>(s.path.size()));
        EXPECT_GT(m.candidatesMoves[i].gain, (cost_t{0, 0})) << "non-improving move " << i << " selected";
        ivs.push_back(m.transformToInterval(i));
    }
    for (size_t a = 0; a < ivs.size(); ++a)
        for (size_t b = a + 1; b < ivs.size(); ++b) {
            const Interval &u = ivs[a], &v = ivs[b];
            EXPECT_EQ(u.rotated, v.rotated) << "moves " << u.id << " and " << v.id << " from different orientation groups";
            if (u.y == v.y)
                EXPECT_TRUE(u.x2 < v.x1 || v.x2 < u.x1)
                    << "overlapping moves " << u.id << " [" << u.x1 << "," << u.x2 << "] and " << v.id
                    << " [" << v.x1 << "," << v.x2 << "] on line " << u.y;
            else
                EXPECT_GE(std::abs(u.y - v.y), spacing) << "moves " << u.id << " and " << v.id << " on close lines";
        }
    if constexpr (kIsReallocation<N>) {
        for (size_t a = 0; a < sel.size(); ++a)
            for (size_t b = a + 1; b < sel.size(); ++b)
                for (int x : MoveIndices<N>::of(m.candidatesMoves[sel[a]], sel[a]))
                    for (int y : MoveIndices<N>::of(m.candidatesMoves[sel[b]], sel[b]))
                        if (x >= 0 && y >= 0)
                            EXPECT_GE(std::abs(x - y), 2) << "moves " << sel[a] << " and " << sel[b]
                                                          << " reference close indices " << x << " and " << y;
    }
    const segment start = s.path[0];
    m.applyMoves(sel);
    m.clear();
    EXPECT_EQ(s.path[0], start) << "start segment displaced";
    cppied_solution fresh = s;
    m.coverage.reset(fresh);
    EXPECT_TRUE((fresh.coverage.array() >= P.req.array()).all()) << "coverage broken";
    EXPECT_TRUE(fresh.coverage.isApprox(s.coverage)) << "tracked coverage out of sync";
    EXPECT_EQ(s.cost, m.geometry.cost(s)) << "tracked cost differs from geometric cost (" << sel.size() << " moves)";
    for (const auto& seg : s.path) EXPECT_TRUE(path_engine::is_node(seg.source)) << "NULL segment left in path";
}

template<typename N>
class selection_small_fixture : public cppied_context_fixture {};
template<typename N>
class selection_large_fixture : public cppied_context_large_fixture {};

using SimpleNeighborhoodTypes = ::testing::Types<neighborhood_trim, neighborhood_n1, neighborhood_cut,
                                                 neighborhood_reduce, neighborhood_n12, neighborhood_n21>;
TYPED_TEST_SUITE(selection_small_fixture, SimpleNeighborhoodTypes);
TYPED_TEST_SUITE(selection_large_fixture, SimpleNeighborhoodTypes);

// Small-grid inputs: fixture paths and the n21 inputs captured from ils_full
// runs (known to produce multi-move selections).
static std::vector<std::vector<segment>> smallInputs() {
    return {
        {{6, 11}, {23, 18}, {44, 42}, {48, 51}, {30, 35}, {75, 73}, {62, 63}},
        {{6, 11}, {23, 18}, {44, 43}, {6, path_engine::NULL_NODE}, {48, 51}, {30, 35}, {75, 73}, {62, 63}},
        {{6, 11}, {79, path_engine::NULL_NODE}, {23, 18}, {44, 42}, {48, 51}, {30, 35}, {75, 74}, {10, 9}, {62, 63}},
        // n21 inputs captured from ils_full runs (multi-move selections).
        {{6,-1},{12,-2},{6,11},{79,80},{35,30},{45,42},{54,57},{18,23},{29,28},{63,62},{16,-1}},
        {{6,-1},{48,-2},{7,6},{18,22},{23,-1},{35,30},{44,-2},{6,11},{79,-1},{75,74},{16,-2},{15,12},{18,21}},
        {{6,-1},{48,51},{30,35},{23,18},{44,-2},{6,9},{72,73},{62,63},{75,73},{11,-1}},
        {{6,-1},{50,-1},{18,19},{51,48},{60,-1},{62,65},{77,73},{61,63},{75,72},{48,53}},
        {{6,-1},{50,53},{18,-1},{50,49},{48,-2},{6,7},{14,16},{11,9},{48,-2},{54,57},{65,63},{21,23},{17,-2},{75,77}},
        {{6,-1},{7,11},{23,20},{25,24},{45,42},{0,1},{9,10},{79,81},{35,30},{44,-2},{12,13},{56,57},{27,28},{16,15}},
        {{6,-1},{7,11},{79,-1},{23,20},{19,18},{6,7},{67,-1},{20,18},{44,-2},{6,-1},{21,23},{29,24},{30,35}},
        {{6,-1},{7,11},{80,81},{29,-2},{77,-1},{71,68},{56,59},{53,48},{42,44},{18,-1},{20,22},{73,-2},{22,23}},
        {{6,-1},{7,8},{48,51},{52,53},{65,62},{61,60},{10,11},{73,75},{76,77},{23,22},{21,20},{19,18},{44,-2},{43,42}},
        {{6,-1},{7,9},{73,75},{69,67},{10,11},{79,81},{35,33},{63,62},{56,59},{53,51},{50,-2},{6,-2},{18,-1},{50,48},{0,-2}}};
}

TYPED_TEST(selection_small_fixture, SelectionIsJointlyApplicable) {
    const std::vector<std::vector<segment>> paths = smallInputs();
    SelectionStats stats;
    for (const auto& p : paths) {
        for (int draw = 0; draw < 10; ++draw) {
            cppied_solution s;
            s.path = p;
            s.coverage = Eigen::VectorXd::Zero(this->P.req.size());
            path_engine g(this->P);
            coverage_engine c(this->P);
            c.reset(s);
            s.cost = g.cost(s);
            ASSERT_TRUE((s.coverage.array() >= this->P.req.array()).all());
            checkDraw<TypeParam>(this->ctx, this->P, s, stats);
            if (this->HasFailure()) return;
        }
    }
    // n21 multi-move selections need two disjoint intervals on one line or
    // lines 6*max_range apart; the test grids rarely allow either, so the
    // multi-move application is covered by the hand-made-move tests below.
}

TYPED_TEST(selection_large_fixture, SelectionIsJointlyApplicable) {
    path_engine g(this->P);
    coverage_engine c(this->P);
    cppied_solution start;
    start.coverage = Eigen::VectorXd::Zero(this->P.req.size());
    start.cost = {0, 0};
    DpSweeper h(this->ctx, this->P, start);
    h.construct([](cppied_solution&){});
    g.complete(start);
    c.reset(start);
    start.cost = g.cost(start);
    ASSERT_TRUE((start.coverage.array() >= this->P.req.array()).all());

    SelectionStats stats;
    for (int run = 0; run < 3; ++run) {
        cppied_solution s = start;
        for (int it = 0; it < 6; ++it) {
            checkDraw<TypeParam>(this->ctx, this->P, s, stats);
            if (this->HasFailure()) return;
        }
    }

    // Inflated start: every segment extended to its whole line and three
    // segments duplicated. Still feasible, with many independent improving
    // moves on far-apart lines, so that multi-move applications happen.
    cppied_solution inflated = start;
    for (size_t k = 1; k < inflated.path.size(); ++k) {
        const segment u = g.getNonOriented(inflated.path[k]);
        segment full;
        if (g.is_horizontal(u)) {
            const int row = g.get_row(u);
            full = {row * g.n_cols, row * g.n_cols + g.n_cols - 1};
        } else {
            const int col = g.get_column(u);
            full = {g.horizontal_bound + col * g.n_rows, g.horizontal_bound + col * g.n_rows + g.n_rows - 1};
        }
        g.correct_segment_direction(full, g.get_direction(inflated.path[k]));
        inflated.path[k] = full;
    }
    for (size_t k = 1; k < 4 && k < start.path.size(); ++k) inflated.path.push_back(inflated.path[k]);
    c.reset(inflated);
    inflated.cost = g.cost(inflated);
    ASSERT_TRUE((inflated.coverage.array() >= this->P.req.array()).all());
    for (int run = 0; run < 3; ++run) {
        cppied_solution s = inflated;
        for (int it = 0; it < 4; ++it) {
            checkDraw<TypeParam>(this->ctx, this->P, s, stats);
            if (this->HasFailure()) return;
        }
    }
    // The multi-move application paths must actually be exercised (n21 is
    // exercised on the captured inputs of the small test).
    if constexpr (!std::is_same_v<TypeParam, neighborhood_n21>)
        EXPECT_GT(stats.multi, 0) << "no draw selected two or more moves (max selected " << stats.maxSize
                                  << ", max improving candidates " << stats.maxImproving << ", draws with two "
                                  << "far-apart compatible improving moves " << stats.multiCompatible << "/" << stats.draws << ")";
}

static int pathVertexCount(const std::vector<segment>& path) {
    int count = 0;
    for (const segment& seg : path)
        count += path_engine::is_node(seg.target) ? std::abs(seg.target - seg.source) + 1 : 1;
    return count;
}

// local_search returns true only when it made progress, and callers loop on
// its return value (nested::n21_search has no other stopping criterion), so
// returning true without progress never ends. Progress means:
//   trim, n1, n12, n21: strictly lower geometric cost;
//   cut:    cost not higher and fewer vertices (gain = removed vertices);
//   reduce: cost not higher and fewer segments (equal-cost removals of
//           redundant segments get a {0,1} bonus).
// A false return leaves the cost unchanged.
// Regressions: n21::applyMoves applied nothing (selection not copied into
// the member it walks) while local_search returned true; cut pushed the
// pieces of a reversed segment in reverse travel order, raising the cost.
TYPED_TEST(selection_small_fixture, LocalSearchReturnsTrueOnlyOnProgress) {
    for (const auto& p : smallInputs()) {
        for (int draw = 0; draw < 5; ++draw) {
            cppied_solution s;
            s.path = p;
            s.coverage = Eigen::VectorXd::Zero(this->P.req.size());
            path_engine g(this->P);
            coverage_engine c(this->P);
            c.reset(s);
            s.cost = g.cost(s);
            const cost_t before = g.cost(s);
            // Count after the start-vertex split local_search performs.
            cppied_solution split = s;
            if (path_engine::is_node(split.path[0].target)) {
                auto sp = g.splitSegment(split.path[0], 0);
                split.path[0] = sp.first;
                split.path.insert(split.path.begin() + 1, sp.second);
            }
            const int verticesBefore = pathVertexCount(split.path);
            const size_t segmentsBefore = split.path.size();
            bool improved = units_accessor<TypeParam>(this->ctx, this->P, s).local_search();
            const cost_t after = g.cost(s);
            if (!improved) {
                ASSERT_EQ(after, before) << "local_search returned false but changed the cost";
            } else if constexpr (std::is_same_v<TypeParam, neighborhood_cut>) {
                ASSERT_LE(after, before) << "cut raised the cost";
                ASSERT_LT(pathVertexCount(s.path), verticesBefore) << "cut returned true without removing a vertex";
            } else if constexpr (std::is_same_v<TypeParam, neighborhood_reduce>) {
                ASSERT_LE(after, before) << "reduce raised the cost";
                ASSERT_LT(s.path.size(), segmentsBefore) << "reduce returned true without removing a segment";
            } else {
                ASSERT_LT(after, before) << "local_search returned true without lowering the cost";
            }
        }
    }
}

// Repeated local searches reach a local optimum in a bounded number of
// calls (each improving call lowers the integer cost).
TYPED_TEST(selection_small_fixture, LocalSearchLoopTerminates) {
    for (const auto& p : smallInputs()) {
        cppied_solution s;
        s.path = p;
        s.coverage = Eigen::VectorXd::Zero(this->P.req.size());
        path_engine g(this->P);
        coverage_engine c(this->P);
        c.reset(s);
        s.cost = g.cost(s);
        // Each progressing call lowers the cost or removes vertices/segments.
        const int bound = s.cost.length + s.cost.turns + pathVertexCount(s.path) + 2;
        int calls = 0;
        while (units_accessor<TypeParam>(this->ctx, this->P, s).local_search())
            ASSERT_LE(++calls, bound) << "local_search keeps returning true";
    }
}

// ----------------------------------------------------------------------------
// Multi-move application with hand-made moves (n21 / n12)
//
// Insertion positions are given in the opposite order of the moved segments
// so that the insertions must be sorted. A replacement is inserted before the
// segment currently at its reallocation index; an index equal to the path
// size appends it.
// ----------------------------------------------------------------------------

using n21_acc = units_accessor<neighborhood_n21>;
using n12_acc = units_accessor<neighborhood_n12>;

static std::vector<segment> expectedAfterMoves(const std::vector<segment>& path, const std::set<int>& removed,
                                               std::vector<std::pair<segment, int>> inserts) {
    std::stable_sort(inserts.begin(), inserts.end(), [](auto& a, auto& b){ return a.second < b.second; });
    std::vector<segment> out;
    auto it = inserts.begin();
    for (int i = 0; i < static_cast<int>(path.size()); ++i) {
        while (it != inserts.end() && it->second == i) out.push_back((it++)->first);
        if (!removed.count(i)) out.push_back(path[i]);
    }
    for (; it != inserts.end(); ++it) out.push_back(it->first);
    return out;
}

TEST_F(simple_units_fixture, N21AppliesSeveralMovesInPositionOrder) {
    // path: {6,11} {23,18} {44,42} {48,51} {30,35} {75,73} {62,63}
    n21_acc m(ctx, P, sol);
    m.candidatesMoves.assign(sol.path.size(), TwoOneReallocationMove{});
    // move on 1 (+ paired removal 2) inserted at 6; move on 4 (+ removal 5) inserted at 1.
    m.candidatesMoves[1] = {{20, 21}, 6, 2, {2, 0}};
    m.candidatesMoves[4] = {{12, 13}, 1, 5, {1, 1}};
    const cost_t before = sol.cost;
    auto expected = expectedAfterMoves(sol.path, {1, 2, 4, 5}, {{{20, 21}, 6}, {{12, 13}, 1}});
    std::vector<int> sel = {4, 1};
    m.applyMoves(sel);
    EXPECT_EQ(sol.path, expected);
    EXPECT_EQ(sol.cost, (before - cost_t{2, 0} - cost_t{1, 1}));
    cppied_solution fresh = sol; m.coverage.reset(fresh);
    EXPECT_TRUE(fresh.coverage.isApprox(sol.coverage)) << "tracked coverage out of sync";
}

TEST_F(simple_units_fixture, N21AppliesDirectRemovalAndTailInsertion) {
    n21_acc m(ctx, P, sol);
    m.candidatesMoves.assign(sol.path.size(), TwoOneReallocationMove{});
    const int size = static_cast<int>(sol.path.size());
    m.candidatesMoves[2] = {{path_engine::NULL_NODE, path_engine::NULL_NODE}, -1, -1, {1, 0}};   // direct removal
    m.candidatesMoves[5] = {{20, 21}, size, 3, {1, 0}};                                           // append at the end
    auto expected = expectedAfterMoves(sol.path, {2, 3, 5}, {{{20, 21}, size}});
    std::vector<int> sel = {5, 2};
    m.applyMoves(sel);
    EXPECT_EQ(sol.path, expected);
    cppied_solution fresh = sol; m.coverage.reset(fresh);
    EXPECT_TRUE(fresh.coverage.isApprox(sol.coverage));
}

// Applying twice on the same object starts from a clean state (members that
// accumulate between applications are reset).
TEST_F(simple_units_fixture, N21RepeatedApplicationsDoNotAccumulate) {
    n21_acc m(ctx, P, sol);
    m.candidatesMoves.assign(sol.path.size(), TwoOneReallocationMove{});
    m.candidatesMoves[1] = {{20, 21}, 6, 2, {1, 0}};
    std::vector<int> sel = {1};
    m.applyMoves(sel);
    m.clear();
    auto afterFirst = sol.path;
    m.candidatesMoves.assign(sol.path.size(), TwoOneReallocationMove{});
    m.candidatesMoves[3] = {{path_engine::NULL_NODE, path_engine::NULL_NODE}, -1, -1, {1, 0}};
    std::vector<int> sel2 = {3};
    m.applyMoves(sel2);
    auto expected = afterFirst;
    expected.erase(expected.begin() + 3);
    EXPECT_EQ(sol.path, expected) << "state of the first application leaked into the second";
}

TEST_F(simple_units_fixture, N12AppliesSeveralMovesInPositionOrder) {
    n12_acc m(ctx, P, sol);
    m.candidatesMoves.assign(sol.path.size(), OneTwoReallocationMove{});
    // move on 2 split into two segments inserted at 6 and 5; move on 4 into one inserted at 1.
    m.candidatesMoves[2] = {{segment{44, 43}, segment{42, path_engine::NULL_NODE}}, {6, 5}, {1, 0}};
    m.candidatesMoves[4] = {{segment{30, 33}, segment{path_engine::NULL_NODE, path_engine::NULL_NODE}}, {1, -1}, {0, 1}};
    const cost_t before = sol.cost;
    auto expected = expectedAfterMoves(sol.path, {2, 4}, {{{44, 43}, 6}, {{42, path_engine::NULL_NODE}, 5}, {{30, 33}, 1}});
    std::vector<int> sel = {4, 2};
    m.applyMoves(sel);
    EXPECT_EQ(sol.path, expected);
    EXPECT_EQ(sol.cost, (before - cost_t{1, 0} - cost_t{0, 1}));
    cppied_solution fresh = sol; m.coverage.reset(fresh);
    EXPECT_TRUE(fresh.coverage.isApprox(sol.coverage));
}
