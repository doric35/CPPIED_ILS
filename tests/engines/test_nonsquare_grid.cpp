#include "../test_fixture.hpp"
#include "../../include/construction/dp_sweeper.hpp"
#include "../../include/neighborhoods/neighborhood_n1.hpp"
#include "../../include/neighborhoods/neighborhood_n12.hpp"
#include "../../include/neighborhoods/neighborhood_n21.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_cut.hpp"
#include "../../include/neighborhoods/neighborhood_reduce.hpp"
#include "../../include/neighborhoods/neighborhood_nested.hpp"
#include "../../include/neighborhoods/neighborhood_tsp.hpp"
#include "../../include/neighborhoods/neighborhood_gtsp.hpp"
#include "../../include/metaheuristics/ils.hpp"

// ============================================================================
// Non-square grids.
//
// Vertex numbering (cppied_instance::initialize_vertices):
//   horizontal vertex (row r in [0, n_rows], col j in [0, n_cols)) has id
//   r*n_cols + j and sits at (j+0.5, r);
//   vertical vertex (col c in [0, n_cols], row i in [0, n_rows)) has id
//   horizontal_bound + c*n_rows + i and sits at (c, i+0.5),
//   with horizontal_bound = (n_rows+1)*n_cols.
// On a square grid horizontal_bound is a multiple of n_rows, which hides
// arithmetic that uses `id % n_rows` for the position of a vertical vertex.
// The instances below have n_rows not dividing n_cols (and vice versa).
// ============================================================================

struct GridCase {
    const char* name;
    int rows;
    int cols;
};

static std::string instanceFile(const GridCase& g, const char* kind) {
    return std::string(PROJECT_SOURCE_DIR "/tests/configurations/test_ns") +
           std::to_string(g.rows) + "x" + std::to_string(g.cols) + "_" + kind + ".txt";
}

static void PrintTo(const GridCase& g, std::ostream* os) { *os << g.name; }

class nonsquare_grid_fixture : public ::testing::TestWithParam<GridCase> {
protected:
    static constexpr const char* config_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_config.txt";

    cppied_context ctx;
    cppied_instance P;
    path_engine geometry;
    coverage_engine coverage;

    nonsquare_grid_fixture()
        : ctx(read_configuration(config_path)),
          P(read_matrix<int>(instanceFile(GetParam(), "seabed")),
            read_matrix<double>(instanceFile(GetParam(), "pod")),
            read_matrix<double>(instanceFile(GetParam(), "req"))),
          geometry(P), coverage(P) {}

    int hId(int row, int col) const { return row * geometry.n_cols + col; }
    int vId(int col, int row) const { return geometry.horizontal_bound + col * geometry.n_rows + row; }
    bool isVertical(int v) const { return v >= geometry.horizontal_bound; }
    // Position of a vertex along its own line.
    int position(int v) const {
        return isVertical(v) ? static_cast<int>(std::floor(P.vertex[v].y))
                             : static_cast<int>(std::floor(P.vertex[v].x));
    }

    bool feasible(const cppied_solution& s) {
        cppied_solution c = s;
        coverage.reset(c);
        return (c.coverage.array() >= P.req.array()).all();
    }

    cppied_solution construct() {
        cppied_solution s;
        s.coverage = Eigen::VectorXd::Zero(P.req.size());
        s.cost = {0, 0};
        dp_sweeper h(ctx, P);
        h.construct(s, [](cppied_solution&){});
        geometry.complete(s);
        coverage.reset(s);
        s.cost = geometry.cost(s);
        return s;
    }
};

// ----------------------------------------------------------------------------
// Instance and numbering
// ----------------------------------------------------------------------------

TEST_P(nonsquare_grid_fixture, InstanceIsNonSquare) {
    EXPECT_EQ(geometry.n_rows, GetParam().rows);
    EXPECT_EQ(geometry.n_cols, GetParam().cols);
    EXPECT_NE(geometry.horizontal_bound % geometry.n_rows, 0)
        << "instance does not exercise vertical positions != id % n_rows";
    EXPECT_EQ(static_cast<int>(P.vertex.size()),
              (geometry.n_rows + 1) * geometry.n_cols + geometry.n_rows * (geometry.n_cols + 1));
}

TEST_P(nonsquare_grid_fixture, VertexNumberingMatchesCoordinates) {
    for (int r = 0; r <= geometry.n_rows; ++r)
        for (int j = 0; j < geometry.n_cols; ++j) {
            const int v = hId(r, j);
            segment s{v, path_engine::NULL_NODE};
            EXPECT_TRUE(geometry.is_horizontal(s)) << "vertex " << v;
            EXPECT_EQ(geometry.get_row(s), r) << "vertex " << v;
            EXPECT_DOUBLE_EQ(P.vertex[v].x, j + 0.5);
            EXPECT_DOUBLE_EQ(P.vertex[v].y, r);
        }
    for (int c = 0; c <= geometry.n_cols; ++c)
        for (int i = 0; i < geometry.n_rows; ++i) {
            const int v = vId(c, i);
            segment s{v, path_engine::NULL_NODE};
            EXPECT_FALSE(geometry.is_horizontal(s)) << "vertex " << v;
            EXPECT_EQ(geometry.get_column(s), c) << "vertex " << v;
            EXPECT_DOUBLE_EQ(P.vertex[v].x, c);
            EXPECT_DOUBLE_EQ(P.vertex[v].y, i + 0.5);
        }
}

// Every vertex covers exactly the cells next to it on its side of the grid:
// a horizontal vertex at (j+0.5, r) covers column j, rows [r-R, r+R-1];
// a vertical vertex at (c, i+0.5) covers row i, columns [c-R, c+R-1]
// (R = max_range, clamped to the grid). Cells are numbered i*n_cols + j.
// Regression: set_cover numbered vertical cells with i*n_rows + j.
TEST_P(nonsquare_grid_fixture, PodCoversAdjacentCells) {
    const int R = P.max_range, rows = geometry.n_rows, cols = geometry.n_cols;
    for (int v = 0; v < static_cast<int>(P.vertex.size()); ++v) {
        std::set<int> expected;
        if (!isVertical(v)) {
            const int r = static_cast<int>(P.vertex[v].y), j = static_cast<int>(std::floor(P.vertex[v].x));
            for (int i = std::max(0, r - R); i <= std::min(rows - 1, r + R - 1); ++i) expected.insert(i * cols + j);
        } else {
            const int c = static_cast<int>(P.vertex[v].x), i = static_cast<int>(std::floor(P.vertex[v].y));
            for (int j = std::max(0, c - R); j <= std::min(cols - 1, c + R - 1); ++j) expected.insert(i * cols + j);
        }
        std::set<int> actual;
        for (Eigen::SparseMatrix<double>::InnerIterator it(P.s_pod, v); it; ++it) actual.insert(static_cast<int>(it.row()));
        EXPECT_EQ(actual, expected) << "vertex " << v << (isVertical(v) ? " (vertical)" : " (horizontal)");
    }
}

// splitSegment stays on the segment's line for every orientation and index.
TEST_P(nonsquare_grid_fixture, SplitSegmentStaysOnLine) {
    std::vector<segment> segments = {
        {hId(1, 0), hId(1, geometry.n_cols - 1)},
        {hId(geometry.n_rows, geometry.n_cols - 1), hId(geometry.n_rows, 0)},
        {vId(0, 0), vId(0, geometry.n_rows - 1)},
        {vId(geometry.n_cols, geometry.n_rows - 1), vId(geometry.n_cols, 0)}};
    for (auto& s : segments) {
        const int length = geometry.cost(s).length;
        const bool horizontal = geometry.is_horizontal(s);
        const int line = horizontal ? geometry.get_row(s) : geometry.get_column(s);
        for (int index = 0; index < length; ++index) {
            auto [first, second] = geometry.splitSegment(s, index);
            for (const segment& piece : {first, second}) {
                ASSERT_TRUE(path_engine::is_node(piece.source));
                EXPECT_EQ(geometry.is_horizontal(piece), horizontal);
                EXPECT_EQ(horizontal ? geometry.get_row(piece) : geometry.get_column(piece), line)
                    << "split of {" << s.source << "," << s.target << "} at " << index << " left its line";
            }
            EXPECT_EQ(first.source, s.source);
        }
    }
}

// ----------------------------------------------------------------------------
// n21 intervals
// ----------------------------------------------------------------------------

struct nonsquare_n21_accessor : public neighborhood_n21 {
    using neighborhood_n21::neighborhood_n21;
    using neighborhood_n21::candidatesMoves;
    using neighborhood_n21::transformToInterval;
};

class nonsquare_n21_fixture : public nonsquare_grid_fixture {
protected:
    cppied_solution sol;

    Interval interval(const segment& first, int pairedRemoval, const segment& paired = {}) {
        sol.path = {{hId(0, 0), path_engine::NULL_NODE}, first};
        if (pairedRemoval >= 0) sol.path.push_back(paired);
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        nonsquare_n21_accessor m(ctx, P, sol);
        m.candidatesMoves.assign(sol.path.size(), TwoOneReallocationMove{});
        m.candidatesMoves[1].removalIndex = pairedRemoval;
        m.candidatesMoves[1].gain = {1, 0};
        return m.transformToInterval(1);
    }
};

// An interval of a single segment is the segment's positions on its line,
// and its line index is the segment's row (horizontal) or column (vertical).
TEST_P(nonsquare_n21_fixture, SingleSegmentIntervalIsItsPositions) {
    for (int v = 0; v < static_cast<int>(P.vertex.size()); ++v) {
        Interval iv = interval({v, path_engine::NULL_NODE}, -1);
        SCOPED_TRACE("vertex " + std::to_string(v));
        EXPECT_EQ(iv.x1, position(v));
        EXPECT_EQ(iv.x2, position(v));
        EXPECT_EQ(iv.rotated, isVertical(v));
        segment s{v, path_engine::NULL_NODE};
        EXPECT_EQ(iv.y, isVertical(v) ? geometry.get_column(s) : geometry.get_row(s));
    }
}

TEST_P(nonsquare_n21_fixture, MultiNodeSegmentIntervalIsItsPositions) {
    // Whole last horizontal row and whole last vertical line, both orientations.
    const int R = geometry.n_rows, C = geometry.n_cols;
    struct Case { segment s; int x1, x2; };
    std::vector<Case> cases = {
        {{hId(R, 1), hId(R, C - 1)}, 1, C - 1},
        {{hId(R, C - 1), hId(R, 1)}, 1, C - 1},
        {{vId(C, 1), vId(C, R - 1)}, 1, R - 1},
        {{vId(C, R - 1), vId(C, 1)}, 1, R - 1}};
    for (auto& c : cases) {
        Interval iv = interval(c.s, -1);
        EXPECT_EQ(iv.x1, c.x1) << "{" << c.s.source << "," << c.s.target << "}";
        EXPECT_EQ(iv.x2, c.x2) << "{" << c.s.source << "," << c.s.target << "}";
    }
}

// A paired removal extends the interval to every position on the first
// removal's line that its coverage reaches:
//   perpendicular paired vertex at line coordinate L reaches [L-r, L+r-1];
//   parallel paired vertex reaches its own position.
// Checked for every first-removal line and every possible paired vertex.
TEST_P(nonsquare_n21_fixture, PairedIntervalCoversPairedRemoval) {
    const int r = P.max_range;
    auto expectCovers = [&](int first, int paired) {
        Interval iv = interval({first, path_engine::NULL_NODE}, 2, {paired, path_engine::NULL_NODE});
        const bool firstVertical = isVertical(first);
        const int last = firstVertical ? geometry.n_rows - 1 : geometry.n_cols - 1;
        int lo, hi;
        if (isVertical(paired) == firstVertical) {
            lo = hi = position(paired);
        } else {
            // Coordinate of the paired vertex's line along the first line.
            const int L = firstVertical ? static_cast<int>(P.vertex[paired].y)
                                        : static_cast<int>(P.vertex[paired].x);
            lo = std::max(0, L - r);
            hi = std::min(last, L + r - 1);
        }
        lo = std::min(lo, position(first));
        hi = std::max(hi, position(first));
        EXPECT_LE(iv.x1, lo) << "first " << first << " paired " << paired;
        EXPECT_GE(iv.x2, hi) << "first " << first << " paired " << paired;
        // Intervals may extend past the line ends (n21 does not clamp the
        // perpendicular reach); only coverage of the reached positions matters.
    };
    // One first removal per line: horizontal middle row, vertical middle line.
    const int hFirst = hId(geometry.n_rows / 2, geometry.n_cols / 2);
    const int vFirst = vId(geometry.n_cols / 2, geometry.n_rows / 2);
    for (int paired = 0; paired < static_cast<int>(P.vertex.size()); ++paired) {
        if (paired != hFirst) expectCovers(hFirst, paired);
        if (paired != vFirst) expectCovers(vFirst, paired);
    }
}

// ----------------------------------------------------------------------------
// End to end: every neighborhood keeps the solution feasible and consistent
// on a non-square grid.
// ----------------------------------------------------------------------------

class nonsquare_e2e_fixture : public nonsquare_grid_fixture {
protected:
    static constexpr const char* scratch_dir = PROJECT_SOURCE_DIR "/tests/scratch";
    void SetUp() override {
        std::filesystem::create_directories(scratch_dir);
        ctx.start_time = std::chrono::high_resolution_clock::now();
    }

    // Repeated local searches from the same start (the neighborhoods are
    // randomized): every call keeps coverage feasible and the tracked
    // coverage vector in sync.
    template<typename N>
    void runNeighborhood(const cppied_solution& start, const char* name) {
        for (int run = 0; run < 5; ++run) {
            cppied_solution s = start;
            for (int it = 0; it < 20; ++it) {
                bool improved = N(ctx, P, s).local_search();
                ASSERT_TRUE(feasible(s)) << name << " run " << run << " iteration " << it << ": coverage broken";
                cppied_solution fresh = s;
                coverage.reset(fresh);
                ASSERT_TRUE(fresh.coverage.isApprox(s.coverage))
                    << name << " run " << run << " iteration " << it << ": tracked coverage out of sync";
                if (!improved) break;
            }
        }
    }
};

TEST_P(nonsquare_e2e_fixture, ConstructionIsFeasible) {
    cppied_solution s = construct();
    EXPECT_TRUE(feasible(s));
    EXPECT_EQ(s.cost, geometry.cost(s));
}

#define NONSQUARE_NEIGHBORHOOD_TEST(TestName, N)                                                  \
    TEST_P(nonsquare_e2e_fixture, TestName) {                                                     \
        cppied_solution start = construct();                                                      \
        ASSERT_TRUE(feasible(start));                                                             \
        runNeighborhood<N>(start, #N);                                                            \
    }

NONSQUARE_NEIGHBORHOOD_TEST(ReduceKeepsFeasibility, neighborhood_reduce)
NONSQUARE_NEIGHBORHOOD_TEST(TrimKeepsFeasibility, neighborhood_trim)
NONSQUARE_NEIGHBORHOOD_TEST(CutKeepsFeasibility, neighborhood_cut)
NONSQUARE_NEIGHBORHOOD_TEST(N1KeepsFeasibility, neighborhood_n1)
NONSQUARE_NEIGHBORHOOD_TEST(N12KeepsFeasibility, neighborhood_n12)
NONSQUARE_NEIGHBORHOOD_TEST(N21KeepsFeasibility, neighborhood_n21)
NONSQUARE_NEIGHBORHOOD_TEST(NestedKeepsFeasibility, neighborhood_nested)
NONSQUARE_NEIGHBORHOOD_TEST(TspKeepsFeasibility, neighborhood_tsp)
NONSQUARE_NEIGHBORHOOD_TEST(GtspKeepsFeasibility, neighborhood_gtsp)

// Full ILS (all neighborhoods, perturbations and restart; needs Gurobi for R).
class nonsquare_ils_fixture : public nonsquare_e2e_fixture {
protected:
    static constexpr const char* ils_config_path =
            PROJECT_SOURCE_DIR "/tests/configurations/ils_test_config.txt";
};

TEST_P(nonsquare_ils_fixture, IlsSolutionIsFeasibleAndConsistent) {
    cppied_context ils_ctx = read_configuration(ils_config_path);
    ils algorithm(ils_ctx, P);
    algorithm.initialize();
    ils_ctx.start_time = std::chrono::high_resolution_clock::now();
    cppied_solution s;
    s.coverage = Eigen::VectorXd::Zero(P.req.size());
    s.cost = {0, 0};
    ASSERT_NO_THROW(algorithm.d_solve(s));
    EXPECT_TRUE(feasible(s));
    EXPECT_EQ(geometry.cost(s), s.cost);
    cppied_solution fresh = s;
    coverage.reset(fresh);
    EXPECT_TRUE(fresh.coverage.isApprox(s.coverage));
}

static const GridCase kNonSquareGrids[] = {{"Grid4x7", 4, 7}, {"Grid7x4", 7, 4}};
static std::string gridName(const ::testing::TestParamInfo<GridCase>& info) { return info.param.name; }

INSTANTIATE_TEST_SUITE_P(NonSquare, nonsquare_grid_fixture, ::testing::ValuesIn(kNonSquareGrids), gridName);
INSTANTIATE_TEST_SUITE_P(NonSquare, nonsquare_n21_fixture, ::testing::ValuesIn(kNonSquareGrids), gridName);
INSTANTIATE_TEST_SUITE_P(NonSquare, nonsquare_e2e_fixture, ::testing::ValuesIn(kNonSquareGrids), gridName);
INSTANTIATE_TEST_SUITE_P(NonSquare, nonsquare_ils_fixture, ::testing::ValuesIn(kNonSquareGrids), gridName);
