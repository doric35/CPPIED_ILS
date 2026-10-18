#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood.hpp"
#include "../../include/engines/tsp_engine.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_cut.hpp"
// neighborhood_tsp::trial is private; expose it to the accessor so the
// tour -> path and split outputs can be inspected. Dependencies are included
// above so only neighborhood_tsp itself is affected.
#define private protected
#include "../../include/neighborhoods/neighborhood_tsp.hpp"
#undef private

struct neighborhood_tsp_accessor : public neighborhood_tsp {
public:
    // trial is only copied from the bound solution in local_search(); seed it
    // here so the helpers can be exercised on their own.
    neighborhood_tsp_accessor(cppied_context& c, cppied_instance& i, cppied_solution& s)
        : neighborhood_tsp(c, i, s) { trial = s; }

    using neighborhood::geometry;
    using neighborhood::coverage;
    using neighborhood::problem;

    using neighborhood_tsp::trial;
    using neighborhood_tsp::getDistanceMatrix;
    using neighborhood_tsp::splitTrial;
    using neighborhood_tsp::setTrialFromTspTour;
    using neighborhood_tsp::buildTspInstance;
    using neighborhood_tsp::selectSplitIndexes;
    using neighborhood_tsp::getSplitIndex;
    using neighborhood_tsp::constructSplitPath;
    using neighborhood_tsp::splitAndPush;
};

class neighborhood_tsp_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<neighborhood_tsp_accessor> n;
    cppied_solution sol;
    // Same configuration as the neighborhood's own engine.
    tsp_engine engine{ctx};

    // Non-edges of the TSP matrix are set to the trial cost length.
    Eigen::MatrixXi distances(cppied_solution& s){
        return bind(s)->getDistanceMatrix();
    }

    // Neighborhoods are bound to a solution at construction (the TSP
    // neighborhood copies it into its trial solution).
    std::unique_ptr<neighborhood_tsp_accessor> bind(cppied_solution& s){
        return std::make_unique<neighborhood_tsp_accessor>(ctx, P, s);
    }
    bool search(cppied_solution& s){
        return bind(s)->local_search();
    }

    static constexpr const char* scratch_dir =
            PROJECT_SOURCE_DIR "/tests/scratch";
    static constexpr const char* instance_name = "test_configuration_123";

    // Helpers to get canonical file paths matching ctx.config
    [[nodiscard]] std::filesystem::path tsp_file()  const {
        return std::filesystem::path(scratch_dir) / (std::string(instance_name) + ".tsp");
    }
    [[nodiscard]] std::filesystem::path tour_file() const {
        return std::filesystem::path(scratch_dir) / (std::string(instance_name) + ".tour");
    }
    [[nodiscard]] std::filesystem::path sol_file()  const {
        return std::filesystem::path(scratch_dir) / (std::string(instance_name) + ".sol");
    }

    void SetUp() override{
        cppied_context_fixture::SetUp();
        sol.path = {{6, 11},
                    {23,18},
                    {44,42},
                    {48,51},
                    {30,35},
                    {75,73},
                    {62,63}};
        sol.cost = {0,0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        n = std::make_unique<neighborhood_tsp_accessor>(ctx, P, sol);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

//----------Cost matrix testing--------------------------------------
TEST_F(neighborhood_tsp_fixture, BuildSymmetricMatrix){
    Eigen::MatrixXi C = distances(sol);
    ASSERT_EQ(C.rows(), 24);
    EXPECT_TRUE(C.isApprox(C.transpose()));
}

TEST_F(neighborhood_tsp_fixture, BuildCostMatrixNonEdgesDummyAreSentinel){
    Eigen::MatrixXi C = distances(sol);
    ASSERT_EQ(C.rows(), 24);
    for (int i = 1; i< 24; i+=3){
        EXPECT_EQ((C.row(i).array() == 0).count(), 2);
        EXPECT_EQ((C.col(i).array() == 0).count(), 2);
    }
}

TEST_F(neighborhood_tsp_fixture, BuildCostMatrixNonEdgesToDummyAreSentinel){
    Eigen::MatrixXi C = distances(sol);
    ASSERT_EQ(C.rows(), 24);
    const int sentinel = sol.cost.length;
    for (int i = 1; i< 24; i+=3){
        for (int j=0; j<24; ++j){
            if (abs(i-j) > 1)
                EXPECT_EQ(C(i,j), sentinel);
        }
    }
}

TEST_F(neighborhood_tsp_fixture, BuildCostMatrixDiagonalIsSentinel){
    Eigen::MatrixXi C = distances(sol);
    ASSERT_EQ(C.rows(), 24);
    const int sentinel = sol.cost.length;
    EXPECT_EQ((C.diagonal().array() == sentinel).count(), 24);
}

TEST_F(neighborhood_tsp_fixture, BuildCostMatrixCostSelfSegmentIsSentinel){
    Eigen::MatrixXi C = distances(sol);
    ASSERT_EQ(C.rows(), 24);
    const int sentinel = sol.cost.length;
    for (int i=0; i< 24; i+=3){
        EXPECT_EQ(C(i, i+2), sentinel);
    }
}

TEST_F(neighborhood_tsp_fixture, BuildCostMatrixCostTotargetIsZero){
    Eigen::MatrixXi C = distances(sol);
    ASSERT_EQ(C.rows(), 24);
    for (int i=6; i< 24; i+=3){
        EXPECT_EQ(C(i, 0), 0);
        EXPECT_EQ(C(i+2, 0), 0);
    }
}

TEST_F(neighborhood_tsp_fixture, BuildCostMatrixValidCurrentPath){
    Eigen::MatrixXi C = distances(sol);
    ASSERT_EQ(C.rows(), 24);
    cost_t cost = sol.cost;
    int acc = 0;
    for (int i=3; i< 24; i+=3){
        acc += n->geometry.cost(sol.path[(i -1) / 3]).length;
        if (i< 24 - 3)
            acc += C(i+2, i+3);
    }
    EXPECT_EQ(acc, cost.length);
}

//----------TSP files testing--------------------------------------
TEST_F(neighborhood_tsp_fixture, WriteInstanceFileIsCreated){
    Eigen::MatrixXi C(3,3);
    C.setConstant(std::numeric_limits<int>::max());
    C(0,1) = C(1,0) = 5;
    engine.write_instance_file(C);
    EXPECT_TRUE(std::filesystem::exists(tsp_file()));
}

TEST_F(neighborhood_tsp_fixture, WriteInitialTourIsCreated){
    std::vector<int> warm_start = {1,2,3,4,5};
    engine.write_initial_tour(warm_start);
    EXPECT_TRUE(std::filesystem::exists(tour_file()));
}

TEST_F(neighborhood_tsp_fixture, WriteInitialTourContainsRequiredHeaders){
    std::vector<int> warm_start = {1,2,3};
    engine.write_initial_tour(warm_start);
    std::ifstream f(tour_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("NAME : "  + ctx.config.at("NAME")), std::string::npos);
    EXPECT_NE(content.find("TYPE : TOUR"),         std::string::npos);
    EXPECT_NE(content.find("TOUR_SECTION"),        std::string::npos);
}

TEST_F(neighborhood_tsp_fixture, WriteInitialTourDimensionMatchesTourSize){
    std::vector<int> warm_start = {1,2,3,4};
    engine.write_initial_tour(warm_start);
    std::ifstream f(tour_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("DIMENSION : 4"), std::string::npos);
}

TEST_F(neighborhood_tsp_fixture, WriteInitialTourContainsAllNodeIds){
    std::vector<int> warm_start = {1,2,3,4,5};
    engine.write_initial_tour(warm_start);
    std::ifstream f(tour_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    for (int node : warm_start)
        EXPECT_NE(content.find(std::to_string(node) + "\n"), std::string::npos);
}

// Helper: write a syntactically valid .sol file to scratch.
static void write_test_sol(const std::filesystem::path& path,
                           const std::vector<int>& nodes){
    std::ofstream f(path);
    f << "NAME : test_sol\n";
    f << "TYPE : TOUR\n";
    f << "DIMENSION : " << nodes.size() << "\n";
    f << "TOUR_SECTION\n";
    for (int n : nodes) f << n << "\n";
    f << "-1\n";
}

TEST_F(neighborhood_tsp_fixture, ReadSolutionCorrectNumberOfNodes){
    write_test_sol(sol_file(), {1,2,3,4,5});
    std::vector<int> tour;
    engine.read_solution(tour);
    ASSERT_EQ(tour.size(), 5u);
}

TEST_F(neighborhood_tsp_fixture, ReadSolutionCorrectNodeValues){
    write_test_sol(sol_file(), {3,1,4,2});
    std::vector<int> tour;
    engine.read_solution(tour);
    ASSERT_EQ(tour.size(), 4u);
    EXPECT_EQ(tour[0], 3);
    EXPECT_EQ(tour[1], 1);
    EXPECT_EQ(tour[2], 4);
    EXPECT_EQ(tour[3], 2);
}

TEST_F(neighborhood_tsp_fixture, ReadSolutionThrowsWhenFileAbsent){
    std::filesystem::remove(sol_file());
    std::vector<int> tour;
    EXPECT_THROW(engine.read_solution(tour), std::runtime_error);
}

TEST_F(neighborhood_tsp_fixture, ReadSolutionThrowsWhenDimensionMissing){
    std::ofstream f(sol_file());
    f << "NAME : bad\nTYPE : TOUR\nTOUR_SECTION\n1\n2\n-1\n";
    f.close();
    std::vector<int> tour;
    EXPECT_THROW(engine.read_solution(tour), std::runtime_error);
}

TEST_F(neighborhood_tsp_fixture, TspToPathRoundTripPathSize){
    int dim = static_cast<int>(sol.path.size()) * 3 + 3;
    auto m = bind(sol);

    std::vector<int> tour(dim);
    std::iota(tour.begin(), tour.end(), 1);

    m->setTrialFromTspTour(tour);
    const cppied_solution& result = m->trial;
    EXPECT_EQ(result.path.size(), sol.path.size());
}

TEST_F(neighborhood_tsp_fixture, TspToPathRoundTripFirstVertex){
    int dim = static_cast<int>(sol.path.size()) * 3 + 3;
    auto m = bind(sol);

    std::vector<int> tour(dim);
    std::iota(tour.begin(), tour.end(), 1);

    m->setTrialFromTspTour(tour);
    const cppied_solution& result = m->trial;
    EXPECT_EQ(result.path[0].source, sol.path[0].source);
}

// LKH may return the tour in either direction; reversed forms of the identity
// tour (written from node 1, or ending at node 1) must give the same path.
TEST_F(neighborhood_tsp_fixture, TspToPathReversedTourRecoversPath){
    int dim = static_cast<int>(sol.path.size()) * 3 + 3;
    auto m = bind(sol);

    std::vector<int> tour(dim);
    std::iota(tour.begin(), tour.end(), 1);
    std::reverse(std::next(tour.begin()), tour.end());   // 1, N, ..., 2

    m->setTrialFromTspTour(tour);
    ASSERT_EQ(m->trial.path.size(), sol.path.size());
    for (size_t i = 0; i < sol.path.size(); ++i)
        EXPECT_EQ(m->trial.path[i], sol.path[i]) << "segment " << i;
}

TEST_F(neighborhood_tsp_fixture, TspToPathReversedTourEndingAtRootRecoversPath){
    int dim = static_cast<int>(sol.path.size()) * 3 + 3;
    auto m = bind(sol);

    std::vector<int> tour(dim);
    std::iota(tour.begin(), tour.end(), 1);
    std::reverse(tour.begin(), tour.end());              // N, ..., 2, 1

    m->setTrialFromTspTour(tour);
    ASSERT_EQ(m->trial.path.size(), sol.path.size());
    for (size_t i = 0; i < sol.path.size(); ++i)
        EXPECT_EQ(m->trial.path[i], sol.path[i]) << "segment " << i;
}

TEST_F(neighborhood_tsp_fixture, WriteInitialTourEndsWithMinusOne){
    std::vector<int> warm_start = {1,2,3};
    engine.write_initial_tour(warm_start);
    std::ifstream f(tour_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("-1"), std::string::npos);
}


TEST_F(neighborhood_tsp_fixture, WriteInstanceFileContainsRequiredHeaders){
    Eigen::MatrixXi C(3,3);
    C.setConstant(std::numeric_limits<int>::max());
    engine.write_instance_file(C);
    std::ifstream f(tsp_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("NAME : "  + ctx.config.at("NAME")),       std::string::npos);
    EXPECT_NE(content.find("EDGE_WEIGHT_TYPE : EXPLICIT"), std::string::npos);
    EXPECT_NE(content.find("EDGE_WEIGHT_FORMAT : FULL_MATRIX"), std::string::npos);
    EXPECT_NE(content.find("EDGE_WEIGHT_SECTION"),     std::string::npos);
    EXPECT_NE(content.find("EOF"),                     std::string::npos);
}

TEST_F(neighborhood_tsp_fixture, WriteInstanceFileDimensionMatchesMatrix){
    Eigen::MatrixXi C(5,5);
    C.setConstant(0);
    engine.write_instance_file(C);
    std::ifstream f(tsp_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("DIMENSION : 5"), std::string::npos);
}

TEST_F(neighborhood_tsp_fixture, WriteInstanceFileSymmetricMatrixTypeIsTSP){
    Eigen::MatrixXi C(2,2);
    C << 0, 5, 5, 0;
    engine.write_instance_file(C);
    std::ifstream f(tsp_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("TYPE : TSP"), std::string::npos);
}

TEST_F(neighborhood_tsp_fixture, WriteInstanceFileDataRows){
    Eigen::MatrixXi C(2,2);
    C << 0, 7, 7, 0;
    engine.write_instance_file(C);
    std::ifstream f(tsp_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("0 7"), std::string::npos);
    EXPECT_NE(content.find("7 0"), std::string::npos);
}

//----------Path splitting testing--------------------------------------
// The split is applied to the trial solution; returns a copy of it.
static cppied_solution split_trial(neighborhood_tsp_accessor& m){
    m.splitTrial();
    return m.trial;
}

// split only adds nodes; the path must be at least as long as before.
TEST_F(neighborhood_tsp_fixture, SplitNeverReducesPathSize){
    size_t original_size = sol.path.size();
    cppied_solution split = split_trial(*bind(sol));
    EXPECT_GE(split.path.size(), original_size);
}

TEST_F(neighborhood_tsp_fixture, SplitVerticesWithinOriginalRange){
    int lo = std::numeric_limits<int>::max();
    int hi = std::numeric_limits<int>::min();
    for (auto& s : sol.path){
        lo = std::min(lo, s.source);
        hi = std::max(hi, path_engine::is_node(s.target) ? s.target : s.source);
    }
    cppied_solution split = split_trial(*bind(sol));
    for (auto& s : split.path){
        EXPECT_GE(s.source, lo);
        if (path_engine::is_node(s.target)) EXPECT_LE(s.target, hi);
    }
}

TEST_F(neighborhood_tsp_fixture, SplitSingleNodeSegmentsAreUnchanged){
    sol.path = {{6, path_engine::NULL_NODE}, {12, path_engine::NULL_NODE}, {18, path_engine::NULL_NODE}};
    size_t sz = sol.path.size();
    cppied_solution split = split_trial(*bind(sol));
    EXPECT_EQ(split.path.size(), sz);
}

TEST_F(neighborhood_tsp_fixture, SplitKeepPathTheSame){
    cppied_solution other = sol;
    n->geometry.complete(other);
    cppied_solution split = split_trial(*bind(sol));

    n->geometry.complete(split);

    ASSERT_EQ(n->geometry.cost(other), n->geometry.cost(split));
    ASSERT_EQ(other.path.size(), split.path.size());
    for (int i=0; i< other.path.size(); ++i)
        EXPECT_EQ(other.path[i], split.path[i]);
}

// ============================================================================
// End-to-end tests – require the LKH binary at LKH_EXECUTABLE (test_config.txt).
//
// Behavioural contract of neighborhood_tsp::local_search:
//   • The path is ALWAYS modified (split + TSP reorder) regardless of the
//     return value.
//   • On return TRUE  : geometry.complete + trim + cut were applied.
//                       pSol.cost is consistent with geometry.cost(pSol).
//                       pSol.cost < incumbent.
//   • On return FALSE : the path holds the TSP-reordered, non-completed
//                       segments; pSol.cost still holds the pre-call value.
// All fixtures that call local_search from a consistent state therefore need:
//   1. A completed solution (geometry.complete called) as starting point so
//      that the incumbent cost is meaningful.
//   2. ctx.start_time set so that the TIME_LIMIT written to the .par file
//      is non-negative.
//   3. The scratch directory present on disk.
// ============================================================================

class neighborhood_tsp_e2e_fixture : public neighborhood_tsp_fixture {
protected:
    void SetUp() override {
        neighborhood_tsp_fixture::SetUp();
        std::filesystem::create_directories(scratch_dir);
        // Anchor start_time so the generated TIME_LIMIT is sensible.
        ctx.start_time = std::chrono::high_resolution_clock::now();
        // Start from a complete, coverage-feasible solution.
        n->geometry.complete(sol);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

// ──── Smoke / safety ────────────────────────────────────────────────────────

// local_search must not throw on a valid, completed, coverage-feasible input.
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchDoesNotThrow) {
    EXPECT_NO_THROW(search(sol));
}

// The path must contain at least one segment after the call, regardless of
// whether an improvement was found.
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchPathNonEmptyAfterCall) {
    search(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

// ──── Invariants on improvement ─────────────────────────────────────────────

// When local_search returns true the tracked cost must be strictly less than
// the pre-call incumbent.
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchCostDecreasedWhenReturnTrue) {
    cost_t before = sol.cost;
    if (search(sol))
        EXPECT_LT(sol.cost, before);
}

// When local_search returns true the tracked cost must equal geometry.cost
// (trim and cut leave the cost field consistent after their own local_search
// loops both return false).
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchCostConsistentWhenReturnTrue) {
    if (search(sol))
        EXPECT_EQ(n->geometry.cost(sol), sol.cost);
}

// When local_search returns true the coverage vector stored in the solution
// must match a freshly computed reset (trim and cut update coverage
// incrementally; a full reset must agree).
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchCoverageConsistentWhenReturnTrue) {
    if (search(sol)) {
        Eigen::VectorXd saved = sol.coverage;
        n->coverage.reset(sol);
        EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9));
    }
}

// When local_search returns true the coverage constraint must still be
// satisfied (the initial feasible solution must stay feasible after trim+cut).
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchCoverageConstraintWhenReturnTrue) {
    if (search(sol)) {
        n->coverage.reset(sol);
        EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
    }
}

// ──── Convergence ────────────────────────────────────────────────────────────

// Repeated calls (each preceded by a complete+reset to re-establish a
// consistent starting state) must eventually stop improving. At the local
// optimum geometry.cost must match the tracked cost.
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchConvergesToLocalOptimum) {
    // Drive the search until no further improvement is reported.
    // After each call we re-complete the path and re-anchor the cost so that
    // the next call starts from a consistent state (local_search leaves the
    // path non-completed when it returns false).
    static constexpr int max_iters = 20;
    for (int i = 0; i < max_iters; ++i) {
        bool improved = search(sol);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
        if (!improved) break;
    }
    EXPECT_EQ(n->geometry.cost(sol), sol.cost);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

// ──── Suboptimal-ordering scenario ───────────────────────────────────────────

// Interleave the first and second halves of the path so that spatially
// adjacent segments are no longer consecutive: the resulting completed path
// should have higher cost than the original optimal ordering.  LKH should
// then be able to recover a better ordering.
//
// If the scrambled cost happens to equal the original (fully symmetric
// geometry), the improvement assertion is skipped to avoid a false failure.
TEST_F(neighborhood_tsp_e2e_fixture, E2ECurrentPathLengthValidUnderScrambledCostMatrix){
    if (sol.path.size() < 4)
        GTEST_SKIP() << "Path too short for a meaningful scramble test";

    // Interleave back-half and front-half segments.
    std::vector<segment> scrambled;
    scrambled.reserve(sol.path.size());
    std::map<int, int> unscramble;
    const std::size_t half = sol.path.size() / 2;
    scrambled.push_back(sol.path.front());
    unscramble.insert({0, 0});
    scrambled.push_back(sol.path[half]);
    unscramble.insert({half, 1});
    for (std::size_t k = 1; k < half; ++k) {
        scrambled.push_back(sol.path[half + k]);
        unscramble.insert({half + k, 2*k});
        scrambled.push_back(sol.path[k]);
        unscramble.insert({k, 2*k + 1});
    }
    if (sol.path.size() % 2 != 0) {
        scrambled.push_back(sol.path.back());
        unscramble.insert({sol.path.size()-1, sol.path.size()-1});
    }

    sol.path = std::move(scrambled);
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);

    int dim = static_cast<int>(sol.path.size()) * 3 + 3;
    Eigen::MatrixXi C = distances(sol);
    ASSERT_EQ(C.rows(), dim);

    int acc = 0;
    for (int i=0; i< sol.path.size()-1; i++){
       acc += n->geometry.cost(sol.path[unscramble[i]]).length;
       acc += C((unscramble[i]+1)*3, (unscramble[i]+1)*3 + 1) +
               C((unscramble[i]+1)*3 + 1, (unscramble[i]+1)*3 + 2) +
               C((unscramble[i]+1)*3 + 2, (unscramble[i+1]+1)*3);
    }
    acc += n->geometry.cost(sol.path[unscramble[int(sol.path.size())-1]]).length;
    EXPECT_EQ(acc, 39);
}

TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchImprovesScrambledOrdering) {
    if (sol.path.size() < 4)
        GTEST_SKIP() << "Path too short for a meaningful scramble test";
    cost_t original_cost = sol.cost;

    // Interleave back-half and front-half segments.
    std::vector<segment> scrambled;
    scrambled.reserve(sol.path.size());
    const std::size_t half = sol.path.size() / 2;
    scrambled.push_back(sol.path.front());
    scrambled.push_back(sol.path[half]);
    for (std::size_t k = 1; k < half; ++k) {
        scrambled.push_back(sol.path[half + k]);
        scrambled.push_back(sol.path[k]);
    }
    if (sol.path.size() % 2 != 0)
        scrambled.push_back(sol.path.back());

    sol.path = std::move(scrambled);
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);

    // The scrambled solution must still satisfy the coverage constraint.
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());

    if (sol.cost.length <= original_cost.length)
        GTEST_SKIP() << "Scrambled ordering did not increase cost; "
                        "improvement test is inconclusive for this geometry";

    cost_t scrambled_cost = sol.cost;
    bool improved = search(sol);

    EXPECT_TRUE(improved)
        << "Expected TSP neighborhood to improve a suboptimal segment ordering";
    if (improved)
        EXPECT_LT(sol.cost, scrambled_cost);
}

// ──── State after non-improving call ─────────────────────────────────────────

// When local_search returns false the path is non-empty (the TSP reordering
// always produces a non-empty path for a non-empty input) and can be
// re-completed without throwing.
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchPathRecoverableAfterNoImprovement) {
    // Run until the first non-improving call (or until max_iters if the
    // neighborhood keeps improving, which is also a valid outcome).
    static constexpr int max_iters = 10;
    bool last_result = true;
    for (int i = 0; i < max_iters && last_result; ++i) {
        last_result = search(sol);
        if (last_result) {
            n->coverage.reset(sol);
            sol.cost = n->geometry.cost(sol);
        }
    }

    // Whether or not the loop reached a non-improving call, the path must be
    // non-empty and completable without exception.
    ASSERT_GT(sol.path.size(), 0u);
    EXPECT_NO_THROW({
        n->geometry.complete(sol);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    });
    EXPECT_EQ(n->geometry.cost(sol), sol.cost);
}

// The neighborhood is bound (and its trial seeded) before the solution
// changes, as in VND where earlier neighborhoods modify pSol. The current
// solution is the bound one plus a single-node segment on a vertex the bound
// path never visits, so a result built from the stale trial would be
// cheaper, accepted, and miss that vertex.
// Regression: trial was copied at construction (stale).
TEST_F(neighborhood_tsp_e2e_fixture, LocalSearchUsesCurrentSolution){
    auto visits = [&](const std::vector<segment>& path, int v){
        return std::any_of(path.begin(), path.end(), [&](const segment& s){
            segment u = n->geometry.getNonOriented(s);
            int last = path_engine::is_node(u.target) ? u.target : u.source;
            return u.source <= v && v <= last && n->geometry.is_horizontal(u) == (v < n->geometry.horizontal_bound);
        });
    };
    int extra = -1;
    for (int v = 0; v < n->geometry.horizontal_bound && extra < 0; ++v)
        if (!visits(sol.path, v)) extra = v;
    ASSERT_GE(extra, 0) << "the bound path visits every horizontal vertex";

    auto m = bind(sol);                           // trial = completed fixture path
    sol.path.push_back({extra, path_engine::NULL_NODE});
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);
    m->local_search();
    EXPECT_TRUE(visits(sol.path, extra))
        << "result does not derive from the current solution (vertex " << extra << " lost)";
    cppied_solution fresh = sol;
    n->coverage.reset(fresh);
    EXPECT_TRUE((fresh.coverage.array() >= P.req.array()).all());
}

//----------Special cases where bugs were encountered----------------
TEST_F(neighborhood_tsp_e2e_fixture, SpecialCaseDoesNotIncreaseCost){
    sol.path = {{6, -1},
    {49, -1},
    {12, -2},
    {44, 45},
    {24, -1},
    {51, 50},
    {13, -1},
    {56, 57},
    {26, -1},
    {63, -2},
    {21, 23},
    {35, 30},
    {46, -2},
    {24, -1},
    {48, -2},
    {0, -2},
    {42, -1},
    {6, 11},
    {78, -2},
    {5, -2},
    {72, 75},
    {28, 27},
    {62, -2}};
    sol.cost = n->geometry.cost(sol);
    n->coverage.reset(sol);
    cost_t c = sol.cost;
    search(sol);
    EXPECT_LE(sol.cost, c);
    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

TEST_F(neighborhood_tsp_e2e_fixture, SpecialCaseTwoDoesNotIncreaseCost){
    sol.path = {{6, -1},
            {49, -1},
            {12, -2},
            {44, -1},
            {18, -1},
            {50, -2},
            {12, -2},
            {44, -1},
            {18, 23},
            {81, 82},
            {35, 30},
            {46, -2},
            {24, -1},
            {51, 48},
            {0, -2},
            {42, -1},
            {6, 11},
            {78, -2},
            {5, -2},
            {72, 75},
            {63, 62}};
    sol.cost = n->geometry.cost(sol);
    n->coverage.reset(sol);
    cost_t c = sol.cost;
    search(sol);
    EXPECT_LE(sol.cost, c);
    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}



//----------Unit tests of the TSP construction helpers----------------

// The warm-start tour handed to LKH is a permutation of the 1-based node ids
// 1..dimension (LKH numbers nodes from 1).
TEST_F(neighborhood_tsp_fixture, BuildTspInstanceWarmStartIsOneBasedPermutation){
    auto m = bind(sol);
    TspInstance instance = m->buildTspInstance();
    const int dim = static_cast<int>(sol.path.size()) * 3 + 3;
    ASSERT_EQ(instance.distances.rows(), dim);
    std::vector<int> expected(dim);
    std::iota(expected.begin(), expected.end(), 1);
    EXPECT_EQ(instance.tour, expected);
}

// floor(sqrt(n)) segments are selected for splitting, sorted by path index,
// and no unselected segment is strictly costlier than a selected one.
TEST_F(neighborhood_tsp_fixture, SelectSplitIndexesPicksCostliestSorted){
    auto m = bind(sol);
    std::vector<int> sel = m->selectSplitIndexes();
    ASSERT_EQ(sel.size(), static_cast<size_t>(std::sqrt(static_cast<double>(sol.path.size()))));
    EXPECT_TRUE(std::is_sorted(sel.begin(), sel.end()));
    cost_t cheapest_selected = {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()};
    for (int i : sel) cheapest_selected = std::min(cheapest_selected, n->geometry.cost(sol.path[i]));
    for (int i = 0; i < static_cast<int>(sol.path.size()); ++i)
        if (std::find(sel.begin(), sel.end(), i) == sel.end())
            EXPECT_LE(n->geometry.cost(sol.path[i]), cheapest_selected) << "segment " << i;
}

TEST_F(neighborhood_tsp_fixture, SelectSplitIndexesOnTinyPaths){
    for (size_t size : {1u, 2u, 3u, 4u}) {
        cppied_solution s = sol;
        s.path.resize(size);
        auto m = bind(s);
        auto sel = m->selectSplitIndexes();
        EXPECT_EQ(sel.size(), static_cast<size_t>(std::sqrt(static_cast<double>(size)))) << "size " << size;
        for (int i : sel) EXPECT_LT(i, static_cast<int>(size));
    }
}

// The split index is a valid cut position: [0, length - 1].
TEST_F(neighborhood_tsp_fixture, GetSplitIndexInRange){
    auto m = bind(sol);
    for (const segment& s : sol.path) {
        const int length = n->geometry.cost(s).length;
        for (int draw = 0; draw < 100; ++draw) {
            int index = m->getSplitIndex(s);
            EXPECT_GE(index, 0);
            EXPECT_LE(index, std::max(0, length - 1)) << "segment {" << s.source << "," << s.target << "}";
        }
    }
    segment single = {6, path_engine::NULL_NODE};
    EXPECT_EQ(m->getSplitIndex(single), 0);
}

// Splitting adds exactly one segment per selected multi-node segment, keeps
// every other segment as is, and never splits a single node.
TEST_F(neighborhood_tsp_fixture, ConstructSplitPathSplitsOnlySelected){
    sol.path = {{6, 11}, {23, 18}, {44, path_engine::NULL_NODE}, {48, 51}};
    auto m = bind(sol);
    std::vector<int> sel = {1, 2};
    auto path = m->constructSplitPath(sel);
    ASSERT_EQ(path.size(), sol.path.size() + 1) << "{44} must not be split";
    EXPECT_EQ(path[0], sol.path[0]);
    EXPECT_EQ(path[1].source, 23);
    EXPECT_EQ(path[3], sol.path[2]);
    EXPECT_EQ(path[4], sol.path[3]);
    std::vector<segment> pushed;
    m->splitAndPush(pushed, {44, path_engine::NULL_NODE});
    EXPECT_EQ(pushed, (std::vector<segment>{{44, path_engine::NULL_NODE}}));
}

// Tour ids: 3k+4 is the entry of path[k] (segment as is), 3k+6 its exit
// (segment reversed), 3k+5 its middle node, which a valid tour never puts at
// a group start.
TEST_F(neighborhood_tsp_fixture, TourIdToSegmentMapping){
    auto m = bind(sol);
    for (int k = 0; k < static_cast<int>(sol.path.size()); ++k) {
        EXPECT_EQ(m->tour_id_to_segment(3 * k + 4), sol.path[k]) << "entry of " << k;
        EXPECT_EQ(m->tour_id_to_segment(3 * k + 6), n->geometry.flip_segment(sol.path[k])) << "exit of " << k;
        EXPECT_THROW(m->tour_id_to_segment(3 * k + 5), std::runtime_error) << "middle of " << k;
    }
}
