#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood_tsp.hpp"

struct neighborhood_tsp_accessor : public neighborhood_tsp {
public:
    using neighborhood_tsp::neighborhood_tsp;

    using neighborhood::geometry;
    using neighborhood::coverage;
    using neighborhood::problem;

    using neighborhood_tsp::tspEngine;       // protected → exposed for testing
};

class neighborhood_tsp_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<neighborhood_tsp_accessor> n;
    cppied_solution sol;

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
        n = std::make_unique<neighborhood_tsp_accessor>(ctx, P);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

//----------Cost matrix testing--------------------------------------
TEST_F(neighborhood_tsp_fixture, BuildSymmetricMatrix){
    Eigen::MatrixXi C(24,24);
    C.setConstant(std::numeric_limits<int>::max());
    n->path_to_tsp(sol, C);
    EXPECT_TRUE(C.isApprox(C.transpose()));
}

TEST_F(neighborhood_tsp_fixture, BuildCostMatrixNonEdgesDummyAreIntMax){
    Eigen::MatrixXi C(24,24);
    C.setConstant(std::numeric_limits<int>::max());
    n->path_to_tsp(sol, C);
    for (int i = 1; i< 24; i+=3){
        EXPECT_EQ((C.row(i).array() == 0).count(), 2);
        EXPECT_EQ((C.col(i).array() == 0).count(), 2);
    }
}

TEST_F(neighborhood_tsp_fixture, BuildCostMatrixNonEdgesToDummyAreIntMax){
    Eigen::MatrixXi C(24,24);
    C.setConstant(std::numeric_limits<int>::max());
    n->path_to_tsp(sol, C);
    for (int i = 1; i< 24; i+=3){
        for (int j=0; j<24; ++j){
            if (abs(i-j) > 1)
                EXPECT_EQ(C(i,j), std::numeric_limits<int>::max());
        }
    }
}

TEST_F(neighborhood_tsp_fixture, BuildCostMatrixDiagonalIsIntMax){
    Eigen::MatrixXi C(24,24);
    C.setConstant(std::numeric_limits<int>::max());
    n->path_to_tsp(sol, C);
    EXPECT_EQ((C.diagonal().array() == std::numeric_limits<int>::max()).count(), 24);
}

TEST_F(neighborhood_tsp_fixture, BuildCostMatrixCostSelfSegmentIsIntMax){
    Eigen::MatrixXi C(24,24);
    C.setConstant(std::numeric_limits<int>::max());
    n->path_to_tsp(sol, C);
    for (int i=0; i< 24; i+=3){
        EXPECT_EQ(C(i, i+2), std::numeric_limits<int>::max());
    }
}

TEST_F(neighborhood_tsp_fixture, BuildCostMatrixCostTotargetIsZero){
    Eigen::MatrixXi C(24,24);
    C.setConstant(std::numeric_limits<int>::max());
    n->path_to_tsp(sol, C);
    for (int i=6; i< 24; i+=3){
        EXPECT_EQ(C(i, 0), 0);
        EXPECT_EQ(C(i+2, 0), 0);
    }
}

TEST_F(neighborhood_tsp_fixture, BuildCostMatrixValidCurrentPath){
    Eigen::MatrixXi C(24,24);
    C.setConstant(std::numeric_limits<int>::max());
    n->path_to_tsp(sol, C);
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
    n->tspEngine.write_instance_file(C);
    EXPECT_TRUE(std::filesystem::exists(tsp_file()));
}

TEST_F(neighborhood_tsp_fixture, WriteInitialTourIsCreated){
    std::vector<int> warm_start = {1,2,3,4,5};
    n->tspEngine.write_initial_tour(warm_start);
    EXPECT_TRUE(std::filesystem::exists(tour_file()));
}

TEST_F(neighborhood_tsp_fixture, WriteInitialTourContainsRequiredHeaders){
    std::vector<int> warm_start = {1,2,3};
    n->tspEngine.write_initial_tour(warm_start);
    std::ifstream f(tour_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("NAME : "  + ctx.config.at("NAME")), std::string::npos);
    EXPECT_NE(content.find("TYPE : TOUR"),         std::string::npos);
    EXPECT_NE(content.find("TOUR_SECTION"),        std::string::npos);
}

TEST_F(neighborhood_tsp_fixture, WriteInitialTourDimensionMatchesTourSize){
    std::vector<int> warm_start = {1,2,3,4};
    n->tspEngine.write_initial_tour(warm_start);
    std::ifstream f(tour_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("DIMENSION : 4"), std::string::npos);
}

TEST_F(neighborhood_tsp_fixture, WriteInitialTourContainsAllNodeIds){
    std::vector<int> warm_start = {1,2,3,4,5};
    n->tspEngine.write_initial_tour(warm_start);
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
    n->tspEngine.read_solution(tour);
    ASSERT_EQ(tour.size(), 5u);
}

TEST_F(neighborhood_tsp_fixture, ReadSolutionCorrectNodeValues){
    write_test_sol(sol_file(), {3,1,4,2});
    std::vector<int> tour;
    n->tspEngine.read_solution(tour);
    ASSERT_EQ(tour.size(), 4u);
    EXPECT_EQ(tour[0], 3);
    EXPECT_EQ(tour[1], 1);
    EXPECT_EQ(tour[2], 4);
    EXPECT_EQ(tour[3], 2);
}

TEST_F(neighborhood_tsp_fixture, ReadSolutionThrowsWhenFileAbsent){
    std::filesystem::remove(sol_file());
    std::vector<int> tour;
    EXPECT_THROW(n->tspEngine.read_solution(tour), std::runtime_error);
}

TEST_F(neighborhood_tsp_fixture, ReadSolutionThrowsWhenDimensionMissing){
    std::ofstream f(sol_file());
    f << "NAME : bad\nTYPE : TOUR\nTOUR_SECTION\n1\n2\n-1\n";
    f.close();
    std::vector<int> tour;
    EXPECT_THROW(n->tspEngine.read_solution(tour), std::runtime_error);
}

TEST_F(neighborhood_tsp_fixture, TspToPathRoundTripPathSize){
    std::vector<tsp::edge> instance;
    std::vector<segment> tsp_map;
    int dim = static_cast<int>(sol.path.size()) * 3 + 3;
    Eigen::MatrixXi C(dim, dim);
    n->path_to_tsp(sol, C);

    std::vector<int> tour(dim);
    std::iota(tour.begin(), tour.end(), 1);

    cppied_solution result = sol;
    n->tsp_to_path(result, tour);
    EXPECT_EQ(result.path.size(), sol.path.size());
}

TEST_F(neighborhood_tsp_fixture, TspToPathRoundTripFirstVertex){
    std::vector<tsp::edge> instance;
    std::vector<segment> tsp_map;
    int dim = static_cast<int>(sol.path.size()) * 3 + 3;
    Eigen::MatrixXi C(dim, dim);
    n->path_to_tsp(sol, C);

    std::vector<int> tour(dim);
    std::iota(tour.begin(), tour.end(), 1);

    cppied_solution result = sol;
    n->tsp_to_path(result, tour);
    EXPECT_EQ(result.path[0].source, sol.path[0].source);
}

TEST_F(neighborhood_tsp_fixture, WriteInitialTourEndsWithMinusOne){
    std::vector<int> warm_start = {1,2,3};
    n->tspEngine.write_initial_tour(warm_start);
    std::ifstream f(tour_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("-1"), std::string::npos);
}


TEST_F(neighborhood_tsp_fixture, WriteInstanceFileContainsRequiredHeaders){
    Eigen::MatrixXi C(3,3);
    C.setConstant(std::numeric_limits<int>::max());
    n->tspEngine.write_instance_file(C);
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
    n->tspEngine.write_instance_file(C);
    std::ifstream f(tsp_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("DIMENSION : 5"), std::string::npos);
}

TEST_F(neighborhood_tsp_fixture, WriteInstanceFileSymmetricMatrixTypeIsTSP){
    Eigen::MatrixXi C(2,2);
    C << 0, 5, 5, 0;
    n->tspEngine.write_instance_file(C);
    std::ifstream f(tsp_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("TYPE : TSP"), std::string::npos);
}

TEST_F(neighborhood_tsp_fixture, WriteInstanceFileDataRows){
    Eigen::MatrixXi C(2,2);
    C << 0, 7, 7, 0;
    n->tspEngine.write_instance_file(C);
    std::ifstream f(tsp_file());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_NE(content.find("0 7"), std::string::npos);
    EXPECT_NE(content.find("7 0"), std::string::npos);
}

//----------Path splitting testing--------------------------------------
// split only adds nodes; the path must be at least as long as before.
TEST_F(neighborhood_tsp_fixture, SplitNeverReducesPathSize){
    size_t original_size = sol.path.size();
    n->split(sol);
    EXPECT_GE(sol.path.size(), original_size);
}

TEST_F(neighborhood_tsp_fixture, SplitVerticesWithinOriginalRange){
    int lo = std::numeric_limits<int>::max();
    int hi = std::numeric_limits<int>::min();
    for (auto& s : sol.path){
        lo = std::min(lo, s.source);
        hi = std::max(hi, path_engine::is_node(s.target) ? s.target : s.source);
    }
    n->split(sol);
    for (auto& s : sol.path){
        EXPECT_GE(s.source, lo);
        if (path_engine::is_node(s.target)) EXPECT_LE(s.target, hi);
    }
}

TEST_F(neighborhood_tsp_fixture, SplitSingleNodeSegmentsAreUnchanged){
    sol.path = {{6, path_engine::NULL_NODE}, {12, path_engine::NULL_NODE}, {18, path_engine::NULL_NODE}};
    size_t sz = sol.path.size();
    n->split(sol);
    EXPECT_EQ(sol.path.size(), sz);
}

TEST_F(neighborhood_tsp_fixture, SplitKeepPathTheSame){
    cppied_solution other = sol;
    n->geometry.complete(other);
    n->split(sol);

    n->geometry.complete(sol);

    ASSERT_EQ(n->geometry.cost(other), n->geometry.cost(sol));
    ASSERT_EQ(other.path.size(), sol.path.size());
    for (int i=0; i< other.path.size(); ++i)
        EXPECT_EQ(other.path[i], sol.path[i]);
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
    EXPECT_NO_THROW(n->local_search(sol));
}

// The path must contain at least one segment after the call, regardless of
// whether an improvement was found.
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchPathNonEmptyAfterCall) {
    n->local_search(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

// ──── Invariants on improvement ─────────────────────────────────────────────

// When local_search returns true the tracked cost must be strictly less than
// the pre-call incumbent.
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchCostDecreasedWhenReturnTrue) {
    cost_t before = sol.cost;
    if (n->local_search(sol))
        EXPECT_LT(sol.cost, before);
}

// When local_search returns true the tracked cost must equal geometry.cost
// (trim and cut leave the cost field consistent after their own local_search
// loops both return false).
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchCostConsistentWhenReturnTrue) {
    if (n->local_search(sol))
        EXPECT_EQ(n->geometry.cost(sol), sol.cost);
}

// When local_search returns true the coverage vector stored in the solution
// must match a freshly computed reset (trim and cut update coverage
// incrementally; a full reset must agree).
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchCoverageConsistentWhenReturnTrue) {
    if (n->local_search(sol)) {
        Eigen::VectorXd saved = sol.coverage;
        n->coverage.reset(sol);
        EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9));
    }
}

// When local_search returns true the coverage constraint must still be
// satisfied (the initial feasible solution must stay feasible after trim+cut).
TEST_F(neighborhood_tsp_e2e_fixture, E2ELocalSearchCoverageConstraintWhenReturnTrue) {
    if (n->local_search(sol)) {
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
        bool improved = n->local_search(sol);
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
    Eigen::MatrixXi C(dim, dim);
    C.setConstant(39);
    n->path_to_tsp(sol, C);

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

    if (sol.cost <= original_cost)
        GTEST_SKIP() << "Scrambled ordering did not increase cost; "
                        "improvement test is inconclusive for this geometry";

    cost_t scrambled_cost = sol.cost;
    bool improved = n->local_search(sol);

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
        last_result = n->local_search(sol);
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
    n->local_search(sol);
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
    n->local_search(sol);
    EXPECT_LE(sol.cost, c);
    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}


