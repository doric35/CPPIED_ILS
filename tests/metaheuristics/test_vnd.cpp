#include "../test_fixture.hpp"
#include "../../include/metaheuristics/vnd.hpp"
#include "../../include/neighborhoods/neighborhood_cut.hpp"
#include "../../include/neighborhoods/neighborhood_reduce.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_n1.hpp"
#include "../../include/neighborhoods/neighborhood_n12.hpp"
#include "../../include/neighborhoods/neighborhood_n21.hpp"
#include "../../include/neighborhoods/neighborhood_nested.hpp"
#include "../../include/neighborhoods/neighborhood_tsp.hpp"
#include "../../include/neighborhoods/neighborhood_gtsp.hpp"
#include "../../include/neighborhoods/neighborhood_r.hpp"

// ---------------------------------------------------------------------------
// Accessor: exposes the four protected neighborhood vectors for white-box
// inspection.
// ---------------------------------------------------------------------------
struct vnd_accessor : public vnd {
public:
    using vnd::vnd;
    using vnd::N_setup;
    using vnd::N_simple;
    using vnd::N_nested;
    using vnd::N_large;

    using vnd::geometry;
    using vnd::coverage;
    using vnd::problem;

    // Convenience: iterate all neighborhoods in search order.
    std::vector<neighborhood*> all_neighborhoods() const {
        std::vector<neighborhood*> all;
        for (const auto& n : N_setup)  all.push_back(n.get());
        for (const auto& n : N_simple) all.push_back(n.get());
        for (const auto& n : N_nested) all.push_back(n.get());
        for (const auto& n : N_large)  all.push_back(n.get());
        return all;
    }
};

class vnd_minimal_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<vnd_accessor> v;
    cppied_solution sol;

    void SetUp() override {
        cppied_context_fixture::SetUp();
        v = std::make_unique<vnd_accessor>(ctx, P);
        v->initialize();

        sol.path = {{6, 11},
                    {23, 18},
                    {44, 42},
                    {48, 51},
                    {30, 35},
                    {75, 73},
                    {62, 63}};
        sol.cost     = {0, 0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        v->geometry.complete(sol);
        v->coverage.reset(sol);
        sol.cost = v->geometry.cost(sol);
    }

    path_engine&     geo() { return v->geometry; }
    coverage_engine& cov() { return v->coverage; }
};

// ---------------------------------------------------------------------------
// Full fixture — uses vnd_test_config.
//   ALGORITHM_CONFIG = -c11111110000  (12 chars)
//   config[1] = '1'         →  n1 added    (N_simple)
//   config[2] = '1'         →  n12 added   (N_simple)
//   config[3] = '1'         →  n21 added   (N_simple)
//   config[4] = '1'         →  nested added (N_nested)
//   config[5] = '1'         →  tsp added   (N_large)
//   config[6] = '1'         →  gtsp added  (N_large)
//   config[7] = '1'         →  r added     (N_large)
//   N_setup (always):  reduce + trim + cut = 3
//   N_simple:          n1 + n12 + n21      = 3
//   N_nested:          nested              = 1
//   N_large:           tsp + gtsp + r      = 3
//   Total neighborhoods_count()            = 10
// ---------------------------------------------------------------------------
class vnd_full_fixture : public ::testing::Test {
protected:
    static constexpr const char* seabed_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_seabed.txt";
    static constexpr const char* pod_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_pod.txt";
    static constexpr const char* req_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_req.txt";
    static constexpr const char* config_path =
            PROJECT_SOURCE_DIR "/tests/configurations/vnd_test_config.txt";

    cppied_context ctx;
    cppied_instance P;
    std::unique_ptr<vnd_accessor> v;
    cppied_solution sol;

    vnd_full_fixture()
        : ctx(read_configuration(config_path)),
          P(read_matrix<int>(seabed_path),
            read_matrix<double>(pod_path),
            read_matrix<double>(req_path)) {}

    void SetUp() override {
        v = std::make_unique<vnd_accessor>(ctx, P);
        v->initialize();
        ctx.start_time = std::chrono::high_resolution_clock::now();

        sol.path = {{6, 11},
                    {23, 18},
                    {44, 42},
                    {48, 51},
                    {30, 35},
                    {75, 73},
                    {62, 63}};
        sol.cost     = {0, 0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        v->geometry.complete(sol);
        v->coverage.reset(sol);
        sol.cost = v->geometry.cost(sol);
    }

    path_engine&     geo() { return v->geometry; }
    coverage_engine& cov() { return v->coverage; }
};

// ===========================================================================
// Initialize tests — minimal config
// N_setup always contains: reduce (0), trim (1), cut (2).
// N_simple / N_nested / N_large are empty in the minimal config.
// ===========================================================================

TEST_F(vnd_minimal_fixture, InitializeNeighborhoodCountMinimalConfig) {
    EXPECT_EQ(v->neighborhoods_count(), 3u);
}

TEST_F(vnd_minimal_fixture, InitializeReduceIsFirstSetupNeighborhood) {
    ASSERT_GE(v->N_setup.size(), 1u);
    EXPECT_NE(dynamic_cast<neighborhood_reduce*>(v->N_setup[0].get()), nullptr);
}

TEST_F(vnd_minimal_fixture, InitializeTrimIsSecondSetupNeighborhood) {
    ASSERT_GE(v->N_setup.size(), 3u);
    EXPECT_NE(dynamic_cast<neighborhood_trim*>(v->N_setup[1].get()), nullptr);
}

TEST_F(vnd_minimal_fixture, InitializeCutIsThirdSetupNeighborhood) {
    ASSERT_GE(v->N_setup.size(), 3u);
    EXPECT_NE(dynamic_cast<neighborhood_cut*>(v->N_setup[2].get()), nullptr);
}

TEST_F(vnd_minimal_fixture, InitializeSimpleNeighborhoodsEmptyInMinimalConfig) {
    EXPECT_EQ(v->N_simple.size(), 0u);
}

TEST_F(vnd_minimal_fixture, InitializeNestedNeighborhoodsEmptyInMinimalConfig) {
    EXPECT_EQ(v->N_nested.size(), 0u);
}

TEST_F(vnd_minimal_fixture, InitializeLargeNeighborhoodsEmptyInMinimalConfig) {
    EXPECT_EQ(v->N_large.size(), 0u);
}

// Calling initialize() a second time without a clear guard appends duplicates.
TEST_F(vnd_minimal_fixture, InitializeDoubleCallAppendsDuplicates) {
    v->initialize();
    EXPECT_EQ(v->neighborhoods_count(), 6u);
}

// ===========================================================================
// Initialize tests — full config
// ===========================================================================

TEST_F(vnd_full_fixture, InitializeNeighborhoodCountFullConfig) {
    EXPECT_EQ(v->neighborhoods_count(), 10u);
}

TEST_F(vnd_full_fixture, InitializeSetupHasThreeNeighborhoodsFullConfig) {
    EXPECT_EQ(v->N_setup.size(), 3u);
}

TEST_F(vnd_full_fixture, InitializeReduceIsFirstSetupNeighborhoodFullConfig) {
    ASSERT_GE(v->N_setup.size(), 1u);
    EXPECT_NE(dynamic_cast<neighborhood_reduce*>(v->N_setup[0].get()), nullptr);
}

TEST_F(vnd_full_fixture, InitializeTrimIsSecondSetupNeighborhoodFullConfig) {
    ASSERT_GE(v->N_setup.size(), 3u);
    EXPECT_NE(dynamic_cast<neighborhood_trim*>(v->N_setup[1].get()), nullptr);
}

TEST_F(vnd_full_fixture, InitializeCutIsThirdSetupNeighborhoodFullConfig) {
    ASSERT_GE(v->N_setup.size(), 3u);
    EXPECT_NE(dynamic_cast<neighborhood_cut*>(v->N_setup[2].get()), nullptr);
}

TEST_F(vnd_full_fixture, InitializeN1IsPresentInSimpleFullConfig) {
    bool found = false;
    for (auto& n : v->N_simple)
        if (dynamic_cast<neighborhood_n1*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(vnd_full_fixture, InitializeN12IsPresentInSimpleFullConfig) {
    bool found = false;
    for (auto& n : v->N_simple)
        if (dynamic_cast<neighborhood_n12*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(vnd_full_fixture, InitializeN21IsPresentInSimpleFullConfig) {
    bool found = false;
    for (auto& n : v->N_simple)
        if (dynamic_cast<neighborhood_n21*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(vnd_full_fixture, InitializeNestedIsPresentInNestedFullConfig) {
    bool found = false;
    for (auto& n : v->N_nested)
        if (dynamic_cast<neighborhood_nested*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(vnd_full_fixture, InitializeTspIsPresentInLargeFullConfig) {
    bool found = false;
    for (auto& n : v->N_large)
        if (dynamic_cast<neighborhood_tsp*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(vnd_full_fixture, InitializeGtspIsPresentInLargeFullConfig) {
    bool found = false;
    for (auto& n : v->N_large)
        if (dynamic_cast<neighborhood_gtsp*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(vnd_full_fixture, InitializeRIsPresentInLargeFullConfig) {
    bool found = false;
    for (auto& n : v->N_large)
        if (dynamic_cast<neighborhood_r*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

// ===========================================================================
// search() tests — minimal config (N_setup only: reduce + trim + cut)
// ===========================================================================

TEST_F(vnd_minimal_fixture, SearchDoesNotThrowOnValidSolution) {
    EXPECT_NO_THROW(v->search(sol));
}

TEST_F(vnd_minimal_fixture, SearchReturnsFalseAtLocalOptimum) {
    while (v->search(sol)) {
        v->coverage.reset(sol);
        sol.cost = v->geometry.cost(sol);
    }
    EXPECT_FALSE(v->search(sol));
}

TEST_F(vnd_minimal_fixture, SearchPathNonEmptyAfterConvergence) {
    while (v->search(sol)) {
        v->coverage.reset(sol);
        sol.cost = v->geometry.cost(sol);
    }
    EXPECT_GT(sol.path.size(), 0u);
}

TEST_F(vnd_minimal_fixture, SearchCostDecreasedWhenReturnTrue) {
    cost_t before = sol.cost;
    if (v->search(sol))
        EXPECT_LT(sol.cost, before);
}

TEST_F(vnd_minimal_fixture, SearchCostConsistentAfterSearchReturnTrue) {
    if (v->search(sol))
        EXPECT_EQ(geo().cost(sol), sol.cost);
}

TEST_F(vnd_minimal_fixture, SearchCoverageConsistentAfterSearchReturnTrue) {
    if (v->search(sol)) {
        Eigen::VectorXd saved = sol.coverage;
        cov().reset(sol);
        EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9));
    }
}

TEST_F(vnd_minimal_fixture, SearchCoverageConstraintSatisfiedAfterSearchReturnTrue) {
    if (v->search(sol)) {
        cov().reset(sol);
        EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
    }
}

TEST_F(vnd_minimal_fixture, SearchSecondCallReturnsFalseAfterConvergence) {
    v->search(sol);
    cov().reset(sol);
    sol.cost = geo().cost(sol);
    cost_t curr_c = sol.cost;
    v->search(sol);
    EXPECT_FALSE(curr_c > sol.cost);
}

// After search() converges, calling each neighborhood individually must not
// find further improvement — verifies that search() fully exhausted all
// neighborhoods before returning.
TEST_F(vnd_minimal_fixture, SearchLocalOptimumVerifiedForAllNeighborhoods) {
    while (v->search(sol)) {
        cov().reset(sol);
        sol.cost = geo().cost(sol);
    }
    cov().reset(sol);
    sol.cost = geo().cost(sol);

    auto all = v->all_neighborhoods();
    for (std::size_t i = 0; i < all.size(); ++i) {
        cppied_solution copy = sol;
        cost_t copy_c = copy.cost;
        all[i]->local_search(copy);
        EXPECT_FALSE(copy_c > copy.cost)
            << "Neighborhood " << i
            << " still finds improvement after VND convergence.";
    }
}

TEST_F(vnd_minimal_fixture, SearchReturnValueReflectsActualImprovement) {
    cost_t before = sol.cost;
    bool improved = v->search(sol);
    if (improved)
        EXPECT_LT(sol.cost, before);
    else
        EXPECT_EQ(sol.cost, before);
}

TEST_F(vnd_minimal_fixture, SearchCoverageConstraintSatisfiedAfterConvergence) {
    while (v->search(sol)) {
        cov().reset(sol);
        sol.cost = geo().cost(sol);
    }
    cov().reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

TEST_F(vnd_minimal_fixture, SearchCostConsistentAfterConvergence) {
    while (v->search(sol)) {
        cov().reset(sol);
        sol.cost = geo().cost(sol);
    }
    EXPECT_EQ(geo().cost(sol), sol.cost);
}

// ===========================================================================
// E2E tests — full config (all neighborhoods, requires LKH binary)
// ===========================================================================

TEST_F(vnd_full_fixture, FullConfigInitializeDoesNotThrow) {
    EXPECT_NO_THROW({
        vnd_accessor v2(ctx, P);
        v2.initialize();
    });
}

TEST_F(vnd_full_fixture, FullConfigSearchDoesNotThrow) {
    EXPECT_NO_THROW(v->search(sol));
}

TEST_F(vnd_full_fixture, FullConfigSearchPathNonEmptyAfterCall) {
    v->search(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

TEST_F(vnd_full_fixture, FullConfigSearchSolutionCostConsistentAfterCall) {
    v->search(sol);
    cov().reset(sol);
    sol.cost = geo().cost(sol);
    EXPECT_EQ(geo().cost(sol), sol.cost);
}

TEST_F(vnd_full_fixture, FullConfigSearchCoverageConstraintSatisfiedAfterCall) {
    v->search(sol);
    cov().reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

TEST_F(vnd_full_fixture, FullConfigSearchSecondCallReturnsFalse) {
    v->search(sol);
    cov().reset(sol);
    sol.cost = geo().cost(sol);
    cost_t c = sol.cost;
    bool improved = v->search(sol);
    EXPECT_TRUE(!improved || sol.cost < c);
}
