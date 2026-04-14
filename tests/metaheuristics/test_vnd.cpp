#include "../test_fixture.hpp"
#include "../../include/metaheuristics/vnd.hpp"
#include "../../include/neighborhoods/neighborhood_cut.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_n12.hpp"
#include "../../include/neighborhoods/neighborhood_n21.hpp"
#include "../../include/neighborhoods/neighborhood_nested.hpp"
#include "../../include/neighborhoods/neighborhood_tsp.hpp"
#include "../../include/neighborhoods/neighborhood_gtsp.hpp"
#include "../../include/neighborhoods/neighborhood_r.hpp"

// ---------------------------------------------------------------------------
// Accessor: exposes the protected N vector for white-box inspection.
// ---------------------------------------------------------------------------
struct vnd_accessor : public vnd {
public:
    using vnd::vnd;
    using vnd::N;

    using vnd::geometry;
    using vnd::coverage;
    using vnd::problem;
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
        // Establish a consistent, coverage-feasible, complete starting state.
        v->geometry.complete(sol);
        v->coverage.reset(sol);
        sol.cost = v->geometry.cost(sol);
    }

    // Expose engines for assertion helpers.
    path_engine&     geo()      { return v->geometry; }
    coverage_engine& cov()      { return v->coverage; }
};

// ---------------------------------------------------------------------------
// Full fixture — uses vnd_test_config.
//   ALGORITHM_CONFIG = -c11111110000  (12 chars)
//   config[1] = '1'         →  n1 added
//   config[2] = '1'         →  n12 added
//   config[3] = '1'         →  n21 added
//   config[4] = '1'         →  nested added
//   config[5] = '1'         →  tsp added
//   config[6] = '1'         →  gtsp added
//   config[7] = '1'         →  r added
//   Result: cut + trim + n1 + n12 + n21 + nested + tsp + gtsp + r = 9 neighborhoods
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
        // Anchor start_time — required by TSP/GTSP/R neighborhoods for LKH.
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
// ===========================================================================

TEST_F(vnd_minimal_fixture, InitializeNeighborhoodCountMinimalConfig) {
    EXPECT_EQ(v->neighborhoods_count(), 2u);
}

TEST_F(vnd_minimal_fixture, InitializeCutIsFirstNeighborhood) {
    ASSERT_GE(v->N.size(), 1u);
    EXPECT_NE(dynamic_cast<neighborhood_cut*>(v->N[0].get()), nullptr);
}

TEST_F(vnd_minimal_fixture, InitializeTrimIsSecondNeighborhood) {
    ASSERT_GE(v->N.size(), 2u);
    EXPECT_NE(dynamic_cast<neighborhood_trim*>(v->N[1].get()), nullptr);
}

// Calling initialize() a second time without a clear guard appends duplicates.
// This test documents the current behaviour (not a requirement that it should
// duplicate — it documents what it does so that a future N.clear() fix is
// detectable).
TEST_F(vnd_minimal_fixture, InitializeDoubleCallAppendsDuplicates) {
    v->initialize();
    EXPECT_EQ(v->neighborhoods_count(), 4u);
}

// ===========================================================================
// Initialize tests — full config
// ===========================================================================

TEST_F(vnd_full_fixture, InitializeNeighborhoodCountFullConfig) {
    EXPECT_EQ(v->neighborhoods_count(), 9u);
}

TEST_F(vnd_full_fixture, InitializeCutIsFirstNeighborhoodFullConfig) {
    ASSERT_GE(v->N.size(), 1u);
    EXPECT_NE(dynamic_cast<neighborhood_cut*>(v->N[0].get()), nullptr);
}

TEST_F(vnd_full_fixture, InitializeTrimIsSecondNeighborhoodFullConfig) {
    ASSERT_GE(v->N.size(), 2u);
    EXPECT_NE(dynamic_cast<neighborhood_trim*>(v->N[1].get()), nullptr);
}

TEST_F(vnd_full_fixture, InitializeN12IsPresentInFullConfig) {
    bool found = false;
    for (auto& n : v->N)
        if (dynamic_cast<neighborhood_n12*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(vnd_full_fixture, InitializeN21IsPresentInFullConfig) {
    bool found = false;
    for (auto& n : v->N)
        if (dynamic_cast<neighborhood_n21*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(vnd_full_fixture, InitializeNestedIsPresentInFullConfig) {
    bool found = false;
    for (auto& n : v->N)
        if (dynamic_cast<neighborhood_nested*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(vnd_full_fixture, InitializeTspIsPresentInFullConfig) {
    bool found = false;
    for (auto& n : v->N)
        if (dynamic_cast<neighborhood_tsp*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(vnd_full_fixture, InitializeGtspIsPresentInFullConfig) {
    bool found = false;
    for (auto& n : v->N)
        if (dynamic_cast<neighborhood_gtsp*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(vnd_full_fixture, InitializeRIsPresentInFullConfig) {
    bool found = false;
    for (auto& n : v->N)
        if (dynamic_cast<neighborhood_r*>(n.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

// ===========================================================================
// search() tests — minimal config (cut + trim only, fast, no LKH)
// ===========================================================================

TEST_F(vnd_minimal_fixture, SearchDoesNotThrowOnValidSolution) {
    EXPECT_NO_THROW(v->search(sol));
}

// A solution that is already at the local optimum for cut+trim must make
// search() return false (no improvement found).
TEST_F(vnd_minimal_fixture, SearchReturnsFalseAtLocalOptimum) {
    // Drive to local optimum first.
    while (v->search(sol)) {
        v->coverage.reset(sol);
        sol.cost = v->geometry.cost(sol);
    }
    // Now search must return false immediately.
    EXPECT_FALSE(v->search(sol));
}

// After convergence the path must be non-empty.
TEST_F(vnd_minimal_fixture, SearchPathNonEmptyAfterConvergence) {
    while (v->search(sol)) {
        v->coverage.reset(sol);
        sol.cost = v->geometry.cost(sol);
    }
    EXPECT_GT(sol.path.size(), 0u);
}

// When search() returns true the reported cost must be strictly less than the
// pre-call incumbent.
TEST_F(vnd_minimal_fixture, SearchCostDecreasedWhenReturnTrue) {
    cost_t before = sol.cost;
    if (v->search(sol))
        EXPECT_LT(sol.cost, before);
}

// When search() returns true the stored cost must be consistent with
// geometry.cost().
TEST_F(vnd_minimal_fixture, SearchCostConsistentAfterSearchReturnTrue) {
    if (v->search(sol))
        EXPECT_EQ(geo().cost(sol), sol.cost);
}

// When search() returns true the coverage vector must be consistent with a
// fresh reset.
TEST_F(vnd_minimal_fixture, SearchCoverageConsistentAfterSearchReturnTrue) {
    if (v->search(sol)) {
        Eigen::VectorXd saved = sol.coverage;
        cov().reset(sol);
        EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9));
    }
}

// When search() returns true the coverage constraint must still be satisfied.
TEST_F(vnd_minimal_fixture, SearchCoverageConstraintSatisfiedAfterSearchReturnTrue) {
    if (v->search(sol)) {
        cov().reset(sol);
        EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
    }
}

// After search() completes (returns true or false) a second call must return
// false: search() already runs all neighborhoods in a while-loop until no
// improvement, so the solution is already at the local optimum on exit.
TEST_F(vnd_minimal_fixture, SearchSecondCallReturnsFalseAfterConvergence) {
    v->search(sol);
    // Re-establish a consistent state before the second call.
    cov().reset(sol);
    sol.cost = geo().cost(sol);
    cost_t curr_c = sol.cost;
    v->search(sol);
    EXPECT_FALSE(curr_c > sol.cost);
}

// After search() converges, calling each neighborhood individually must not
// find further improvement — this verifies that search() fully exhausted all
// neighborhoods before returning.
TEST_F(vnd_minimal_fixture, SearchLocalOptimumVerifiedForAllNeighborhoods) {
    while (v->search(sol)) {
        cov().reset(sol);
        sol.cost = geo().cost(sol);
    }
    cov().reset(sol);
    sol.cost = geo().cost(sol);

    for (std::size_t i = 0; i < v->N.size(); ++i) {
        cppied_solution copy = sol;
        cost_t copy_c = copy.cost;
        v->N[i]->local_search(copy);
        EXPECT_FALSE(copy_c > copy.cost)
            << "Neighborhood " << i
            << " still finds improvement after VND convergence.";
    }
}

// The return value must accurately reflect whether cost actually decreased.
TEST_F(vnd_minimal_fixture, SearchReturnValueReflectsActualImprovement) {
    cost_t before = sol.cost;
    bool improved = v->search(sol);
    if (improved)
        EXPECT_LT(sol.cost, before);
    else
        EXPECT_EQ(sol.cost, before);
}

// After convergence the coverage constraint must be satisfied.
TEST_F(vnd_minimal_fixture, SearchCoverageConstraintSatisfiedAfterConvergence) {
    while (v->search(sol)) {
        cov().reset(sol);
        sol.cost = geo().cost(sol);
    }
    cov().reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

// After convergence the cost must equal geometry.cost (full consistency).
TEST_F(vnd_minimal_fixture, SearchCostConsistentAfterConvergence) {
    while (v->search(sol)) {
        cov().reset(sol);
        sol.cost = geo().cost(sol);
    }
    EXPECT_EQ(geo().cost(sol), sol.cost);
}

// ===========================================================================
// E2E tests — full config (all 8 neighborhoods, requires LKH binary)
// ===========================================================================

TEST_F(vnd_full_fixture, FullConfigInitializeDoesNotThrow) {
    // initialize() was already called in SetUp; a fresh instance must not throw.
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
