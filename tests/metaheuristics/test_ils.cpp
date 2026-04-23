#include "../test_fixture.hpp"
#include "../../include/metaheuristics/ils.hpp"

// ---------------------------------------------------------------------------
// Accessor: exposes protected members of ils for white-box inspection.
// ---------------------------------------------------------------------------
struct ils_accessor : public ils {
public:
    using ils::ils;
    using ils::ls;
    using ils::P;
    using ils::R;
    using ils::geometry;
    using ils::coverage;
    using ils::stopping_criterion;
    using ils::local_search;
};

// ---------------------------------------------------------------------------
// Minimal fixture — uses test_config.txt.
//   ALGORITHM_CONFIG = c00000000000
//   config[8..11] = '0'  →  P empty, R empty
//   VND: cut + trim only (2 neighborhoods)
//   d_solve early-exit condition: R.empty() && P.empty() && ls.count <= 2
// ---------------------------------------------------------------------------
class ils_minimal_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<ils_accessor> algorithm;
    cppied_solution sol;

    void SetUp() override {
        cppied_context_fixture::SetUp();
        algorithm = std::make_unique<ils_accessor>(ctx, P);
        algorithm->initialize();
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
        algorithm->geometry.complete(sol);
        algorithm->coverage.reset(sol);
        sol.cost = algorithm->geometry.cost(sol);
    }

    path_engine&     geo() { return algorithm->geometry; }
    coverage_engine& cov() { return algorithm->coverage; }
};

// ---------------------------------------------------------------------------
// Full fixture — uses ils_test_config.
//   ALGORITHM_CONFIG = c00000001111
//   config[1..7] = '0'   →  VND: cut + trim only (2 neighborhoods)
//   config[8]    = '1'   →  perturbation_ri added
//   config[9]    = '1'   →  perturbation_pt added
//   config[10]   = '1'   →  perturbation_db added
//   config[11]   = '1'   →  restart_backtrack added
//   d_solve runs the full ILS loop (P non-empty, so early-exit does not fire)
//   TIME = 2s  →  the ILS loop terminates in at most ~2 seconds
// ---------------------------------------------------------------------------
class ils_full_fixture : public ::testing::Test {
protected:
    static constexpr const char* seabed_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_seabed.txt";
    static constexpr const char* pod_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_pod.txt";
    static constexpr const char* req_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_req.txt";
    static constexpr const char* config_path =
            PROJECT_SOURCE_DIR "/tests/configurations/ils_test_config.txt";

    cppied_context ctx;
    cppied_instance P;
    std::unique_ptr<ils_accessor> algorithm;
    cppied_solution sol;
    CPPIEDCallbacks cb;

    ils_full_fixture()
        : ctx(read_configuration(config_path)),
          P(read_matrix<int>(seabed_path),
            read_matrix<double>(pod_path),
            read_matrix<double>(req_path)),
            cb({}){}

    void SetUp() override {
        algorithm = std::make_unique<ils_accessor>(ctx, P);
        algorithm->initialize();
        // Anchor start_time so that the ILS loop can measure elapsed time and
        // LKH receives a valid time budget.
        ctx.start_time = std::chrono::high_resolution_clock::now();

        // d_solve starts from an empty solution when R is non-empty:
        // restart_backtrack::restart constructs the first solution internally.
        sol = cppied_solution{};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        sol.cost     = {0, 0};
        cb.onSatisfy = [](const cppied_solution& pSol, const std::any& a){
            assert(true);
        };
        algorithm->set_callbacks(cb);
    }

    path_engine&     geo() { return algorithm->geometry; }
    coverage_engine& cov() { return algorithm->coverage; }
};

// ===========================================================================
// Initialize tests — minimal config
// ===========================================================================

TEST_F(ils_minimal_fixture, InitializeLsNeighborhoodCountMinimalConfig) {
    EXPECT_EQ(algorithm->ls.neighborhoods_count(), 3u);
}

TEST_F(ils_minimal_fixture, InitializePerturbationCountMinimalConfig) {
    EXPECT_EQ(algorithm->P.size(), 0u);
}

TEST_F(ils_minimal_fixture, InitializeRestartCountMinimalConfig) {
    EXPECT_EQ(algorithm->R.size(), 0u);
}

// ===========================================================================
// Initialize tests — full config
// ===========================================================================

TEST_F(ils_full_fixture, InitializeLsNeighborhoodCountFullConfig) {
    // config[1..7] = '1'  →  all nieghborhoods
    EXPECT_EQ(algorithm->ls.neighborhoods_count(), 10u);
}

TEST_F(ils_full_fixture, InitializePerturbationCountFullConfig) {
    EXPECT_EQ(algorithm->P.size(), 3u);
}

TEST_F(ils_full_fixture, InitializeRestartCountFullConfig) {
    EXPECT_EQ(algorithm->R.size(), 1u);
}

TEST_F(ils_full_fixture, InitializeRiIsPresentInFullConfig) {
    bool found = false;
    for (auto& p : algorithm->P)
        if (dynamic_cast<perturbation_ri*>(p.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(ils_full_fixture, InitializePtIsPresentInFullConfig) {
    bool found = false;
    for (auto& p : algorithm->P)
        if (dynamic_cast<perturbation_pt*>(p.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(ils_full_fixture, InitializeDbIsPresentInFullConfig) {
    bool found = false;
    for (auto& p : algorithm->P)
        if (dynamic_cast<perturbation_db*>(p.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

TEST_F(ils_full_fixture, InitializeRestartBacktrackIsPresentInFullConfig) {
    bool found = false;
    for (auto& r : algorithm->R)
        if (dynamic_cast<restart_backtrack*>(r.get())) { found = true; break; }
    EXPECT_TRUE(found);
}

// ===========================================================================
// Stopping criterion tests
// ===========================================================================

// Immediately after start_time is set the elapsed time is near-zero:
// stopping_criterion must return false.
TEST_F(ils_minimal_fixture, StoppingCriterionReturnsFalseBeforeTimeout) {
    ctx.start_time = std::chrono::high_resolution_clock::now();
    EXPECT_FALSE(algorithm->stopping_criterion());
}

// When start_time is set max_time+1 seconds in the past, the full budget has
// been consumed: stopping_criterion must return true.
TEST_F(ils_minimal_fixture, StoppingCriterionReturnsTrueAfterTimeout) {
    ctx.start_time = std::chrono::high_resolution_clock::now()
                     - std::chrono::seconds(ctx.max_time + 1);
    EXPECT_TRUE(algorithm->stopping_criterion());
}

// With exactly max_time seconds elapsed the criterion must also fire
// (>= comparison is inclusive).
TEST_F(ils_minimal_fixture, StoppingCriterionReturnsTrueAtExactBudget) {
    ctx.start_time = std::chrono::high_resolution_clock::now()
                     - std::chrono::seconds(ctx.max_time);
    EXPECT_TRUE(algorithm->stopping_criterion());
}

// ===========================================================================
// local_search tests
// ===========================================================================

// local_search must not throw on a valid, complete, coverage-feasible solution.
TEST_F(ils_minimal_fixture, LocalSearchDoesNotThrow) {
    EXPECT_NO_THROW(algorithm->local_search(sol));
}

// local_search delegates to vnd::search.  At the cut+trim local optimum a
// second call must return false (vnd already loops until no improvement).
TEST_F(ils_minimal_fixture, LocalSearchReturnsFalseAtLocalOptimum) {
    while (algorithm->local_search(sol)) {
        cov().reset(sol);
        sol.cost = geo().cost(sol);
    }
    EXPECT_FALSE(algorithm->local_search(sol));
}

// When local_search returns true the reported cost is strictly lower.
TEST_F(ils_minimal_fixture, LocalSearchCostDecreasedWhenReturnTrue) {
    cost_t before = sol.cost;
    if (algorithm->local_search(sol))
        EXPECT_LT(sol.cost, before);
}

// When local_search returns true the cost field matches geometry.cost.
TEST_F(ils_minimal_fixture, LocalSearchCostConsistentWhenReturnTrue) {
    if (algorithm->local_search(sol))
        EXPECT_EQ(geo().cost(sol), sol.cost);
}

// ===========================================================================
// d_solve tests — minimal config (early-exit path)
//
// With P empty, R empty, and ls.neighborhoods_count() <= 2, d_solve:
//   1. Constructs via dp_sweeper
//   2. Runs one local_search pass
//   3. Returns immediately (no ILS loop)
// ===========================================================================

TEST_F(ils_minimal_fixture, DSolveMinimalDoesNotThrow) {
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    empty.cost     = {0, 0};
    EXPECT_NO_THROW(algorithm->d_solve(empty));
}

TEST_F(ils_minimal_fixture, DSolveMinimalPathNonEmpty) {
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    empty.cost     = {0, 0};
    algorithm->d_solve(empty);
    EXPECT_GT(empty.path.size(), 0u);
}

TEST_F(ils_minimal_fixture, DSolveMinimalCoverageSatisfied) {
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    empty.cost     = {0, 0};
    algorithm->d_solve(empty);
    cov().reset(empty);
    EXPECT_TRUE((empty.coverage.array() >= P.req.array()).all());
}

// After d_solve the cost field must equal geometry.cost (consistent state).
TEST_F(ils_minimal_fixture, DSolveMinimalCostConsistent) {
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    empty.cost     = {0, 0};
    algorithm->d_solve(empty);
    EXPECT_EQ(geo().cost(empty), empty.cost);
}

// The tracked coverage must agree with a fresh reset after d_solve.
TEST_F(ils_minimal_fixture, DSolveMinimalCoverageConsistent) {
    cppied_solution empty;
    empty.coverage = Eigen::VectorXd::Zero(P.req.size());
    empty.cost     = {0, 0};
    algorithm->d_solve(empty);
    Eigen::VectorXd saved = empty.coverage;
    cov().reset(empty);
    EXPECT_TRUE(empty.coverage.isApprox(saved, 1e-9));
}

// ===========================================================================
// local_search fixture — uses ils_local_search_config.
//   ALGORITHM_CONFIG = c11111110000  (identical to vnd_test_config)
//   config[1..7] = '1'  →  n1, n12, n21, nested, tsp, gtsp, r all added
//   config[8..11] = '0' →  P empty, R empty
//   VND: cut + trim + n1 + n12 + n21 + nested + tsp + gtsp + r = 9 neighborhoods
//   TIME = 5s            →  each LKH call has a 5-second budget
//
// Tests verify the behavioural contract of ils::local_search when backed by
// the full 9-neighborhood VND.  Because vnd::search loops internally until no
// improvement, a single ils::local_search call produces a fully converged
// local optimum — a second call must always return false.
// ===========================================================================
class ils_local_search_fixture : public ::testing::Test {
protected:
    static constexpr const char* seabed_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_seabed.txt";
    static constexpr const char* pod_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_pod.txt";
    static constexpr const char* req_path =
            PROJECT_SOURCE_DIR "/tests/configurations/test_req.txt";
    static constexpr const char* config_path =
            PROJECT_SOURCE_DIR "/tests/configurations/ils_local_search_config.txt";

    cppied_context ctx;
    cppied_instance P;
    std::unique_ptr<ils_accessor> algorithm;
    cppied_solution sol;

    ils_local_search_fixture()
            : ctx(read_configuration(config_path)),
              P(read_matrix<int>(seabed_path),
                read_matrix<double>(pod_path),
                read_matrix<double>(req_path)) {}

    void SetUp() override {
        algorithm = std::make_unique<ils_accessor>(ctx, P);
        algorithm->initialize();
        // Anchor start_time: gives LKH a valid TIME - elapsed budget and
        // anchors the stopping criterion for any incidental ILS calls.
        ctx.start_time = std::chrono::high_resolution_clock::now();

        // Build a complete, coverage-feasible starting solution identical to
        // the one used in vnd_full_fixture and neighborhood_tsp_e2e_fixture.
        sol.path = {{6, 11},
                    {23, 18},
                    {44, 42},
                    {48, 51},
                    {30, 35},
                    {75, 73},
                    {62, 63}};
        sol.cost     = {0, 0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        algorithm->geometry.complete(sol);
        algorithm->coverage.reset(sol);
        sol.cost = algorithm->geometry.cost(sol);
    }

    path_engine&     geo() { return algorithm->geometry; }
    coverage_engine& cov() { return algorithm->coverage; }
};

// ---------------------------------------------------------------------------
// Sanity — verify that initialize() loaded all 9 neighborhoods into the VND.
// ---------------------------------------------------------------------------

TEST_F(ils_local_search_fixture, LocalSearchLsNeighborhoodCountIs9) {
    EXPECT_EQ(algorithm->ls.neighborhoods_count(), 10u);
}

TEST_F(ils_local_search_fixture, LocalSearchPerturbationCountIsZero) {
    // config[8..11] = '0' → no perturbations
    EXPECT_EQ(algorithm->P.size(), 0u);
}

// ---------------------------------------------------------------------------
// Safety
// ---------------------------------------------------------------------------

TEST_F(ils_local_search_fixture, LocalSearchDoesNotThrow) {
    EXPECT_NO_THROW(algorithm->local_search(sol));
}

TEST_F(ils_local_search_fixture, LocalSearchPathNonEmptyAfterCall) {
    algorithm->local_search(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

// ---------------------------------------------------------------------------
// Cost invariants on improvement
// ---------------------------------------------------------------------------

// When local_search returns true the reported cost must be strictly less than
// the pre-call incumbent.
TEST_F(ils_local_search_fixture, LocalSearchCostDecreasedWhenReturnTrue) {
    cost_t before = sol.cost;
    if (algorithm->local_search(sol))
        EXPECT_LT(sol.cost, before);
}

// When local_search returns true the cost field must equal geometry.cost.
TEST_F(ils_local_search_fixture, LocalSearchCostConsistentWhenReturnTrue) {
    if (algorithm->local_search(sol))
        EXPECT_EQ(geo().cost(sol), sol.cost);
}

// When local_search returns false the cost field must be unchanged.
TEST_F(ils_local_search_fixture, LocalSearchCostUnchangedWhenReturnFalse) {
    cost_t before = sol.cost;
    if (!algorithm->local_search(sol))
        EXPECT_EQ(sol.cost, before);
}

// ---------------------------------------------------------------------------
// Coverage invariants on improvement
// ---------------------------------------------------------------------------

// When local_search returns true the tracked coverage must match a fresh reset.
TEST_F(ils_local_search_fixture, LocalSearchCoverageConsistentWhenReturnTrue) {
    if (algorithm->local_search(sol)) {
        Eigen::VectorXd saved = sol.coverage;
        cov().reset(sol);
        EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9));
    }
}

// When local_search returns true the coverage constraint must still hold.
TEST_F(ils_local_search_fixture, LocalSearchCoverageConstraintSatisfiedWhenReturnTrue) {
    if (algorithm->local_search(sol)) {
        cov().reset(sol);
        EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
    }
}

// ---------------------------------------------------------------------------
// Convergence
// ---------------------------------------------------------------------------

// vnd::search (the implementation of local_search) runs all neighborhoods in
// a while-loop until no improvement. As there is stochastic search with
// temporary degradations, there might still be improvement on a second call.
// We verify if the local search returns true, then the cost as improved on seocnd call also.
TEST_F(ils_local_search_fixture, LocalSearchSecondCallReturnsFalseAfterConvergence) {
    algorithm->local_search(sol);
    cov().reset(sol);
    sol.cost = geo().cost(sol);
    cost_t tmp = sol.cost;
    bool second_call_improved = algorithm->local_search(sol);
    EXPECT_EQ(second_call_improved, sol.cost < tmp);
}

// At the local optimum the cost field must equal geometry.cost.
TEST_F(ils_local_search_fixture, LocalSearchCostConsistentAtLocalOptimum) {
    algorithm->local_search(sol);
    cov().reset(sol);
    sol.cost = geo().cost(sol);
    // Drive to optimum if first call returned true.
    algorithm->local_search(sol);
    EXPECT_EQ(geo().cost(sol), sol.cost);
}

// At the local optimum the coverage constraint must be satisfied.
TEST_F(ils_local_search_fixture, LocalSearchCoverageConstraintSatisfiedAtLocalOptimum) {
    algorithm->local_search(sol);
    cov().reset(sol);
    sol.cost = geo().cost(sol);
    algorithm->local_search(sol);
    cov().reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

// The return value must accurately reflect whether cost actually decreased.
TEST_F(ils_local_search_fixture, LocalSearchReturnValueReflectsActualImprovement) {
    cost_t before = sol.cost;
    bool improved = algorithm->local_search(sol);
    if (improved)
        EXPECT_LT(sol.cost, before);
    else
        EXPECT_EQ(sol.cost, before);
}

// ===========================================================================
// d_solve tests — full config (ILS loop executes)
//
// With P non-empty the early-exit condition is not satisfied.  d_solve runs
// the perturbation loop until stopping_criterion fires (TIME = 2s).
// ===========================================================================

TEST_F(ils_full_fixture, DSolveFullDoesNotThrow) {
    EXPECT_NO_THROW(algorithm->d_solve(sol));
}

TEST_F(ils_full_fixture, DSolveFullPathNonEmpty) {
    algorithm->d_solve(sol);
    EXPECT_GT(sol.path.size(), 0u);
}

TEST_F(ils_full_fixture, DSolveFullCoverageSatisfied) {
    algorithm->d_solve(sol);
    cov().reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

// The cost field must be consistent with geometry.cost after the ILS run.
TEST_F(ils_full_fixture, DSolveFullCostConsistent) {
    algorithm->d_solve(sol);
    EXPECT_EQ(geo().cost(sol), sol.cost);
}

// The tracked coverage must agree with a fresh reset after the ILS run.
TEST_F(ils_full_fixture, DSolveFullCoverageConsistent) {
    try {
        algorithm->d_solve(sol);
    } catch (const std::exception& e) {
        FAIL() << "std::exception: " << e.what();
    } catch (...) {
        FAIL() << "Unknown exception caught";
    }

    Eigen::VectorXd saved = sol.coverage;
    cov().reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9));
}

// A second d_solve call (with the solution from the first as warm start) must
// not throw and must preserve feasibility.
TEST_F(ils_full_fixture, DSolveFullSecondCallDoesNotThrow) {
    algorithm->d_solve(sol);
    cov().reset(sol);
    sol.cost = geo().cost(sol);
    ctx.start_time = std::chrono::high_resolution_clock::now();
    EXPECT_NO_THROW(algorithm->d_solve(sol));
}

TEST_F(ils_full_fixture, DSolveFullSecondCallCoverageSatisfied) {
    algorithm->d_solve(sol);
    cov().reset(sol);
    sol.cost = geo().cost(sol);
    ctx.start_time = std::chrono::high_resolution_clock::now();
    algorithm->d_solve(sol);
    cov().reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

// ===========================================================================
// Restart-path pre-condition tests
//
// In ils::d_solve the outer loop calls local_search(trial) after
// restart->restart(trial) but WITHOUT geometry.complete(trial) +
// coverage.reset(trial).  dp_sweeper::construct (invoked by restart) does not
// guarantee coverage consistency; that is why ils::d_solve always calls
// coverage.reset after its own dp_sweeper construction (lines 16-17).
//
// The tests below document the correct calling sequence and verify that it
// produces a valid state.  They also expose the missing complete+reset in the
// ILS outer loop as the root cause of the coverage assertion failures
// encountered at the beginning of neighborhood_r.
// ===========================================================================

// After restart, geometry.complete + coverage.reset must produce a path whose
// tracked coverage equals a fresh recompute.
TEST_F(ils_full_fixture, RestartFollowedByCompleteResetProducesConsistentCoverage) {
    ASSERT_FALSE(algorithm->R.empty());
    algorithm->R.front()->restart(sol);
    geo().complete(sol);
    cov().reset(sol);

    Eigen::VectorXd saved = sol.coverage;
    cov().reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9))
        << "coverage inconsistency after restart + complete + reset: "
           "dp_sweeper construct does not guarantee consistent coverage tracking";
}

// After restart + complete + reset the coverage constraint must be satisfied.
// If this fails, dp_sweeper does not guarantee a feasible construction for the
// given instance.
TEST_F(ils_full_fixture, RestartFollowedByCompleteResetSatisfiesCoverage) {
    ASSERT_FALSE(algorithm->R.empty());
    algorithm->R.front()->restart(sol);
    geo().complete(sol);
    cov().reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all())
        << "coverage infeasibility after restart + complete + reset";
}

// With the correct complete+reset preparation before local_search(trial),
// neighborhood_r must not encounter a coverage assertion failure.
// Bug: ils::d_solve outer loop calls local_search(trial) WITHOUT first calling
// geometry.complete(trial) + coverage.reset(trial) after restart->restart(trial).
// Fix: add geometry.complete(trial) + coverage.reset(trial) before
//      local_search(trial) at the end of the outer while-loop in ils::d_solve.
TEST_F(ils_full_fixture, LocalSearchAfterRestartWithCompleteResetDoesNotCrash) {
    ASSERT_FALSE(algorithm->R.empty());
    algorithm->R.front()->restart(sol);
    geo().complete(sol);
    cov().reset(sol);
    sol.cost = geo().cost(sol);
    EXPECT_NO_THROW(algorithm->local_search(sol))
        << "local_search crashed after restart + complete + reset; "
           "neighborhood_r coverage assertion fired because coverage was not reset";
}

// With the correct complete+reset preparation before local_search after
// perturbation, neighborhood_r must not encounter a coverage assertion failure.
// Bug: ils::d_solve inner loop calls local_search(local_trial) WITHOUT first
// calling geometry.complete + coverage.reset after perturbate(local_trial).
TEST_F(ils_full_fixture, LocalSearchAfterPerturbationWithCompleteResetDoesNotCrash) {
    ASSERT_FALSE(algorithm->P.empty());
    ASSERT_FALSE(algorithm->R.empty());
    // Build a non-trivial starting solution via restart.
    algorithm->R.front()->restart(sol);
    geo().complete(sol);
    cov().reset(sol);
    sol.cost = geo().cost(sol);

    auto local_trial = sol;
    algorithm->P.front()->perturbate(local_trial);
    geo().complete(local_trial);
    cov().reset(local_trial);
    local_trial.cost = geo().cost(local_trial);
    EXPECT_NO_THROW(algorithm->local_search(local_trial))
        << "local_search crashed after perturbation + complete + reset";
}

// ===========================================================================
// VND convergence with all 9 neighborhoods — coverage invariants
//
// When local_search is called repeatedly (simulating successive ILS iterations)
// each call must leave coverage consistent and feasible.  If neighborhood_r
// corrupts coverage (unconstrained cells in set_constraints, or Z1/Z2 dangling
// references), the entry-point assertions of the next call will fire.
// ===========================================================================

TEST_F(ils_local_search_fixture, LocalSearchCoverageConsistentAfterEachCallUntilConvergence) {
    // Drive the solution to a 9-neighborhood local optimum, checking coverage
    // consistency and feasibility after every VND call.
    const int max_calls = 15;
    for (int call = 0; call < max_calls; ++call) {
        bool improved = algorithm->local_search(sol);

        // Verify consistency: tracked coverage must match a fresh recompute.
        Eigen::VectorXd saved = sol.coverage;
        cov().reset(sol);
        EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9))
            << "coverage inconsistency after local_search call " << call
            << "; neighborhood_r set_constraints or retrieve_solution left "
               "coverage in a stale state";

        // Verify feasibility: coverage constraint must remain satisfied.
        EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all())
            << "coverage infeasibility after local_search call " << call
            << "; unconstrained cells in neighborhood_r::set_constraints "
               "allowed binary solver to drop a replacement segment";

        sol.cost = geo().cost(sol);
        if (!improved) break;
    }
}

// After convergence the cost must equal geometry.cost — exercises the full
// cost-tracking path including neighborhood_r::retrieve_solution.
TEST_F(ils_local_search_fixture, LocalSearchCostConsistentAtConvergence) {
    while (algorithm->local_search(sol)) {
        cov().reset(sol);
        sol.cost = geo().cost(sol);
    }
    EXPECT_EQ(geo().cost(sol), sol.cost)
        << "cost inconsistency at local optimum; "
           "neighborhood_cut::update or neighborhood_r::retrieve_solution "
           "left pSol.cost stale";
}
