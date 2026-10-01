#include "../test_fixture.hpp"
#include "../../include/metaheuristics/ils.hpp"

// ---------------------------------------------------------------------------
// Regression tests for the ils refactor (d_solve split into helpers).
//
// None of these fixtures enable neighborhood_r (config[7]), the only
// component that requires a Gurobi license, so they run without one.
// ---------------------------------------------------------------------------

struct ils_refactor_accessor : public ils {
    using ils::ils;
    using ils::ls;
    using ils::P;
    using ils::R;
    using ils::iterations;
    using ils::incumbent;
    using ils::geometry;
    using ils::coverage;
};

class ils_refactor_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<ils_refactor_accessor> algorithm;

    // Rebuilds the algorithm with the given ALGORITHM_CONFIG and time budget.
    void make_algorithm(const std::string& config, int max_time) {
        ctx.config["ALGORITHM_CONFIG"] = config;
        ctx.max_time = max_time;
        algorithm = std::make_unique<ils_refactor_accessor>(ctx, P);
        algorithm->initialize();
        ctx.start_time = std::chrono::high_resolution_clock::now();
    }

    cppied_solution sized_empty_solution() {
        cppied_solution s;
        s.coverage = Eigen::VectorXd::Zero(P.req.size());
        s.cost     = {0, 0};
        return s;
    }

    static double seconds_since(std::chrono::high_resolution_clock::time_point t) {
        return std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - t).count();
    }
};

// ===========================================================================
// Construction preconditions
//
// d_solve copies pSolution into the incumbent before construction, so the
// caller is responsible for sizing pSolution.coverage to P.req.size().
// ===========================================================================

TEST_F(ils_refactor_fixture, DefaultSolutionHasEmptyCoverage) {
    cppied_solution s;
    EXPECT_EQ(s.coverage.size(), 0);
    EXPECT_NE(s.coverage.size(), P.req.size());
}

TEST_F(ils_refactor_fixture, DpSweeperSizedCoverageDoesNotThrow) {
    dp_sweeper h(ctx, P);
    cppied_solution s = sized_empty_solution();
    EXPECT_NO_THROW(h.construct(s, [](const cppied_solution&){}));
    EXPECT_GT(s.path.size(), 0u);
}

TEST_F(ils_refactor_fixture, DpSweeperDefaultCoverageDies) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_DEATH({
        dp_sweeper h(ctx, P);
        cppied_solution s;
        h.construct(s, [](const cppied_solution&){});
    }, "index >= 0 && index < size\\(\\)");
}

// coverage_engine::reset only calls setZero(); it does not resize.
TEST_F(ils_refactor_fixture, CoverageResetDoesNotResize) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    EXPECT_DEATH({
        coverage_engine cov(P);
        cppied_solution s;
        s.path.push_back(segment{6, 11});
        cov.reset(s);
    }, "");
}

// ===========================================================================
// Incumbent initialisation (fix: incumbent = pSolution)
// ===========================================================================

// Previously aborted: construction ran on a default incumbent.
TEST_F(ils_refactor_fixture, DSolveSizedInputDoesNotAbort) {
    make_algorithm("c00000000000", 2);
    cppied_solution s = sized_empty_solution();
    EXPECT_NO_THROW(algorithm->d_solve(s));
    EXPECT_GT(s.path.size(), 0u);
}

TEST_F(ils_refactor_fixture, DSolveIncumbentMatchesOutput) {
    make_algorithm("c00000000000", 2);
    cppied_solution s = sized_empty_solution();
    algorithm->d_solve(s);
    EXPECT_EQ(algorithm->incumbent.cost, s.cost);
    EXPECT_EQ(algorithm->incumbent.path.size(), s.path.size());
}

// A second call on the same object starts from a fresh incumbent and a
// reset iteration counter. Construction is stochastic (dp_sweeper uses
// neighborhood_tsp's rng), so only invariants are compared.
TEST_F(ils_refactor_fixture, DSolveTwiceOnSameObjectIsIndependent) {
    make_algorithm("c00000000000", 2);
    cppied_solution first = sized_empty_solution();
    algorithm->d_solve(first);

    ctx.start_time = std::chrono::high_resolution_clock::now();
    cppied_solution second = sized_empty_solution();
    algorithm->d_solve(second);

    EXPECT_EQ(algorithm->iterations, 1);
    EXPECT_EQ(algorithm->incumbent.cost, second.cost);
    EXPECT_GT(second.path.size(), 0u);
    EXPECT_EQ(algorithm->geometry.cost(second), second.cost);
    Eigen::VectorXd saved = second.coverage;
    algorithm->coverage.reset(second);
    EXPECT_TRUE(second.coverage.isApprox(saved, 1e-9));
    EXPECT_TRUE((second.coverage.array() >= P.req.array()).all());
}

// ===========================================================================
// Luby cutoff (fix: no_improvement_count += setIncumbent(...))
//
// Config c00000001000: perturbation_ri only, R empty, VND setup only.
// With the cutoff working, iteratedSearch returns after L(i)+1 non-improving
// perturbations, so the outer loop runs many times within the budget.
// With the old bug, the first iteratedSearch ran until the time limit and
// `iterations` stayed at 2.
// ===========================================================================

TEST_F(ils_refactor_fixture, IteratedSearchRespectsLubyCutoff) {
    make_algorithm("c00000001000", 2);
    ASSERT_EQ(algorithm->P.size(), 1u);
    ASSERT_TRUE(algorithm->R.empty());
    cppied_solution s = sized_empty_solution();
    algorithm->d_solve(s);
    EXPECT_GT(algorithm->iterations, 2);
}

TEST_F(ils_refactor_fixture, DSolvePerturbationOnlyFeasible) {
    make_algorithm("c00000001000", 2);
    cppied_solution s = sized_empty_solution();
    algorithm->d_solve(s);
    EXPECT_EQ(algorithm->geometry.cost(s), s.cost);
    Eigen::VectorXd saved = s.coverage;
    algorithm->coverage.reset(s);
    EXPECT_TRUE(s.coverage.isApprox(saved, 1e-9));
    EXPECT_TRUE((s.coverage.array() >= P.req.array()).all());
}

// d_solve must not overrun the time budget by more than one step.
TEST_F(ils_refactor_fixture, DSolvePerturbationOnlyStopsNearBudget) {
    make_algorithm("c00000001000", 2);
    cppied_solution s = sized_empty_solution();
    auto t0 = ctx.start_time;
    algorithm->d_solve(s);
    EXPECT_LT(seconds_since(t0), ctx.max_time + 2.0);
}

// ===========================================================================
// Local-search-only path (R and P empty, VND has more than setup moves)
//
// Config c10000000000: n1 added to N_simple, no perturbation, no restart.
// d_solve should run the root local search once and return; the ILS loop
// cannot make progress without perturbations or restarts.
// ===========================================================================

TEST_F(ils_refactor_fixture, DSolveLocalSearchOnlyFeasible) {
    make_algorithm("c10000000000", 3);
    ASSERT_EQ(algorithm->ls.neighborhoods_count(), 4u);
    cppied_solution s = sized_empty_solution();
    algorithm->d_solve(s);
    EXPECT_EQ(algorithm->geometry.cost(s), s.cost);
    Eigen::VectorXd saved = s.coverage;
    algorithm->coverage.reset(s);
    EXPECT_TRUE(s.coverage.isApprox(saved, 1e-9));
    EXPECT_TRUE((s.coverage.array() >= P.req.array()).all());
}

TEST_F(ils_refactor_fixture, DSolveLocalSearchOnlyReturnsBeforeBudget) {
    make_algorithm("c10000000000", 3);
    cppied_solution s = sized_empty_solution();
    auto t0 = ctx.start_time;
    algorithm->d_solve(s);
    EXPECT_LT(seconds_since(t0), double(ctx.max_time))
        << "d_solve ran the ILS loop until timeout on the local-search-only path";
    EXPECT_EQ(algorithm->iterations, 1)
        << "iteratedSearch was entered on the local-search-only path";
}
