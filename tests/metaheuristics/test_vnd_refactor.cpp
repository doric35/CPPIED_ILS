#include "../test_fixture.hpp"
#include "../../include/metaheuristics/vnd.hpp"

// ---------------------------------------------------------------------------
// Control-flow tests for vnd::search after it was split into
// simple/nested/large descent helpers.
//
// Scripted mock neighborhoods replace the real ones: each mock logs its name
// when called and lowers the solution cost by 1 for its first `improvements`
// calls. The solution path is left untouched, so setCompleteSolution inside
// search() is a no-op on cost. Expected call sequences follow the
// pre-refactor vnd::search.
// ---------------------------------------------------------------------------

struct vnd_refactor_accessor : public vnd {
    using vnd::vnd;
    using vnd::N_setup;
    using vnd::N_simple;
    using vnd::N_nested;
    using vnd::N_large;
    using vnd::geometry;
    using vnd::coverage;
};

struct mock_neighborhood : public neighborhood {
    mock_neighborhood(cppied_context& c, cppied_instance& i,
                      std::vector<std::string>& log, std::string name, int improvements)
        : neighborhood(c, i), log(log), name(std::move(name)), improvements(improvements) {}

    bool local_search(cppied_solution& pSol) override {
        log.push_back(name);
        if (improvements <= 0) return false;
        --improvements;
        pSol.cost.length -= 1;
        return true;
    }

    std::vector<std::string>& log;
    std::string name;
    int improvements;
};

class vnd_refactor_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<vnd_refactor_accessor> v;
    cppied_solution sol;
    std::vector<std::string> log;

    void SetUp() override {
        cppied_context_fixture::SetUp();
        v = std::make_unique<vnd_refactor_accessor>(ctx, P);
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

    // One mock per group: S (setup), M (simple), N (nested), L (large).
    void install_mocks(int setup, int simple, int nested, int large) {
        v->N_setup.push_back(std::make_unique<mock_neighborhood>(ctx, P, log, "S", setup));
        v->N_simple.push_back(std::make_unique<mock_neighborhood>(ctx, P, log, "M", simple));
        v->N_nested.push_back(std::make_unique<mock_neighborhood>(ctx, P, log, "N", nested));
        v->N_large.push_back(std::make_unique<mock_neighborhood>(ctx, P, log, "L", large));
    }

    void set_stop(std::function<bool()> stop) {
        CPPIEDCallbacks cb{};
        cb.stopCriteria = [stop](const cppied_solution&, const std::any&) { return stop(); };
        v->set_callbacks(cb);
    }

    using seq = std::vector<std::string>;
};

// ===========================================================================
// Ordering and termination
// ===========================================================================

TEST_F(vnd_refactor_fixture, NoImprovementSinglePassInOrder) {
    install_mocks(0, 0, 0, 0);
    cost_t before = sol.cost;
    EXPECT_FALSE(v->search(sol));
    EXPECT_EQ(log, (seq{"S", "M", "N", "L"}));
    EXPECT_EQ(sol.cost, before);
}

// A large-neighborhood improvement triggers a second full pass.
TEST_F(vnd_refactor_fixture, LargeImprovementRestartsOuterLoop) {
    install_mocks(0, 0, 0, 1);
    cost_t before = sol.cost;
    EXPECT_TRUE(v->search(sol));
    EXPECT_EQ(log, (seq{"S", "M", "N", "L",
                        "S", "M", "N", "L"}));
    EXPECT_EQ(sol.cost.length, before.length - 1);
}

// Setup neighborhoods are driven to a local optimum (find_local_optima).
TEST_F(vnd_refactor_fixture, SetupRunsUntilLocalOptimum) {
    install_mocks(2, 0, 0, 0);
    EXPECT_TRUE(v->search(sol));
    EXPECT_EQ(log, (seq{"S", "S", "S", "M", "N", "L",
                        "S", "M", "N", "L"}));
}

// The simple group repeats while it improves, before moving to nested.
// Each simple neighborhood is driven to its local optimum (M, M, M), then
// the group loop runs once more to confirm no improvement (M).
TEST_F(vnd_refactor_fixture, SimpleGroupRepeatsWhileImproving) {
    install_mocks(0, 2, 0, 0);
    EXPECT_TRUE(v->search(sol));
    EXPECT_EQ(log, (seq{"S", "M", "M", "M", "M", "N", "L",
                        "S", "M", "N", "L"}));
}

// A nested improvement does not end the pass: the large neighborhoods still
// run, then the outer loop restarts because the pass improved.
TEST_F(vnd_refactor_fixture, NestedImprovementStillRunsLargeInSamePass) {
    install_mocks(0, 0, 1, 0);
    EXPECT_TRUE(v->search(sol));
    EXPECT_EQ(log, (seq{"S", "M", "N", "L",
                        "S", "M", "N", "L"}));
}

// Return value reflects total improvement across passes.
TEST_F(vnd_refactor_fixture, ReturnValueReflectsTotalImprovement) {
    install_mocks(1, 1, 0, 1);
    cost_t before = sol.cost;
    EXPECT_TRUE(v->search(sol));
    EXPECT_EQ(sol.cost.length, before.length - 3);
}

// ===========================================================================
// Stopping criterion
// ===========================================================================

TEST_F(vnd_refactor_fixture, StopCriterionTrueCallsNoNeighborhood) {
    install_mocks(5, 5, 5, 5);
    set_stop([] { return true; });
    cost_t before = sol.cost;
    EXPECT_FALSE(v->search(sol));
    EXPECT_TRUE(log.empty());
    EXPECT_EQ(sol.cost, before);
}

// Stop fires after the first neighborhood call: nothing else may run.
TEST_F(vnd_refactor_fixture, StopCriterionMidSearchHaltsAllGroups) {
    install_mocks(5, 5, 5, 5);
    set_stop([this] { return !log.empty(); });
    EXPECT_TRUE(v->search(sol));
    EXPECT_EQ(log, (seq{"S"}));
}

// Stop fires during the large group: search must terminate even though
// the pass improved.
TEST_F(vnd_refactor_fixture, StopCriterionTerminatesOuterLoop) {
    install_mocks(0, 0, 0, 100);
    set_stop([this] { return log.size() >= 4; });
    EXPECT_TRUE(v->search(sol));
    EXPECT_EQ(log, (seq{"S", "M", "N", "L"}));
}

// ===========================================================================
// State consistency (setCompleteSolution inside search)
// ===========================================================================

TEST_F(vnd_refactor_fixture, SearchLeavesCoverageConsistent) {
    install_mocks(0, 0, 0, 0);
    sol.coverage.setZero();  // corrupt tracked coverage
    v->search(sol);
    Eigen::VectorXd saved = sol.coverage;
    v->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9));
}

// ===========================================================================
// Integration — real neighborhoods except neighborhood_r (no Gurobi needed).
//   ALGORITHM_CONFIG = c11111100000: n1, n12, n21, nested, tsp, gtsp.
// ===========================================================================

class vnd_refactor_no_ilp_fixture : public vnd_refactor_fixture {
protected:
    void SetUp() override {
        vnd_refactor_fixture::SetUp();
        ctx.config["ALGORITHM_CONFIG"] = "c11111100000";
        ctx.max_time = 10;
        v->initialize();
    }
};

TEST_F(vnd_refactor_no_ilp_fixture, NeighborhoodCount) {
    EXPECT_EQ(v->neighborhoods_count(), 9u);
}

TEST_F(vnd_refactor_no_ilp_fixture, SearchFeasibleAndConsistent) {
    cost_t before = sol.cost;
    bool improved = v->search(sol);
    EXPECT_EQ(improved, sol.cost < before);
    EXPECT_EQ(v->geometry.cost(sol), sol.cost);
    Eigen::VectorXd saved = sol.coverage;
    v->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(saved, 1e-9));
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}
