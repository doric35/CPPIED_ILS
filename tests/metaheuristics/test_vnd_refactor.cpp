#include "../test_fixture.hpp"
#include "../../include/metaheuristics/vnd.hpp"

// ---------------------------------------------------------------------------
// Control-flow tests for vnd::search after it was split into
// simple/nested/large descent helpers.
//
// Scripted mock neighborhoods replace the real ones in the LocalSearchFactory:
// each mock logs its name when called and, for the first `improvements` calls
// of its group, makes a real improvement by removing the last segment(s) of the
// path until geometry.cost strictly decreases. The improvement must be real:
// setCompleteSolution inside search() recomputes the cost from the path. Each
// cost drop is recorded in the fixture. The factory instantiates neighborhoods
// on every descent, so the remaining improvements are kept in the fixture.
// Expected call sequences follow the pre-refactor vnd::search.
// ---------------------------------------------------------------------------

struct vnd_refactor_accessor : public vnd {
    using vnd::vnd;
    using vnd::factory;
    using vnd::geometry;
    using vnd::coverage;
};

struct mock_neighborhood : public neighborhood {
    mock_neighborhood(cppied_context& c, cppied_instance& i, cppied_solution& s,
                      std::vector<std::string>& log, std::string name, int& improvements,
                      std::vector<cost_t>& drops)
        : neighborhood(c, i, s), log(log), name(std::move(name)), improvements(improvements),
          drops(drops) {}

    bool local_search() override {
        log.push_back(name);
        if (improvements <= 0) return false;
        const cost_t before = geometry.cost(solution);
        do {
            if (solution.path.size() <= 1) return false;
            coverage.remove(solution, solution.path.back());
            solution.path.pop_back();
        } while (!(geometry.cost(solution) < before));
        --improvements;
        solution.cost = geometry.cost(solution);
        drops.push_back(before - solution.cost);
        return true;
    }

    std::vector<std::string>& log;
    std::string name;
    int& improvements;
    std::vector<cost_t>& drops;
};


class vnd_refactor_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<vnd_refactor_accessor> v;
    cppied_solution sol;
    std::vector<std::string> log;
    std::array<int, 4> remaining{};
    std::vector<cost_t> drops;          // cost drop of each mock improvement, in call order

    cost_t total_drop() const {
        cost_t total{0, 0};
        for (const auto& d : drops) total += d;
        return total;
    }

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
        remaining = {setup, simple, nested, large};
        using factory_t = Sequence<neighborhood>::TFactory;
        auto mock = [this](const std::string& name, int& counter) {
            return [this, name, &counter](cppied_context& c, cppied_instance& i, cppied_solution& s)
                    -> std::unique_ptr<neighborhood> {
                return std::make_unique<mock_neighborhood>(c, i, s, log, name, counter, drops);
            };
        };
        v->factory.setupNeighborhoodsFactory  = Sequence<neighborhood>(factory_t{mock("S", remaining[0])});
        v->factory.simpleNeighborhoodsFactory = Sequence<neighborhood>(factory_t{mock("M", remaining[1])});
        v->factory.nestedNeighborhoodsFactory = Sequence<neighborhood>(factory_t{mock("N", remaining[2])});
        v->factory.largeNeighborhoodsFactory  = Sequence<neighborhood>(factory_t{mock("L", remaining[3])});
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
    ASSERT_EQ(drops.size(), 1u);
    EXPECT_EQ(sol.cost, before - total_drop());
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
    ASSERT_EQ(drops.size(), 3u);
    EXPECT_EQ(sol.cost, before - total_drop());
    EXPECT_EQ(sol.cost, v->geometry.cost(sol));
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
        // The factory reads ALGORITHM_CONFIG at construction.
        v = std::make_unique<vnd_refactor_accessor>(ctx, P);
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
