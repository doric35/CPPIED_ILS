#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_nested.hpp"
#include "../../include/algorithms/MinimumSegmentPairInsertion.hpp"

struct neighborhood_nested_accessor : public neighborhood_nested {
public:
    using neighborhood_nested::neighborhood_nested;

    using neighborhood::geometry;
    using neighborhood::coverage;
    using neighborhood::problem;
};

class neighborhood_nested_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<neighborhood_nested_accessor> n;
    cppied_solution sol;

    // Neighborhoods are bound to a solution at construction.
    bool search(cppied_solution& s){
        return neighborhood_nested_accessor(ctx, P, s).local_search();
    }

    // The nested move is random: a single draw may legitimately not improve.
    // Run independent draws from the same start (~25% of draws do not
    // improve on this fixture, so 20 draws all failing has probability ~1e-12).
    struct Draw {
        bool improved;
        cppied_solution result;
    };
    static constexpr int n_draws = 20;
    std::vector<Draw> draws(const cppied_solution& start){
        std::vector<Draw> out;
        out.reserve(n_draws);
        for (int k = 0; k < n_draws; ++k){
            cppied_solution s = start;
            bool improved = search(s);
            out.push_back({improved, std::move(s)});
        }
        return out;
    }
    static int improvingDraws(const std::vector<Draw>& d){
        return static_cast<int>(std::count_if(d.begin(), d.end(), [](const Draw& x){ return x.improved; }));
    }

    void SetUp() override{
        cppied_context_fixture::SetUp();
        sol.path =  {{6, 11},
                     {23,18},
                     {44,43},
                     {6, path_engine::NULL_NODE},
                     {48, 51},
                     {30, 35},
                     {75,73},
                     {62, 63}};
        sol.cost = {0,0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        n = std::make_unique<neighborhood_nested_accessor>(ctx, P, sol);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

TEST_F(neighborhood_nested_fixture, LocalOptimaFound){
    sol.path = {{6, 11},
                {23,18},
                {44,42},
                {48, 51},
                {30, 35},
                {75,73},
                {62, 63}};
    n->coverage.reset(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    sol.cost = n->geometry.cost(sol);
    cost_t c = sol.cost;
    bool improved = search(sol);
    ASSERT_EQ(n->geometry.cost(sol), sol.cost);
    n->coverage.reset(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    EXPECT_FALSE(improved);
    EXPECT_EQ(c, sol.cost);
}

// At least one draw improves; an improving draw lowers the cost and a
// non-improving draw leaves the solution unchanged.
TEST_F(neighborhood_nested_fixture, ImprovementFound){
    const cost_t c_0 = sol.cost;
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    auto d = draws(sol);
    EXPECT_GT(improvingDraws(d), 0) << "no improvement in " << n_draws << " draws";
    for (int k = 0; k < n_draws; ++k){
        if (d[k].improved)
            EXPECT_LT(d[k].result.cost, c_0) << "draw " << k;
        else {
            EXPECT_EQ(d[k].result.cost, c_0) << "draw " << k;
            EXPECT_EQ(d[k].result.path, sol.path) << "draw " << k;
        }
    }
}

// Every improving draw keeps the tracked cost equal to the geometric cost.
TEST_F(neighborhood_nested_fixture, MarginalCostUpdate){
    auto d = draws(sol);
    ASSERT_GT(improvingDraws(d), 0) << "no improvement in " << n_draws << " draws";
    for (int k = 0; k < n_draws; ++k)
        if (d[k].improved)
            EXPECT_EQ(n->geometry.cost(d[k].result), d[k].result.cost) << "draw " << k;
}

// Every improving draw keeps the tracked coverage equal to a fresh reset.
TEST_F(neighborhood_nested_fixture, ValidCoverageUpdate){
    auto d = draws(sol);
    ASSERT_GT(improvingDraws(d), 0) << "no improvement in " << n_draws << " draws";
    for (int k = 0; k < n_draws; ++k){
        if (!d[k].improved) continue;
        cppied_solution fresh = d[k].result;
        n->coverage.reset(fresh);
        EXPECT_TRUE(fresh.coverage.isApprox(d[k].result.coverage, 1e-9)) << "draw " << k;
    }
}

// Every draw, improving or not, leaves a coverage-feasible solution.
TEST_F(neighborhood_nested_fixture, CoverageConstraintSatisfied){
    auto d = draws(sol);
    ASSERT_GT(improvingDraws(d), 0) << "no improvement in " << n_draws << " draws";
    for (int k = 0; k < n_draws; ++k){
        cppied_solution fresh = d[k].result;
        n->coverage.reset(fresh);
        EXPECT_TRUE((fresh.coverage.array() >= P.req.array()).all()) << "draw " << k;
    }
}

//----------Special cases errors-------------------------------

// Replicates the intermittent ValidCoverageUpdate failure (~10% of draws).
// The random 1-2 move that fails removes path[5] = {30,35} and re-inserts it
// unchanged at the same position with a NULL second segment. That move must
// cost exactly the removal gain, otherwise nested stores a cost that is higher
// than the real one, and the improvement found afterwards by n21 is rejected.
TEST_F(neighborhood_nested_fixture, SpecialCaseSelfReinsertionCostsRemovalGain){
    const int idx = 5;
    ASSERT_EQ(sol.path[idx], (segment{30, 35}));
    const cost_t removal_gain = n->geometry.removal_gain(sol, std::next(sol.path.begin(), idx));

    cppied_solution reduced = sol;
    n->coverage.remove(reduced, reduced.path[idx]);
    reduced.path.erase(std::next(reduced.path.begin(), idx));

    MinimumSegmentPairInsertion insertion(n->problem, n->coverage, n->geometry, reduced,
                                          segment{30, 35},
                                          segment{path_engine::NULL_NODE, path_engine::NULL_NODE},
                                          cost_t{MAX_INT, MAX_INT});
    Insertion best = insertion.solve();
    EXPECT_EQ(best.cost, removal_gain)
        << "single-segment move (NULL second segment) must cost its best single insertion";
}

TEST_F(neighborhood_nested_fixture, SpecialCaseCoverageUnsatfailure){
    sol.path ={{6, -1},
            {49, 53},
            {37, -1},
            {59, 58},
            {25, -2},
            {45, 44},
            {43, 42},
            {0, -1},
            {48, 50},
            {19, 22},
            {74, 73},
            {70, 71},
            {77, 74},
            {16, 15},
            {62, 63},
            {49, -2},
            {7, 9},
            {67, -1},
            {75, -2},
            {73, 72}};
    n->coverage.reset(sol);
    sol.cost = n->geometry.cost(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    search(sol);
    if (!(sol.coverage.array() >= P.req.array()).all()){
        for (auto& s: sol.path)
            std::cout << s << std::endl;
    }
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}