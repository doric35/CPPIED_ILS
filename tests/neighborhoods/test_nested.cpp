#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_nested.hpp"

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

    void SetUp() override{
        cppied_context_fixture::SetUp();
        sol.path =  {{6, 11},
                     {23,18},
                     {44,43},
                     {6, NULL_NODE},
                     {48, 51},
                     {30, 35},
                     {75,73},
                     {62, 63}};
        sol.cost = {0,0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        n = std::make_unique<neighborhood_nested_accessor>(ctx, P);
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
    bool improved = n->local_search(sol);
    ASSERT_EQ(n->geometry.cost(sol), sol.cost);
    n->coverage.reset(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    EXPECT_FALSE(improved);
    EXPECT_EQ(c, sol.cost);
}

TEST_F(neighborhood_nested_fixture, ImprovementFound){
    cost_t c_0 = sol.cost;
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    bool improved = n->local_search(sol);
    EXPECT_TRUE(improved);
    EXPECT_LT(sol.cost, c_0);
}

TEST_F(neighborhood_nested_fixture, MarginalCostUpdate){
    ASSERT_TRUE(n->local_search(sol));
    EXPECT_EQ(n->geometry.cost(sol), sol.cost);
}

TEST_F(neighborhood_nested_fixture, ValidCoverageUpdate){
    ASSERT_TRUE(n->local_search(sol));
    Eigen::VectorXd C = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(C, 1e-9));
}

TEST_F(neighborhood_nested_fixture, CoverageConstraintSatisfied){
    ASSERT_TRUE(n->local_search(sol));
    n->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

//----------Special cases errors-------------------------------

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
    n->local_search(sol);
    if (!(sol.coverage.array() >= P.req.array()).all()){
        for (auto& s: sol.path)
            std::cout << s << std::endl;
    }
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}