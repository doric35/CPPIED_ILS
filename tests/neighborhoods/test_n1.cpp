#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_cut.hpp"
#include "../../include/neighborhoods/neighborhood_n1.hpp"

struct neighborhood_n1_accessor : public neighborhood_n1 {
public:
    using neighborhood_n1::neighborhood_n1;

    using neighborhood::geometry;
    using neighborhood::coverage;
    using neighborhood::problem;

    using neighborhood_n1::gain;
    using neighborhood_n1::update;
};

class neighborhood_n1_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<neighborhood_n1_accessor> n;
    cppied_solution sol;

    void SetUp() override{
        cppied_context_fixture::SetUp();
        sol.path =  {{6, 11},
                     {23,18},
                     {44,42},
                     {48, 51},
                     {30, 35},
                     {75,73},
                     {62, 63}};
        sol.cost = {0,0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        n = std::make_unique<neighborhood_n1_accessor>(ctx, P);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

TEST_F(neighborhood_n1_fixture, LocalOptimaFound){
    bool improved = n->local_search(sol);
    n->coverage.reset(sol);
    ASSERT_EQ(n->geometry.cost(sol), sol.cost);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    EXPECT_FALSE(improved);
    n->geometry.complete(sol);
    n->coverage.reset(sol);
    EXPECT_FALSE(n->local_search(sol));
}

TEST_F(neighborhood_n1_fixture, ImpromentFound){
    cppied_solution other = sol;
    other.path = {{6, 11},
                  {79, path_engine::NULL_NODE},
                  {23,18},
                  {44,42},
                  {48, 51},
                  {30, 35},
                  {75,74},
                  {10, 9},
                  {62, 63}};
    n->coverage.reset(other);
    ASSERT_EQ(n->geometry.cost(other), sol.cost);
    ASSERT_TRUE((other.coverage.array() >= P.req.array()).all());
    bool improved = n->local_search(other);
    EXPECT_TRUE(improved);
    ASSERT_EQ(n->geometry.cost(other), other.cost);
    EXPECT_LT(other.cost, sol.cost);
}

// Likely reveals a bug: update() swaps path elements without updating coverage
TEST_F(neighborhood_n1_fixture, ValidCoverageUpdate){
    cppied_solution other = sol;
    other.path = {{6, 11},
                  {79, path_engine::NULL_NODE},
                  {23,18},
                  {44,42},
                  {48, 51},
                  {30, 35},
                  {75,74},
                  {10, 9},
                  {62, 63}};
    n->coverage.reset(other);
    other.cost = n->geometry.cost(other);
    ASSERT_TRUE(n->local_search(other));
    Eigen::VectorXd C = other.coverage;
    n->coverage.reset(other);
    EXPECT_TRUE(other.coverage.isApprox(C, 1e-9));
}

TEST_F(neighborhood_n1_fixture, CoverageConstraintSatisfied){
    cppied_solution other = sol;
    other.path = {{6, 11},
                  {79, path_engine::NULL_NODE},
                  {23,18},
                  {44,42},
                  {48, 51},
                  {30, 35},
                  {75,74},
                  {10, 9},
                  {62, 63}};
    n->coverage.reset(other);
    other.cost = n->geometry.cost(other);
    ASSERT_TRUE(n->local_search(other));
    n->coverage.reset(other);
    EXPECT_TRUE((other.coverage.array() >= P.req.array()).all());
}

TEST_F(neighborhood_n1_fixture, LocalOptimaAfterImprovement){
    cppied_solution other = sol;
    other.path = {{6, 11},
                  {79, path_engine::NULL_NODE},
                  {23,18},
                  {44,42},
                  {48, 51},
                  {30, 35},
                  {75,74},
                  {10, 9},
                  {62, 63}};
    n->coverage.reset(other);
    other.cost = n->geometry.cost(other);
    ASSERT_TRUE(n->local_search(other));
    EXPECT_FALSE(n->local_search(other));
}