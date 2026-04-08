#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_cut.hpp"

struct neighborhood_trim_accessor : public neighborhood_trim {
public:
    using neighborhood_trim::neighborhood_trim;

    using neighborhood::geometry;
    using neighborhood::coverage;
    using neighborhood::problem;

    using neighborhood_trim::gain;
    using neighborhood_trim::update;
};

class neighborhood_trim_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<neighborhood_trim_accessor> n;
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
        n = std::make_unique<neighborhood_trim_accessor>(ctx, P);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

TEST_F(neighborhood_trim_fixture, LocalOptimaFound){
    EXPECT_FALSE(n->local_search(sol));
    n->geometry.complete(sol);
    n->coverage.reset(sol);
    EXPECT_FALSE(n->local_search(sol));
}

TEST_F(neighborhood_trim_fixture, ImprovementFound){
    cppied_solution other = sol;
    other.path =  {{6, 11},
                 {79, NULL_NODE},
                 {23,18},
                 {44,42},
                 {48, 51},
                 {30, 35},
                 {75,74},
                 {16, REVERSED_NULL_NODE},
                 {61, 63}};
    n->coverage.reset(other);
    ASSERT_TRUE((other.coverage.array() >= P.req.array()).all());
    ASSERT_TRUE((n->geometry.cost(other) == n->geometry.cost(sol) + cost_t{0,2}));
    EXPECT_TRUE(n->local_search(other));
}

TEST_F(neighborhood_trim_fixture, MarginalCostUpdate){
    cppied_solution other = sol;
    other.path = {{6, 11},
                  {79, NULL_NODE},
                  {23,18},
                  {44,42},
                  {48, 51},
                  {30, 35},
                  {75,74},
                  {16, REVERSED_NULL_NODE},
                  {61, 63}};
    n->coverage.reset(other);
    other.cost = n->geometry.cost(other);
    ASSERT_TRUE(n->local_search(other));
    EXPECT_EQ(n->geometry.cost(other), other.cost);
}

TEST_F(neighborhood_trim_fixture, ValidCoverageUpdate){
    cppied_solution other = sol;
    other.path = {{6, 11},
                  {79, NULL_NODE},
                  {23,18},
                  {44,42},
                  {48, 51},
                  {30, 35},
                  {75,74},
                  {16, REVERSED_NULL_NODE},
                  {61, 63}};
    n->coverage.reset(other);
    other.cost = n->geometry.cost(other);
    ASSERT_TRUE(n->local_search(other));
    Eigen::VectorXd C = other.coverage;
    n->coverage.reset(other);
    EXPECT_TRUE(other.coverage.isApprox(C, 1e-9));
}

TEST_F(neighborhood_trim_fixture, CoverageConstraintSatisfied){
    cppied_solution other = sol;
    other.path = {{6, 11},
                  {79, NULL_NODE},
                  {23,18},
                  {44,42},
                  {48, 51},
                  {30, 35},
                  {75,74},
                  {16, REVERSED_NULL_NODE},
                  {61, 63}};
    n->coverage.reset(other);
    other.cost = n->geometry.cost(other);
    ASSERT_TRUE(n->local_search(other));
    n->coverage.reset(other);
    EXPECT_TRUE((other.coverage.array() >= P.req.array()).all());
}

TEST_F(neighborhood_trim_fixture, LocalOptimaAfterImprovement){
    cppied_solution other = sol;
    other.path = {{6, 11},
                  {79, NULL_NODE},
                  {23,18},
                  {44,42},
                  {48, 51},
                  {30, 35},
                  {75,74},
                  {16, REVERSED_NULL_NODE},
                  {61, 63}};
    n->coverage.reset(other);
    other.cost = n->geometry.cost(other);
    ASSERT_TRUE(n->local_search(other));
    EXPECT_FALSE(n->local_search(other));
}
