#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood_trim.hpp"
#include "../../include/neighborhoods/neighborhood_n12.hpp"

struct neighborhood_n12_accessor : public neighborhood_n12 {
public:
    using neighborhood_n12::neighborhood_n12;

    using neighborhood::geometry;
    using neighborhood::coverage;
    using neighborhood::problem;

    using neighborhood_n12::gain;
    using neighborhood_n12::update;
};

class neighborhood_n12_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<neighborhood_n12_accessor> n;
    cppied_solution sol;

    void SetUp() override{
        cppied_context_fixture::SetUp();
        sol.path =  {{6, 11},
                     {23,18},
                     {44,42},
                     {48, 51},
                     {30, 35},
                     {75,74},
                     {73, path_engine::REVERSED_NULL_NODE},
                     {62, 63}};
        sol.cost = {0,0};
        sol.coverage = Eigen::VectorXd::Zero(P.req.size());
        n = std::make_unique<neighborhood_n12_accessor>(ctx, P);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

TEST_F(neighborhood_n12_fixture, ImpromentFound){
    cost_t c_0 = sol.cost;
    bool improved = n->local_search(sol);
    ASSERT_EQ(n->geometry.cost(sol), sol.cost);
    n->coverage.reset(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    EXPECT_TRUE(improved);
    EXPECT_LT(sol.cost, c_0);
}

TEST_F(neighborhood_n12_fixture, LocalOptimaFound){
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
    EXPECT_EQ(sol.cost, c);
}

// Likely reveals a bug: update() swaps path elements without updating coverage
TEST_F(neighborhood_n12_fixture, ValidCoverageUpdate){
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

TEST_F(neighborhood_n12_fixture, MarginalCostUpdate){
    cost_t c_0 = sol.cost;
    ASSERT_TRUE(n->local_search(sol));
    EXPECT_EQ(n->geometry.cost(sol), sol.cost);
    EXPECT_LT(sol.cost, c_0);
}

TEST_F(neighborhood_n12_fixture, CoverageConstraintSatisfied){
    ASSERT_TRUE(n->local_search(sol));
    n->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

TEST_F(neighborhood_n12_fixture, ValidMarginalGainNoImprovement){
    sol.path = {{6, 11},
                {23,18},
                {44,42},
                {48, 51},
                {30, 35},
                {75,73},
                {62, 63}};
    sol.cost = n->geometry.cost(sol);
    n->coverage.reset(sol);
    n->local_search(sol);
    EXPECT_EQ(n->geometry.cost(sol), sol.cost);
}

//----------Test specific edge cases-----------------------
TEST_F(neighborhood_n12_fixture, ValidGainOnImprovementCaseOne){
    sol.path = {{6, -1},
    {49, -1},
    {12, -2},
    {44, -1},
    {18, 22},
    {23, -1},
    {35, 33},
    {32, 30},
    {46, -2},
    {24, 25},
    {58, 59},
    {37, -2},
    {50, 48},
    {0, -2},
    {42, -1},
    {6, 9},
    {10, 11},
    {78, -2},
    {5, -2},
    {72, 75},
    {76, -1},
    {35, -1},
    {83, -1},
    {63, 62}};
    sol.cost = n->geometry.cost(sol);
    n->coverage.reset(sol);
    cost_t c = sol.cost;
    bool improved = n->local_search(sol);
    EXPECT_EQ(sol.cost, n->geometry.cost(sol));
    EXPECT_TRUE((improved && sol.cost < c) || (!improved && sol.cost == c));
}