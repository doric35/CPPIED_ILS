#include "../test_fixture.hpp"
#include "../../include/neighborhoods/neighborhood_cut.hpp"

struct neighborhood_cut_accessor : public neighborhood_cut {
public:
    using neighborhood_cut::neighborhood_cut;

    using neighborhood::geometry;
    using neighborhood::coverage;
    using neighborhood::problem;

    using neighborhood_cut::gain;
    using neighborhood_cut::update;
};

class neighborhood_cut_fixture : public cppied_context_fixture {
protected:
    std::unique_ptr<neighborhood_cut_accessor> n;
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
        n = std::make_unique<neighborhood_cut_accessor>(ctx, P);
        n->coverage.reset(sol);
        sol.cost = n->geometry.cost(sol);
    }
};

TEST_F(neighborhood_cut_fixture, LocalOptimaFound){
    EXPECT_FALSE(n->local_search(sol));
}

TEST_F(neighborhood_cut_fixture, ImprovedSolution){
    n->geometry.complete(sol);
    n->coverage.reset(sol);
    cost_t c = sol.cost;
    EXPECT_TRUE(n->local_search(sol));
    EXPECT_LE(sol.cost, c);
}

TEST_F(neighborhood_cut_fixture, MarginalCostUpdate){
    n->geometry.complete(sol);
    n->coverage.reset(sol);
    ASSERT_TRUE(n->local_search(sol));
    cost_t c_hat = sol.cost;
    cost_t true_c = n->geometry.cost(sol);
    EXPECT_EQ(c_hat, true_c);
}

TEST_F(neighborhood_cut_fixture, ValidCoverageUpdate){
    n->geometry.complete(sol);
    n->coverage.reset(sol);
    ASSERT_TRUE(n->local_search(sol));
    Eigen::VectorXd C = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE(sol.coverage.isApprox(C, 1e-9));
}

TEST_F(neighborhood_cut_fixture, CoverageConstraintSatisfied){
    n->geometry.complete(sol);
    n->coverage.reset(sol);
    ASSERT_TRUE(n->local_search(sol));
    n->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

TEST_F(neighborhood_cut_fixture, CovergeToLocalOptima){
    n->geometry.complete(sol);
    n->coverage.reset(sol);
    bool improved = n->local_search(sol);
    while (n->local_search(sol)){}
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}

TEST_F(neighborhood_cut_fixture, ErrorCaseOnCoverageUpdate){
    sol.path = {
            {6, -1},
            {49, -1},
            {12, -2},
            {44, -1},
            {18, 23},
            {81, 82},
            {35, 30},
            {46, -2},
            {24, -1},
            {51, 49},
            {7, -1},
            {55, 56},
            {19, 18},
            {44, 43},
            {6, -1},
            {48, -2},
            {0, -2},
            {42, -1},
            {6, 11},
            {78, -2},
            {5, -2},
            {72, 75},
            {28, 27},
            {63, 62},
    };
    n->coverage.reset(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    n->local_search(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
    auto tmp = sol.coverage;
    n->coverage.reset(sol);
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
    EXPECT_TRUE(sol.coverage.isApprox(tmp,1e-9));
}

TEST_F(neighborhood_cut_fixture, ErrorCaseUnsatisfyCoverage){
    sol.path = {{6, -1},
            {49, -1},
            {12, -2},
            {18, -1},
            {19, 20},
            {21, -1},
            {22, 23},
            {77, 75},
            {74, 73},
            {5, -1},
            {11, 10},
            {9, -2},
            {8, 6},
            {42, -2},
            {48, -1},
            {50, 51},
            {52, 53},
            {65, -2},
            {64, 62},
            {9, 10},
            {4, -2}};
    sol.cost = n->geometry.cost(sol);
    n->coverage.reset(sol);
    ASSERT_TRUE((sol.coverage.array() >= P.req.array()).all());
    n->local_search(sol);
    if (!(sol.coverage.array() >= P.req.array()).all()){
        for (auto& s: sol.path)
            std::cerr << s << std::endl;
    }
    EXPECT_TRUE((sol.coverage.array() >= P.req.array()).all());
}
