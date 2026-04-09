#include "../test_fixture.hpp"

TEST_F(cppied_method_fixture, ValidRemovalGain){
    cppied_solution sol{
            {{6, 11},
             {23,18},
             {44,42},
             {48, 51},
             {30, 35},
             {75,73},
             {62, 63}},
     Eigen::VectorXd::Zero(P.req.size()),
     cost_t{0,0}
    };
    method_instance.coverage.reset(sol);
    sol.cost = method_instance.geometry.cost(sol);
    EXPECT_EQ(method_instance.geometry.removal_gain(sol,
                                                    std::next(sol.path.begin(),5)),
              (cost_t{2,2}));

    sol.path = {{6, 11},
                {23,18},
                {44,42},
                {48, 51},
                {30, 33},
                {77, 76},
                {75,73},
                {62, 63}};
    sol.cost = method_instance.geometry.cost(sol);
    EXPECT_EQ(method_instance.geometry.removal_gain(sol,
                                                    std::next(sol.path.begin(),4)),
              (cost_t{2,4}));

}

TEST_F(cppied_method_fixture, ValidInsertionGain){
    cppied_solution sol{
            {{6, 11},
             {79, path_engine::NULL_NODE},
             {23,18},
             {44,42},
             {48, 51},
             {30, 35},
             {75,74},
             {10, 9},
             {62, 63}},
            Eigen::VectorXd::Zero(P.req.size()),
            cost_t{0,0}
    };
    method_instance.coverage.reset(sol);
    sol.cost = method_instance.geometry.cost(sol);
    EXPECT_EQ(method_instance.geometry.insert_between_gain(sol.path[6],
                                                           {16,15},
                                                           sol.path[8]),
              (cost_t{0,0}));
}

TEST_F(cppied_method_fixture, ValidTopInsertion){
    cppied_solution sol{
            {{6, 11},
             {23,18},
             {44,42},
             {48, 51},
             {30, 35},
             {75,74},
             //{73, path_engine::REVERSED_NULL_NODE},
             {62, 63}},
            Eigen::VectorXd::Zero(P.req.size()),
            cost_t{0,0}
    };
    method_instance.coverage.reset(sol);
    sol.cost = method_instance.geometry.cost(sol);
    segment s = {16, path_engine::REVERSED_NULL_NODE};
    auto response =
            method_instance.geometry.get_top_k_insertion(sol, s, 1);
    EXPECT_EQ(response[0].second, (cost_t{0,0}));
    EXPECT_EQ(response[0].first, 6);

    response = method_instance.geometry.get_top_k_insertion(sol, s, 2);
    EXPECT_EQ(response[0].second, (cost_t{0,0}));
    EXPECT_EQ(response[0].first, 6);
}

TEST_F(cppied_method_fixture, ValidCompletions){
    int n_nodes = int(P.vertex.size());
    cppied_solution sol{};
    for (int i=0; i< n_nodes; i++){
        segment s_i = {i, path_engine::NULL_NODE};
        segment s_ir = {i, path_engine::REVERSED_NULL_NODE};
        for (int j=0; j< n_nodes; j++){
            segment s_j = {j, path_engine::NULL_NODE};
            segment s_jr = {j, path_engine::REVERSED_NULL_NODE};
            cost_t d_ij = method_instance.geometry.dist(s_i, s_j);
            std::vector<segment> path_ij{s_i, s_j};
            EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
                "Throwed with " << s_i << " " << s_j;
            sol.path = path_ij;
            EXPECT_EQ(method_instance.geometry.cost(sol), d_ij);
            cost_t d_ijr = method_instance.geometry.dist(s_i, s_jr);
            std::vector<segment> path_ijr{s_i, s_jr};
            EXPECT_NO_THROW(method_instance.geometry.complete(path_ijr)) <<
            "Throwed with " << s_i << " " << s_jr;
            sol.path = path_ijr;
            EXPECT_EQ(method_instance.geometry.cost(sol), d_ijr);
            cost_t d_irj = method_instance.geometry.dist(s_ir, s_j);
            std::vector<segment> path_irj{s_ir, s_j};
            EXPECT_NO_THROW(method_instance.geometry.complete(path_irj)) <<
                "Throwed with " << s_ir << " " << s_j;
            sol.path = path_irj;
            EXPECT_EQ(method_instance.geometry.cost(sol), d_irj);
            cost_t d_irjr = method_instance.geometry.dist(s_ir, s_jr);
            std::vector<segment> path_irjr{s_ir, s_jr};
            EXPECT_NO_THROW(method_instance.geometry.complete(path_irjr)) <<
                "Throwed with " << s_ir << " " << s_jr;;
            sol.path = path_irjr;
            EXPECT_EQ(method_instance.geometry.cost(sol), d_irjr);
        }
    }
}

TEST_F(cppied_method_fixture, ValidDistancesComputations){
    segment s1{66, -1}, s2{66, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={66, -1}, s2={66, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,4}));

    s1={63, -1}, s2 = {63, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={63, -1}, s2={63, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,4}));

    s1={63, -1}, s2={63, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,4}));

    s1={26, path_engine::REVERSED_NULL_NODE}, s2={63, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,3}));

    s1={57, path_engine::REVERSED_NULL_NODE}, s2={63, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,2}));

    s1={20, path_engine::NULL_NODE}, s2={63, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{1,1}));

    s1={63, 64}, s2={63, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={27, -2}, s2={27, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,4}));

    s1={27, 26}, s2={27, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={5, -1}, s2={5, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{7,6}));

    s1={78, -1}, s2={5, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,5}));

    s1={11, -2}, s2={5, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={73, -1}, s2={5, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={17, -1}, s2={5, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,2}));

    s1={79, -2}, s2={5, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,1}));

    s1={36, -1}, s2={36, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={37, -1}, s2={36, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,4}));

    s1={59, -2}, s2={36, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,3}));

    s1={31, -2}, s2={36, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,2}));

    s1={0, path_engine::REVERSED_NULL_NODE}, s2={48, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,5}));

    s1={42, path_engine::NULL_NODE}, s2={48, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={6, path_engine::NULL_NODE}, s2={48, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={7, path_engine::NULL_NODE}, s2={48, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,3}));

    s1={0, path_engine::NULL_NODE}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,5}));

    s1={48, path_engine::NULL_NODE}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={7, path_engine::NULL_NODE}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={54, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,2}));

    s1={1, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,1}));

    s1={0, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{1,1}));

    s1={0, 1}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,5}));

    s1={54, -1}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,4}));

    s1={6, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,5}));

    s1={43, -1}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={12, -1}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={49, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,2}));

    s1={48, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,2}));

    s1={0, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{1,1}));

    s1={42, -2}, s2={42, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{7,6}));

    s1={6, -1}, s2={43, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={48, -2}, s2={43, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,2}));

    s1={0, -2}, s2={43, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,1}));

    s1={42, -1}, s2={43, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{1,0}));

    s1={6, 7}, s2={43, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,3}));

    s1={6, -2}, s2={55, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,3}));

    s1={36, -1}, s2={30, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={53, -2}, s2={30, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={52, -2}, s2={30, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,3}));

    s1={24, -2}, s2={30, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{2,2}));

    s1={46, -1}, s2={30, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{1,1}));

    s1={36, 37}, s2={30, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,4}));

    s1={36, -1}, s2={47, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,5}));

    s1={36, -2}, s2={47, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{6,5}));

    s1={47, -2}, s2={47, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{5,4}));

    s1={30, -1}, s2={47, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{4,3}));

    s1={52, -2}, s2={47, -1};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{3,2}));

    s1={42, -2}, s2={50, -2};
    EXPECT_EQ(method_instance.geometry.dist(s1, s2), (cost_t{7,4}));
}

TEST_F(cppied_method_fixture, SpecificPathCompletions){
    segment s_i = {63, path_engine::NULL_NODE};
    segment s_j = {63, path_engine::NULL_NODE};
    std::vector<segment> path_ij{s_i, s_j};
    EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
        "Throwed with " << s_i << " " << s_j;

    s_i = {36, path_engine::NULL_NODE}, s_j = {36, path_engine::REVERSED_NULL_NODE};
    path_ij = {s_i, s_j};
    EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
        "Throwed with " << s_i << " " << s_j;

    s_i = {0, path_engine::NULL_NODE}, s_j = {48, path_engine::REVERSED_NULL_NODE};
    path_ij = {s_i, s_j};
    EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
        "Throwed with " << s_i << " " << s_j;

    s_i = {0, path_engine::NULL_NODE}, s_j = {42, path_engine::NULL_NODE};
    path_ij = {s_i, s_j};
    EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
        "Throwed with " << s_i << " " << s_j;

    s_i = {6, -2}, s_j = {42, -1};
    path_ij = {s_i, s_j};
    EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
        "Throwed with " << s_i << " " << s_j;

    s_i = {6, -1}, s_j = {43, -1};
    path_ij = {s_i, s_j};
    EXPECT_NO_THROW(method_instance.geometry.complete(path_ij)) <<
        "Throwed with " << s_i << " " << s_j;
}