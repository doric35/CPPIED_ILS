#include "test_fixture.hpp"

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}

TEST_F(cppied_context_fixture, AssertValidTestFiles) {
    ASSERT_TRUE(std::filesystem::exists(seabed_path));
    ASSERT_TRUE(std::filesystem::exists(pod_path));
    ASSERT_TRUE(std::filesystem::exists(req_path));
    ASSERT_TRUE(std::filesystem::exists(config_path));
}

TEST_F(cppied_context_fixture, CorrectConfiguration) {
    EXPECT_EQ(ctx.config.at("TIME"), "600") << "Invalid configuration time";
    EXPECT_EQ(ctx.config.at("NAME"), "test_configuration_123") << "Invalid configuration name";
    EXPECT_EQ(ctx.config.at("ALGORITHM_CONFIG"), "c00000000000") << "Invalid algorithm configuration";
    EXPECT_EQ(ctx.config.at("LKH_EXECUTABLE"),
              std::string(PROJECT_SOURCE_DIR) + "/external/LKH-3.0.14/LKH")
                        << "Invalid configuration lkh executable";
    EXPECT_EQ(ctx.config.at("WORKING_DIRECTORY"),
              std::string(PROJECT_SOURCE_DIR) + "/tests/scratch")
                        << "Invalid configuration working directory";
    EXPECT_EQ(ctx.config.size(), 5) << "Invalid number of keys in configuration";
}

TEST_F(cppied_context_fixture, DataStructureSizes) {
    EXPECT_EQ(P.req.size(), 36);
    EXPECT_EQ(P.seabed.size(), 36);
    EXPECT_EQ(P.seabed.rows(), 6);
    EXPECT_EQ(P.seabed.cols(), 6);
    EXPECT_EQ(P.pod.size(), 4);
    EXPECT_EQ(P.pod.rows(), 4);
    EXPECT_EQ(P.pod.cols(), 1);
    EXPECT_EQ(P.vertex.size(), 84);
    EXPECT_EQ(P.cells.cols(), 36);
    EXPECT_EQ(P.adj.nonZeros(),428);
    EXPECT_EQ(P.s_pod.nonZeros(),144);
}

TEST_F(cppied_context_fixture, ParameterValues){
    EXPECT_EQ(P.max_range, 1);
    EXPECT_EQ(P.initial_position, 6);
    EXPECT_EQ(P.s_pod.coeff(0,6), abs(std::log1p(-0.6)));
    EXPECT_EQ(P.s_pod.coeff(6,6), abs(std::log1p(-0.6)));
    EXPECT_EQ(P.s_pod.coeff(0,48), abs(std::log1p(-0.6)));
    EXPECT_EQ(P.s_pod.coeff(1,48), abs(std::log1p(-0.8)));
    EXPECT_NEAR(P.s_pod.col(41).sum(), abs(std::log1p(-0.99)), 1e-12);
}

TEST_F(cppied_method_fixture, ValidDirections){
    for (int i=0; i<P.vertex.size()/2; i++){
        segment e_i = {i, path_engine::NULL_NODE};
        segment re_i = {i, path_engine::REVERSED_NULL_NODE};
        direction dir = method_instance.geometry.get_direction(e_i);
        direction rdir = method_instance.geometry.get_direction(re_i);
        ASSERT_EQ(dir, direction::E) << "Invalid direction computed: i=" << i;
        ASSERT_EQ(rdir, direction::W) << "Invalid direction computed: i=" << i;
    }
    for (int i=P.vertex.size()/2; i<P.vertex.size(); i++){
        segment e_i = {i, path_engine::NULL_NODE};
        segment re_i = {i, path_engine::REVERSED_NULL_NODE};
        direction dir = method_instance.geometry.get_direction(e_i);
        direction rdir = method_instance.geometry.get_direction(re_i);
        ASSERT_EQ(dir, direction::S) << "Invalid direction computed: i=" << i;
        ASSERT_EQ(rdir, direction::N) << "Invalid direction computed: i=" << i;
    }
}

TEST_F(cppied_method_fixture, SymmetricDistances){
    for (int i=0; i<P.vertex.size(); i++){
        for (int j=0; j< P.vertex.size(); j++){
            segment e_i = {i, path_engine::NULL_NODE};
            segment e_j = {j, path_engine::NULL_NODE};
            segment re_i = {i, path_engine::REVERSED_NULL_NODE};
            segment re_j = {j, path_engine::REVERSED_NULL_NODE};
            cost_t d_ij = method_instance.geometry.dist(e_i, e_j);
            cost_t dr_ij = method_instance.geometry.dist(re_j, re_i);

            EXPECT_EQ(d_ij, dr_ij) << "Non-symmetric distance computed: i=" << i << " j=" << j;

            d_ij = method_instance.geometry.dist(e_i, re_j);
            dr_ij = method_instance.geometry.dist(e_j, re_i);

            EXPECT_EQ(d_ij, dr_ij) << "Non-symmetric distance computed: i=" << i << " j=" << j;
        }
    }
}

TEST_F(cppied_method_fixture, SpecificDistances){
    segment e1 = {1,2};
    segment e2 = {78, 83};
    cost_t d1 = method_instance.geometry.dist(e1, e2);
    cost_t t1 = {4,1};
    ASSERT_EQ(d1, t1);

    e1 = {3,path_engine::NULL_NODE};
    e2 = {78, 83};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {3,1};
    ASSERT_EQ(d1, t1);

    e1 = {1, 2};
    e2 = {3,path_engine::NULL_NODE};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {1,0};
    ASSERT_EQ(d1, t1);

    e1 = {6, 11};
    e2 = {23,18};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {3,2};
    ASSERT_EQ(d1, t1);

    e1 = {79, path_engine::NULL_NODE};
    e2 = {23,18};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {2,1};
    ASSERT_EQ(d1, t1);

    e1 = {80, path_engine::NULL_NODE};
    e2 = {23,18};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {1,1};
    ASSERT_EQ(d1, t1);

    e1 = {23,18};
    e2 = {44,42};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {1,1};
    ASSERT_EQ(d1, t1);

    e1 = {75,73};
    e2 = {62, 63};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {4,2};
    ASSERT_EQ(d1, t1);

    e1 = {75,74};
    e2 = {62, 63};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {3,2};
    ASSERT_EQ(d1, t1);

    e1 = {10,9};
    e2 = {61, path_engine::NULL_NODE};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {1,1};
    ASSERT_EQ(d1, t1);

    e1 = {10,9};
    e2 = {62, 63};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {2,1};
    ASSERT_EQ(d1, t1);

    e1 = {61, path_engine::NULL_NODE};
    e2 = {62,63};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {1,0};
    ASSERT_EQ(d1, t1);

    e1 = {46, path_engine::NULL_NODE};
    e2 = {30,35};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {1,1};
    ASSERT_EQ(d1, t1);

    e1 = {48, 51};
    e2 = {77, 76};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {7,2};
    ASSERT_EQ(d1, t1);

    e2 = {30, 33};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {3,3};
    ASSERT_EQ(d1, t1);

    e1 = {30, 33};
    e2 = {77, 76};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {3,3};
    ASSERT_EQ(d1, t1);

    e1 = {46, path_engine::NULL_NODE};
    e2 = {6, 11};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {7,5};
    ASSERT_EQ(d1, t1);

    e1 = {79, 82};
    e2 = {30, 35};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {9,3};
    ASSERT_EQ(d1, t1);

    e1 = {30, 35};
    e2 = {79, 80};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {7,5};
    ASSERT_EQ(d1, t1);

    e1 = {45, path_engine::NULL_NODE};
    e2 = {29, path_engine::REVERSED_NULL_NODE};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {9,3};
    ASSERT_EQ(d1, t1);

    e1 = {23, 18};
    e2 = {29, path_engine::REVERSED_NULL_NODE};
    d1 = method_instance.geometry.dist(e1, e2);
    t1 = {10,4};
    ASSERT_EQ(d1, t1);
}

TEST_F(cppied_method_fixture, ValidEdgeExtensions){
    segment e1 = {1,2};
    segment e2 = {78, 83};
    method_instance.geometry.extend(e1, e2);
    EXPECT_EQ(e1.target, 5) << "Simple east extension invalid";

    e1 = {1,2};
    e2 = {3, 4};
    method_instance.geometry.extend(e1, e2);
    EXPECT_EQ(e1.target, 2) << "No east extension failed";

    e1 = {5,path_engine::NULL_NODE};
    e2 = {83,path_engine::NULL_NODE};
    method_instance.geometry.extend(e1, e2);
    EXPECT_EQ(e1.target, path_engine::NULL_NODE) << "No east 2 extension failed";

    e1 = {79,path_engine::NULL_NODE};
    e2 = {23,18};
    method_instance.geometry.extend(e1, e2);
    EXPECT_EQ(e1.target, 80) << "South extension failed";

    e1 = {23,18};
    e2 = {44,42};
    method_instance.geometry.extend(e1, e2);
    EXPECT_EQ(e1.target, 18) << "No extension 4 extension failed";

    e1 = {75,73};
    e2 = {62,63};
    method_instance.geometry.extend(e1, e2);
    EXPECT_EQ(e1.target, 73) << "No extension ns extension failed";

    e1 = {10, path_engine::REVERSED_NULL_NODE};
    e2 = {62,63};
    method_instance.geometry.extend(e1, e2);
    EXPECT_EQ(e1.target, 9) << "No extension ns extension failed";

    e1 = {79, 82};
    e2 = {30,35};
    method_instance.geometry.extend(e1, e2);
    EXPECT_EQ(e1.target, 83) << "Extension south failed";
}

TEST_F(cppied_method_fixture, ValidEdgeTurn){
    segment e1 = {6,11};
    segment e2 = {23,18};
    segment turn_resp = method_instance.geometry.turn(e1,e2);
    segment next_resp = {79, path_engine::NULL_NODE};
    EXPECT_EQ(turn_resp, next_resp) << "First south turn failed";

    e1 = {75,73};
    e2 = {62,63};
    turn_resp = method_instance.geometry.turn(e1,e2);
    next_resp = {10, path_engine::REVERSED_NULL_NODE};
    EXPECT_EQ(turn_resp, next_resp) << "nw turn failed";

    e1 = {10,9};
    e2 = {62,63};
    turn_resp = method_instance.geometry.turn(e1,e2);
    next_resp = {61, path_engine::NULL_NODE};
    EXPECT_EQ(turn_resp, next_resp) << "nw turn failed";

    e1 = {46, path_engine::NULL_NODE};
    e2 = {6, 11};
    turn_resp = method_instance.geometry.turn(e1, e2);
    next_resp = {30, path_engine::NULL_NODE};
    EXPECT_EQ(turn_resp, next_resp) << "se turn failed";
}

TEST_F(cppied_method_fixture, ValidSubPathCompletion){
    std::vector<segment> path1 = {{6, 11},
                                           {23,18}};
    std::vector<segment> response1 = {{6, 11},
                                               {79,80},
                                               {23,18}};
    method_instance.geometry.complete(path1);
    auto it_path = path1.begin();
    auto it_resp = response1.begin();
    while (it_path != path1.end()){
        EXPECT_EQ(*it_path, *it_resp) << "Failed first completion";
        it_path++; it_resp++;
    }

    path1 = {{23,18},
            {44,42}};
    response1 = {{23,18},
                 {44,42}};
    method_instance.geometry.complete(path1);
    it_path = path1.begin();
    it_resp = response1.begin();
    while (it_path != path1.end()){
        EXPECT_EQ(*it_path, *it_resp) << "Failed already completed";
        it_path++; it_resp++;
    }

    path1 = {{44,42},
             {48, 51}};
    response1 = {{44,42},
                 {0, path_engine::NULL_NODE},
                 {48, 51}};
    method_instance.geometry.complete(path1);
    it_path = path1.begin();
    it_resp = response1.begin();
    while (it_path != path1.end()){
        EXPECT_EQ(*it_path, *it_resp) << "Failed add e";
        it_path++; it_resp++;
    }

    path1 = {{48, 51},
             {30, 35}};
    response1 = {{48,51},
                 {24, path_engine::REVERSED_NULL_NODE},
                 {46, path_engine::NULL_NODE},
                 {30,35}};
    method_instance.geometry.complete(path1);
    it_path = path1.begin();
    it_resp = response1.begin();
    while (it_path != path1.end()){
        EXPECT_EQ(*it_path, *it_resp) << "Failed add w, s";
        it_path++; it_resp++;
    }

    path1 = {{30, 35},
             {75,73}};
    response1 = {{30, 35},
                 {82, path_engine::REVERSED_NULL_NODE},
                 {29, path_engine::REVERSED_NULL_NODE},
                 {75,73}};
    method_instance.geometry.complete(path1);
    it_path = path1.begin();
    it_resp = response1.begin();
    while (it_path != path1.end()){
        EXPECT_EQ(*it_path, *it_resp) << "Failed add n, w";
        it_path++; it_resp++;
    }

    path1 = {{75,73},
             {62, 63}};
    response1 = {{75,73},
                 {10, 9},
                 {61, 63}};
    method_instance.geometry.complete(path1);
    it_path = path1.begin();
    it_resp = response1.begin();
    while (it_path != path1.end()){
        EXPECT_EQ(*it_path, *it_resp) << "Failed add ww";
        it_path++; it_resp++;
    }
}

TEST_F(cppied_method_fixture, ValidPathCompletion){
    std::vector<segment> path1 = {{6, 11},
                                          {23,18},
                                          {44,42},
                                          {48, 51},
                                          {30, 35},
                                          {75,73},
                                          {62, 63}};
    std::vector<segment> response1 = {{6, 11},
                                               {79,80},
                                          {23,18},
                                          {44,42},
                                          {0, path_engine::NULL_NODE},
                                          {48, 51},
                                          {24, path_engine::REVERSED_NULL_NODE},
                                          {46, path_engine::NULL_NODE},
                                          {30, 35},
                                          {82, path_engine::REVERSED_NULL_NODE},
                                          {29, path_engine::REVERSED_NULL_NODE},
                                          {75,73},
                                          {10, 9},
                                          {61, 63}};
    method_instance.geometry.complete(path1);

    auto it_path = path1.begin();
    auto it_resp = response1.begin();
    while (it_path != path1.end()){
        EXPECT_EQ(*it_path, *it_resp);
        it_path++; it_resp++;
    }

    std::vector<segment> path2 = {{6, 11},
                                           {23,18},
                                           {44,42},
                                           {48, 51},
                                           {30, 35},
                                           {75,73},
                                           {62, 63}};
    std::vector<segment> response2 = {{6, 11},
                                               {79,80},
                                               {23,18},
                                               {44,42},
                                               {0, path_engine::NULL_NODE},
                                               {48, 51},
                                               {24, path_engine::REVERSED_NULL_NODE},
                                               {46, path_engine::NULL_NODE},
                                               {30, 35},
                                               {82, path_engine::REVERSED_NULL_NODE},
                                               {29, path_engine::REVERSED_NULL_NODE},
                                               {75,73},
                                               {10, 9},
                                               {61, 63}};
    method_instance.geometry.complete(path2);
    auto it_path2 = path2.begin();
    auto it_resp2 = response2.begin();
    while (it_path2 != path2.end()){
        EXPECT_EQ(*it_path2, *it_resp2);
        it_path2++; it_resp2++;
    }

}

TEST_F(cppied_method_fixture, SatisfyPathConstraints){
    std::vector<segment> path1 = {{6, 11},
                                               {79,80},
                                               {23,18},
                                               {44,42},
                                               {0, path_engine::NULL_NODE},
                                               {48, 51},
                                               {24, path_engine::REVERSED_NULL_NODE},
                                               {46, path_engine::NULL_NODE},
                                               {30, 35},
                                               {82, path_engine::REVERSED_NULL_NODE},
                                               {29, path_engine::REVERSED_NULL_NODE},
                                               {75,73},
                                               {10, 9},
                                               {61, 63}};
    for (auto p_it = path1.begin(); p_it != std::prev(path1.end()); p_it++){
        EXPECT_TRUE(method_instance.geometry.satisfy(*p_it, *std::next(p_it)));
    }

    bool valid_path = false;
    path1 = {{6, 11},
               {23,18},
               {44,42},
               {48, 51},
               {30, 35},
               {75,73},
               {62, 63}};
    for (auto p_it = path1.begin(); p_it != std::prev(path1.end()); p_it++){
        valid_path &= method_instance.geometry.satisfy(*p_it, *std::next(p_it));
    }
    EXPECT_FALSE(valid_path);

    int valid_count = 0;
    for (auto p_it = path1.begin(); p_it != std::prev(path1.end()); p_it++){
        valid_count += method_instance.geometry.satisfy(*p_it, *std::next(p_it));
    }
    EXPECT_EQ(valid_count, 1);
}

TEST_F(cppied_method_fixture, ValidLexicographicCosts){
    cost_t ref = {5,4};
    cost_t other = {5,4};
    EXPECT_EQ(ref, other);
    
    ref = {8,2};
    other = {5,4};
    EXPECT_GT(ref, other);
    EXPECT_LT(other, ref);
    
    ref = {2,5};
    EXPECT_LT(ref, other);
    EXPECT_GT(other, ref);
    
    ref = {2,2};
    EXPECT_LT(ref, other);
    EXPECT_GT(other, ref);
    
    ref = {2,4};
    EXPECT_LT(ref, other);
    EXPECT_GT(other, ref);
    
    ref = {6,4};
    EXPECT_GT(ref, other);
    EXPECT_LT(other, ref);
    
    ref = {5,5};
    EXPECT_GT(ref, other);
    EXPECT_LT(other, ref);
    
    ref = {5,3};
    EXPECT_LT(ref, other);
    EXPECT_GT(other, ref);
}

TEST_F(cppied_method_fixture, ValidCoverage){
    cppied_solution unit_sol_1 = {{{24, path_engine::REVERSED_NULL_NODE}},
                                         Eigen::VectorXd::Zero(P.seabed.size()),
                                         {1,0}};
    method_instance.coverage.insert(unit_sol_1, unit_sol_1.path.front());
    Eigen::VectorXd response_u1(P.seabed.size());
    response_u1 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.8, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.99, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    response_u1.array() = (-response_u1.array()).log1p().abs();
    ASSERT_TRUE(unit_sol_1.coverage.isApprox(response_u1, 1e-9));

    cppied_solution sol = {{{6, 11},
                            {79,80},
                            {23,18},
                            {44,42},
                            {0, path_engine::NULL_NODE},
                            {48, 51},
                            {24, path_engine::REVERSED_NULL_NODE},
                            {46, path_engine::NULL_NODE},
                            {30, 35},
                            {82, path_engine::REVERSED_NULL_NODE},
                            {29, path_engine::REVERSED_NULL_NODE},
                            {75,73},
                            {10, 9},
                            {61, 63}},
                           Eigen::VectorXd::Zero(P.seabed.size()),
                           {40,13}};
    method_instance.coverage.insert(sol, sol.path.front());
    Eigen::VectorXd response1(P.seabed.size());
    response1 << 0.6, 0.8, 0.99, 0.99, 0.99, 0.99,
            0.6, 0.8, 0.99, 0.99, 0.8, 0.8,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    response1.array() = (-response1.array()).log1p().abs();
    ASSERT_TRUE(sol.coverage.isApprox(response1, 1e-9));

    method_instance.coverage.insert(sol, sol.path[1]);
    Eigen::VectorXd response2(P.seabed.size());
    response2 << 0.6, 0.8, 0.99, 0.99, 0.99, 0.99,
            0.6, 0.8, 0.99, 0.99, 0.8, 0.96,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.8,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    response2.array() = (-response2.array()).log1p().abs();
    ASSERT_TRUE(sol.coverage.isApprox(response2, 1e-9));

    method_instance.coverage.insert(sol, sol.path[2]);
    Eigen::VectorXd response3(P.seabed.size());
    response3 << 0.6, 0.8, 0.99, 0.99, 0.99, 0.99,
            0.6, 0.8, 0.99, 0.99, 0.8, 0.96,
            0.6, 0.8, 0.8, 0.8, 0.8, 0.96,
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    response3.array() = (-response3.array()).log1p().abs();
    ASSERT_TRUE(sol.coverage.isApprox(response3, 1e-9));

    method_instance.coverage.insert(sol, sol.path[3]);
    Eigen::VectorXd response4(P.seabed.size());
    response4 << 0.84, 0.8, 0.99, 0.99, 0.99, 0.99,
            0.84, 0.8, 0.99, 0.99, 0.8, 0.96,
            0.84, 0.8, 0.8, 0.8, 0.8, 0.96,
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    response4.array() = (-response4.array()).log1p().abs();
    ASSERT_TRUE(sol.coverage.isApprox(response4, 1e-9));

    method_instance.coverage.insert(sol, sol.path[4]);
    Eigen::VectorXd response5(P.seabed.size());
    response5 << 0.936, 0.8, 0.99, 0.99, 0.99, 0.99,
            0.84, 0.8, 0.99, 0.99, 0.8, 0.96,
            0.84, 0.8, 0.8, 0.8, 0.8, 0.96,
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    response5.array() = (-response5.array()).log1p().abs();
    ASSERT_TRUE(sol.coverage.isApprox(response5, 1e-9));

    method_instance.coverage.insert(sol, sol.path[5]);
    Eigen::VectorXd response6(P.seabed.size());
    response6 << 0.9744, 0.96, 0.99, 0.99, 0.99, 0.99,
            0.936, 0.96, 0.99, 0.99, 0.8, 0.96,
            0.936, 0.96, 0.8, 0.8, 0.8, 0.96,
            0.96, 0.96, 0.8, 0.8, 0.8, 0.8,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    response6.array() = (-response6.array()).log1p().abs();
    ASSERT_TRUE(sol.coverage.isApprox(response6, 1e-9));

    method_instance.coverage.insert(sol, sol.path[6]);
    Eigen::VectorXd response7(P.seabed.size());
    response7 << 0.9744, 0.96, 0.99, 0.99, 0.99, 0.99,
            0.936, 0.96, 0.99, 0.99, 0.8, 0.96,
            0.936, 0.96, 0.8, 0.8, 0.8, 0.96,
            0.992, 0.96, 0.8, 0.8, 0.8, 0.8,
            0.99, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    response7.array() = (-response7.array()).log1p().abs();
    ASSERT_TRUE(sol.coverage.isApprox(response7, 1e-9));

    //Test full path
    method_instance.coverage.reset(sol);
    Eigen::VectorXd responsef(P.seabed.size());
    responsef << 0.9744, 0.96, 0.99, 0.9999, 0.9999, 0.99,
                0.936, 0.96, 0.9999, 0.999999, 0.992, 0.992,
                0.936, 0.96, 0.96, 0.96, 0.96, 0.992,
                0.992, 0.96, 0.96, 0.96, 0.96, 0.992,
                0.999999, 0.99, 0.99, 0.99, 0.99, 0.999999,
                0.99, 0.99, 0.99, 0.99, 0.99, 0.99;
    responsef.array() = (-responsef.array()).log1p().abs();
    ASSERT_TRUE(sol.coverage.isApprox(responsef, 1e-9));
}

TEST_F(cppied_method_fixture, ValidCoverageExtraction){
    cppied_solution sol = {{{6, 11},
                            {79,80},
                            {23,18},
                            {44,42},
                            {0, path_engine::NULL_NODE},
                            {48, 51},
                            {24, path_engine::REVERSED_NULL_NODE},
                            {46, path_engine::NULL_NODE},
                            {30, 35},
                            {82, path_engine::REVERSED_NULL_NODE},
                            {29, path_engine::REVERSED_NULL_NODE},
                            {75,73},
                            {10, 9},
                            {61, 63}},
                           Eigen::VectorXd::Zero(P.seabed.size()),
                           {40,13}};
    method_instance.coverage.reset(sol);
    method_instance.coverage.remove(sol, sol.path.front());
    Eigen::VectorXd response1(P.seabed.size());
    response1 << 0.936, 0.8, 0.0, 0.99, 0.99, 0.0,
            0.84, 0.8, 0.99, 0.9999, 0.96, 0.96,
            0.936, 0.96, 0.96, 0.96, 0.96, 0.992,
            0.992, 0.96, 0.96, 0.96, 0.96, 0.992,
            0.999999, 0.99, 0.99, 0.99, 0.99, 0.999999,
            0.99, 0.99, 0.99, 0.99, 0.99, 0.99;
    response1.array() = (-response1.array()).log1p().abs();
    ASSERT_TRUE(sol.coverage.isApprox(response1, 1e-9));

    for (int i=1; i< sol.path.size(); i++){
        method_instance.coverage.remove(sol, sol.path[i]);
    }
    Eigen::VectorXd responsef(P.seabed.size());
    responsef << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    responsef.array() = (-responsef.array()).log1p().abs();
    // Print the matrix
    //std::cout << "Matrix m:\n" << sol.coverage << std::endl;
    ASSERT_TRUE(sol.coverage.isZero(1e-12));
}

// ── SetCover tests ────────────────────────────────────────────────────────────

struct cppied_set_cover_fixture : public ::testing::Test {
protected:
    struct testable_instance : public cppied_instance_base{
        using cppied_instance_base::cppied_instance_base;
        using cppied_instance_base::set_cover;
        using cppied_instance_base::transform_log1p;
        using cppied_instance_base::initialize_cells;
        using cppied_instance_base::initialize_vertices;
        using cppied_instance_base::initialize_sparse_pod;
    };
    cppied_set_cover_fixture() : P(
        read_matrix<int>(PROJECT_SOURCE_DIR "/tests/configurations/test_seabed.txt"),
        read_matrix<double>(PROJECT_SOURCE_DIR "/tests/configurations/test_pod.txt"),
        read_matrix<double>(PROJECT_SOURCE_DIR "/tests/configurations/test_req.txt")
    ) {}
    void SetUp() override{
        P.transform_log1p();
        P.max_range = int(P.pod.cols());
        P.initial_position = static_cast<int>(P.seabed.cols() * P.max_range);
        P.initialize_cells();
        P.initialize_vertices();
    }
    testable_instance P;
};

// Horizontal vertex 0: (0.5, 0.0) — top boundary, clamps to row 0 only.
TEST_F(cppied_set_cover_fixture, HorizontalTopBoundary) {
    std::vector<int> C;
    P.set_cover(0, C, true);
    ASSERT_EQ(C.size(), 1u);
    EXPECT_EQ(C[0], 0);  // row0*6+col0
}

// Horizontal vertex 6: (0.5, 1.0) — interior, covers rows 0 and 1 in col 0.
TEST_F(cppied_set_cover_fixture, HorizontalInterior) {
    std::vector<int> C;
    P.set_cover(6, C, true);
    ASSERT_EQ(C.size(), 2u);
    EXPECT_EQ(C[0], 0);   // row0*6+col0
    EXPECT_EQ(C[1], 6);   // row1*6+col0
}

// Horizontal vertex 36: (0.5, 6.0) — bottom boundary, clamps to row 5 only.
TEST_F(cppied_set_cover_fixture, HorizontalBottomBoundary) {
    std::vector<int> C;
    P.set_cover(36, C, true);
    ASSERT_EQ(C.size(), 1u);
    EXPECT_EQ(C[0], 30);  // row5*6+col0
}

// Horizontal vertex 9: (3.5, 1.0) — interior, column 3, rows 0 and 1.
TEST_F(cppied_set_cover_fixture, HorizontalOffsetColumn) {
    std::vector<int> C;
    P.set_cover(9, C, true);
    ASSERT_EQ(C.size(), 2u);
    EXPECT_EQ(C[0], 3);   // row0*6+col3
    EXPECT_EQ(C[1], 9);   // row1*6+col3
}

// Horizontal vertex 21: (3.5, 3.0) — interior, column 3, rows 2 and 3.
TEST_F(cppied_set_cover_fixture, HorizontalMiddle) {
    std::vector<int> C;
    P.set_cover(21, C, true);
    ASSERT_EQ(C.size(), 2u);
    EXPECT_EQ(C[0], 15);  // row2*6+col3
    EXPECT_EQ(C[1], 21);  // row3*6+col3
}

// Vertical vertex 42: (0.0, 0.5) — left boundary, clamps to col 0 only.
TEST_F(cppied_set_cover_fixture, VerticalLeftBoundary) {
    std::vector<int> C;
    P.set_cover(42, C, false);
    ASSERT_EQ(C.size(), 1u);
    EXPECT_EQ(C[0], 0);   // col0*6+row0
}

// Vertical vertex 48: (1.0, 0.5) — interior, row 0, cols 0 and 1.
TEST_F(cppied_set_cover_fixture, VerticalInterior) {
    std::vector<int> C;
    P.set_cover(48, C, false);
    ASSERT_EQ(C.size(), 2u);
    EXPECT_EQ(C[0], 0);   // col0*6+row0
    EXPECT_EQ(C[1], 1);
}

// Vertical vertex 78: (6.0, 0.5) — right boundary, clamps to col 5 only.
TEST_F(cppied_set_cover_fixture, VerticalRightBoundary) {
    std::vector<int> C;
    P.set_cover(78, C, false);
    ASSERT_EQ(C.size(), 1u);
    EXPECT_EQ(C[0], 5);  // col5*6+row0
}

// Vertical vertex 51: (1.0, 3.5) — interior, row 3, cols 0 and 1.
TEST_F(cppied_set_cover_fixture, VerticalOffsetRow) {
    std::vector<int> C;
    P.set_cover(51, C, false);
    ASSERT_EQ(C.size(), 2u);
    EXPECT_EQ(C[0], 18);   // col0*6+row3
    EXPECT_EQ(C[1], 19);   // col1*6+row3
}

// Vertical vertex 60: (3.0, 0.5) — interior, row 0, cols 2 and 3.
TEST_F(cppied_set_cover_fixture, VerticalMiddleColumn) {
    std::vector<int> C;
    P.set_cover(60, C, false);
    ASSERT_EQ(C.size(), 2u);
    EXPECT_EQ(C[0], 2);  // col2*6+row0
    EXPECT_EQ(C[1], 3);  // col3*6+row0
}

TEST_F(cppied_set_cover_fixture, ConstructSparsePodNoThrow){
    EXPECT_NO_THROW(P.initialize_sparse_pod());
}

