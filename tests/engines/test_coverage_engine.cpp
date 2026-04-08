#include "../test_fixture.hpp"

TEST_F(cppied_method_fixture, ValidUnsatCells){
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
    std::vector<int> unsat;
    method_instance.coverage.remove(sol, sol.path[5]);
    method_instance.coverage.unsatisfied_cells(sol, unsat);
    std::sort(unsat.begin(), unsat.end());
    std::vector<int> expected = {10,11,16,17,22,23};
    EXPECT_EQ(unsat, expected);
}

TEST_F(cppied_method_fixture, ValidUnsatBoxFromCells){
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
    iRectangle box = {
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::min(),
            std::numeric_limits<int>::min(),
    };
    method_instance.coverage.remove(sol, sol.path[5]);
    std::vector<int> unsat;
    method_instance.coverage.unsatisfied_cells(sol, unsat);
    method_instance.coverage.unsatisfied_rectangle(unsat, box);
    EXPECT_EQ(box.ul.x, 4);
    EXPECT_EQ(box.lr.x, 6);
    EXPECT_EQ(box.ul.y, 1);
    EXPECT_EQ(box.lr.y, 4);
}
