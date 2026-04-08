#pragma once

#include "gurobi_c++.h"
#include "../cppied_method_base.hpp"
#include "../metaheuristics/ils.hpp"

namespace lp {
    struct node{
        std::vector<GRBVar> outgoing_variables;
        std::vector<int> outgoing_arcs_V1;
        std::map<int,int> vertex_to_arc;
        std::vector<int> variables_flow;
    };

    struct V_minus{
        std::vector<int> V;
    };

    struct V_plus{
        std::vector<int> V;
    };
}

class ilp : public cppied_method_base {
public:
    ilp(cppied_context& c, cppied_instance& i) :
        cppied_method_base(c, i), ils_heuristic(c, i){};

    void initialize() override {
        std::string configuration = ctx.config["SOLVER_CONFIG"];
        assert(configuration.size() == 3);
        if (configuration[1] == '1')
            ils_start = true;
        if (configuration[2] == '1')
            no_rel = true;
        if (ils_start)
            ils_heuristic.initialize();

    }

    void terminate() override {

    }

    void d_solve(cppied_solution& pSolution) override;
    void set_variables(cppied_solution& pSolution, GRBModel& model);
    void set_constraints(cppied_solution& pSolution, GRBModel& model);
    void set_objectives(cppied_solution& pSolution, GRBModel& model);
    void set_flow_sets(cppied_solution& pSolution);
    void set_warm_start(cppied_solution& pSolution, GRBModel& model);
    void retrieve_solution(cppied_solution& pSolution);
    void find_source_to_target(cppied_solution& pSolution, std::list<int>& path);
    void find_cycle(cppied_solution& pSolution,
                    std::list<int>& path,
                    int current_node,
                    bool curr_minus);
    void set_solution(cppied_solution& pSolution, std::vector<int>& path);

protected:
    GRBLinExpr Z1=0;
    GRBLinExpr Z2=0;

    std::vector<lp::node> variables;
    std::vector<lp::V_minus> minus_sets;
    std::vector<lp::V_plus> plus_sets;
    lp::node source;
    std::vector<GRBVar> targets_minus;
    std::vector<GRBVar> targets_plus;
    ils ils_heuristic;

    bool starting_minus;

    bool ils_start;
    bool no_rel;
};