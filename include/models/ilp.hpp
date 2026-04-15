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
        cppied_method_base(c, i){};

    friend class subtour_elimination;

    void initialize() override {
        std::string configuration = ctx.config["SOLVER_CONFIG"];
        assert(configuration.size() == 2);
        if (configuration[1] == '1')
            no_rel = true;
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
    void set_solution(cppied_solution& pSolution, std::list<int>& path);

protected:
    GRBLinExpr Z1=0;
    GRBLinExpr Z2=0;

    std::vector<lp::node> variables;
    std::vector<lp::V_minus> minus_sets;
    std::vector<lp::V_plus> plus_sets;
    lp::node source;
    std::vector<GRBVar> targets_minus;
    std::vector<GRBVar> targets_plus;

    bool starting_minus;

    bool ils_start;
    bool no_rel;
};

class subtour_elimination: public GRBCallback{
public:
    ilp& math_program;
    friend class ilp;
    explicit subtour_elimination(ilp& pProgram) : math_program(pProgram), dummy(){
        dummy.path = {};
        dummy.coverage = Eigen::VectorXd::Zero(math_program.problem.req.size());
        dummy.cost = {0,0};
    }

protected:
    cppied_solution dummy;
    const int M = 10;
    void callback() override {
        try {
            if (where == GRB_CB_MIPSOL) {
                //Found an integer solution, identify subtours;
                math_program.retrieve_solution(dummy);
                std::pair<int, int> root_edge = subtour_root();
                if (path_engine::is_node(root_edge.first)) {
                    std::list<int> subtour;
                    math_program.find_cycle(dummy, subtour,
                                            root_edge.second, true);
                    bool minus;
                    if (std::find(math_program.minus_sets[root_edge.second].V.begin(),
                                  math_program.minus_sets[root_edge.second].V.end(),
                                  root_edge.first)
                        != math_program.minus_sets[root_edge.second].V.end()) {
                        minus = true;
                    } else
                        minus = false;
                    math_program.variables[root_edge.first].variables_flow[root_edge.second] -= 1;
                    subtour = {root_edge.first, root_edge.second};
                    math_program.find_cycle(dummy, subtour,
                                            root_edge.second, minus);
                    add_subtour_elimination_constraint(subtour);
                }
            }
        }
        catch (GRBException& e) {
            std::cerr << "Error number: " << e.getErrorCode() << std::endl;
            std::cerr << e.getMessage() << std::endl;
            throw e;
        }
        catch (std::runtime_error& e){
            std::cerr << "Error occured during callback" << std::endl;
            throw e;
        }

    }

    std::pair<int,int> subtour_root(){
        for (int u=0; u<math_program.problem.vertex.size(); ++u){
            for (int v =0; v<math_program.variables[u].outgoing_variables.size(); ++v){
                if (math_program.variables[u].variables_flow[v] > 0.5)
                    return {u, math_program.variables[u].outgoing_arcs_V1[v]};
            }
        }
        return {path_engine::NULL_NODE, path_engine::NULL_NODE};
    }

    void add_subtour_elimination_constraint(std::list<int>& subtour){
        std::vector<bool> S(math_program.problem.vertex.size(), false);

        std::set<int> subtour_set(subtour.begin(), subtour.end());

        auto add_v = [&](int v){
            S[v] = true;
        };
        std::for_each(subtour_set.begin(), subtour_set.end(), add_v);

        int u;
        std::sample(subtour_set.begin(), subtour_set.end(),
                            &u, 1, math_program.rng);

        GRBLinExpr outgoing_arcs_sum = 0.0;
        auto add_arcs_contribution = [&](int u){
            for (int v =0; v<math_program.variables[u].outgoing_arcs_V1.size(); v++){
                if (!S[math_program.variables[u].outgoing_arcs_V1[v]])
                    outgoing_arcs_sum += math_program.variables[u].outgoing_variables[v];
            }
        };

        outgoing_arcs_sum += math_program.targets_minus[u] +
                math_program.targets_plus[u];

        std::for_each(subtour_set.begin(), subtour_set.end(), add_arcs_contribution);
        for (int v =0; v<math_program.variables[u].outgoing_arcs_V1.size(); v++)
            if (S[math_program.variables[u].outgoing_arcs_V1[v]])
                addLazy((M * outgoing_arcs_sum) - math_program.variables[u].outgoing_variables[v] >= 0);
    }
};