#pragma once

#include "gurobi_c++.h"
#include "../cppied_method_base.hpp"
#include "../metaheuristics/ils.hpp"

namespace lp {
    struct node{
        std::vector<GRBVar> outgoing_variables;
        std::vector<int> outgoing_arcs_V1;
        std::map<int,int> vertex_to_arc;
        std::vector<double> variables_flow;
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
        cppied_method_base(c, i), no_rel(false), starting_minus(false){};

    friend class subtour_elimination;

    void initialize() override {
        auto config_item = ctx.config.find("ALGORITHM_CONFIG");
        if (config_item == ctx.config.end())
            throw std::runtime_error("Did not identify ilp solver configuration with c0 or c1.\n");
        std::string configuration = config_item->second;
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

    void retrieve_solution(cppied_solution& pSolution,
                           const std::function<void()>& set_flow_variables);
    void find_source_to_target(cppied_solution& pSolution,
                               std::list<int>& path);
    void find_cycle(cppied_solution& pSolution,
                    std::list<int>& path,
                    int current_node,
                    bool curr_minus);
    void set_solution(cppied_solution& pSolution, std::list<int>& path);

protected:
    std::vector<lp::node> variables;
    std::vector<lp::V_minus> minus_sets;
    std::vector<lp::V_plus> plus_sets;
    lp::node source;
    std::vector<GRBVar> targets_minus;
    std::vector<double> targets_minus_flow;
    std::vector<GRBVar> targets_plus;
    std::vector<double> targets_plus_flow;

    bool no_rel;
    bool starting_minus;

    GRBLinExpr Z1=0;
    GRBLinExpr Z2=0;
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
    double M = 0.0;
    void callback() override {
        try {
            if (where == GRB_CB_MIPSOL) {
                //Found an integer solution, identify subtours;
                M = std::max(M, getDoubleInfo(GRB_CB_MIPSOL_OBJ));
                auto flow_setter = [&](){
                    for (int u=0; u<math_program.problem.vertex.size(); ++u){
                        for (int v =0; v<math_program.variables[u].outgoing_variables.size(); ++v){
                            math_program.variables[u].variables_flow[v] =
                                    getSolution(math_program.variables[u].outgoing_variables[v]);
                        }
                        math_program.targets_minus_flow[u] = getSolution(math_program.targets_minus[u]);
                        math_program.targets_plus_flow[u] = getSolution(math_program.targets_plus[u]);
                    }
                };
                //std::cout << "Before CB retrieve" << std::endl;
                math_program.retrieve_solution(dummy, flow_setter);
                //std::cout << "After CB retrieve" << std::endl;
                std::pair<int, int> root_edge = subtour_root();
                if (path_engine::is_node(root_edge.first)) {
                    bool minus;
                    if (std::find(math_program.minus_sets[root_edge.first].V.begin(),
                                  math_program.minus_sets[root_edge.first].V.end(),
                                  root_edge.second)
                        != math_program.minus_sets[root_edge.first].V.end()) {
                        minus = false;
                    } else
                        minus = true;
                    std::list<int> subtour;
                    math_program.find_cycle(dummy, subtour,
                                            root_edge.first, minus);
                    if (!subtour.empty())
                        add_subtour_elimination_constraint(root_edge.first, minus,
                                                           subtour);
                    else throw std::runtime_error("Entered subtour elimination routine but no subtour was extracted.\n");
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

    void add_subtour_elimination_constraint(int root, bool root_minus,
                                            std::list<int>& subtour){
        std::set<int> S(subtour.begin(), subtour.end());
        GRBLinExpr parity_outgoing_arcs_sum = 0.0;
        std::vector<GRBVar*> candidate_constraints;
        candidate_constraints.reserve(S.size());

        auto process_node = [&](int from, bool curr_minus){
            const auto& exit_set = curr_minus ? math_program.plus_sets[from].V :
                                                                    math_program.minus_sets[from].V;

            for (int w : exit_set){
                auto it = math_program.variables[from].vertex_to_arc.find(w);
                if (it == math_program.variables[from].vertex_to_arc.end()) continue;
                auto& var = math_program.variables[from].outgoing_variables[it->second];
                if (S.contains(w))
                    candidate_constraints.push_back(&var);
                else
                    parity_outgoing_arcs_sum += var;
            }
            if (curr_minus)
                parity_outgoing_arcs_sum += math_program.targets_plus[from];
            else
                parity_outgoing_arcs_sum += math_program.targets_minus[from];
        };

        int prev = root;
        bool curr_minus = root_minus;
        for (int node : subtour){
            process_node(prev, curr_minus);
            curr_minus = std::find(math_program.minus_sets[node].V.begin(),
                                 math_program.minus_sets[node].V.end(), prev)
                       != math_program.minus_sets[node].V.end();
            prev = node;
        }
        parity_outgoing_arcs_sum = M * parity_outgoing_arcs_sum;
        auto add_constraint = [&](GRBVar* v){
            addLazy(parity_outgoing_arcs_sum - *v >= 0);
        };
        std::for_each(candidate_constraints.begin(),
                      candidate_constraints.end(),
                      add_constraint);
    }
};