#pragma once

#include "neighborhood.hpp"
#include "gurobi_c++.h"

struct replacement_set{
    int r;
    std::vector<segment> candidates;
    std::vector<GRBVar> variables;
};

class neighborhood_r : public neighborhood {
public:
    using neighborhood::neighborhood;

    bool local_search() override;
    void select(std::vector<int>& selection);
    void set_initial_replacements(std::vector<int>& selection,
                      std::vector<replacement_set>& rep,
                      GRBModel& model);
    void add_replacements(std::vector<segment>& candidates);
    void set_constraints(std::vector<replacement_set>& R,
                         std::vector<GRBConstr>& constraints,
                         GRBModel& model);
    void extract_duals(std::vector<GRBConstr>& constraints,
                       std::vector<double>& coverage_duals,
                       std::vector<double>& R_duals);

    cost_t gain(sVecIt source);
    cost_t replacement_cost(sVecIt source, segment target);
    void positions_coverage_rcs(std::vector<double>& pRC,
                                std::vector<double>& coverage_duals);
    bool add_stage_columns(std::vector<GRBConstr>& constraints,
                                std::vector<replacement_set>& R,
                                GRBModel& model,
                                int stage);
    void add_stage_column(std::vector<GRBConstr>& constraints,
                                 replacement_set& R,
                                int r_ctr,
                                 GRBModel& model);
    void retrieve_solution(std::vector<replacement_set>& R);

    void set_warm_start(std::vector<replacement_set>& R);

    std::pair<segment, double> dag_heuristic(
            int u, int v,
            std::vector<double>& coverage_duals,
            int r,
            double r_dual,
            double gamma_dual,
            const std::function<double(cost_t)>& f);

    std::pair<segment, double> null_dag_heuristic(
            int u, int v,
            std::vector<double>& coverage_duals,
            int r,
            double r_dual,
            double gamma_dual,
            const std::function<double(cost_t)>& f
            );
    std::pair<segment, double> reversed_dag_heuristic(
            int u, int v,
            std::vector<double>& coverage_duals,
            int r,
            double r_dual,
            double gamma_dual,
            const std::function<double(cost_t)>& f
    );

protected:
    GRBLinExpr Z1 = 0;
    GRBLinExpr Z2 = 0;
};