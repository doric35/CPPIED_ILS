#include "../../include/neighborhoods/neighborhood_r.hpp"

bool neighborhood_r::local_search(cppied_solution &pSol) {
    std::vector<int> selection;
    cost_t init_cost = pSol.cost;

    selection.reserve(pSol.path.size() / 2 +1);
    select(pSol, selection);
    Z1.clear();
    Z2.clear();
    GRBEnv reallocation_env = GRBEnv(true);
    auto log_item = ctx.config.find("WORKING_DIRECTORY");
    if (log_item != ctx.config.end()){
        std::filesystem::path log_path = log_item->second;
        log_path = log_path / "reallocation.log";
        reallocation_env.set("LogFile", log_path);
    } else
        reallocation_env.set("LogFile", "reallocation.log");
    reallocation_env.set(GRB_IntParam_OutputFlag, 0);
    reallocation_env.set(GRB_IntParam_LogToConsole, 0);
    reallocation_env.start();
    GRBModel model = GRBModel(reallocation_env);

    std::vector<replacement_set> replacements;
    replacements.reserve(selection.size());
    set_initial_replacements(pSol,
                             selection,
                             replacements,
                             model);

    std::vector<GRBConstr> constraints;
    set_constraints(pSol, replacements, constraints, model);
    model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);

    model.update();
    bool converged = false;
    double t;
    while (!converged){
        model.setObjective(Z1, GRB_MINIMIZE);
        t = remaining_time();
        if (t < 0.0) {
            // Remaining time too low, skip this optimization
            coverage.reset(pSol);
            return false;  // or break/return from local_search
        }
        model.set(GRB_DoubleParam_TimeLimit, t + 1.0);
        model.update();
        model.optimize();
        int status = model.get(GRB_IntAttr_Status);
        if (status != GRB_OPTIMAL){
            std::cerr << "[Warning] First phase LP terminated with status" << status << "\n";
            break;
        }
        bool converged_z1 = add_stage_columns(pSol, constraints, replacements, model, 1);
        double obj_value = model.get(GRB_DoubleAttr_ObjVal);
        constraints.back().set(GRB_DoubleAttr_RHS, obj_value);
        model.setObjective(Z2, GRB_MINIMIZE);
        t = remaining_time();
        if (t < 0.0) {
            // Remaining time too low, skip this optimization
            coverage.reset(pSol);
            return false;  // or break/return from local_search
        }
        model.set(GRB_DoubleParam_TimeLimit, t + 1.0);
        model.update();
        model.optimize();
        status = model.get(GRB_IntAttr_Status);
        if (status != GRB_OPTIMAL){
            std::cerr << "[Warning] Second phase LP terminated with status" << status << "\n";
            break;
        }
        bool converged_z2 = add_stage_columns(pSol, constraints, replacements, model, 2);
        converged = converged_z1 && converged_z2;
    }
    model.setObjective(GRBLinExpr(0.0));
    model.setObjectiveN(Z1, 0, 2);
    model.setObjectiveN(Z2, 1, 1);
    //Solve binary
    //set variables to binary
    for (auto& R : replacements){
        for (auto& var:R.variables)
            var.set(GRB_CharAttr_VType, GRB_BINARY);
    }

    // Reset gamma constraint to original incumbent before the binary solve.
    constraints.back().set(GRB_DoubleAttr_RHS, init_cost.length);

    t = remaining_time();
    if (t < 0.0) {
        // Remaining time too low, skip this optimization, restore solution
        coverage.reset(pSol);
        return false;  // or break/return from local_search
    }
    model.set(GRB_DoubleParam_TimeLimit, t + 1.0);
    set_warm_start(pSol, replacements);
    model.update();
    model.optimize();
    //Retrieve solution
    int status = model.get(GRB_IntAttr_Status);
    if (status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL) {
        // A feasible solution exists, safe to retrieve
        retrieve_solution(pSol, replacements);
    } else if (status == GRB_INF_OR_UNBD || status == GRB_INFEASIBLE) {
        // No feasible solution found
        std::cerr << "[Error] Model infeasible or unbounded\n";
        throw std::runtime_error("Infeasibility or unboundedness found in reallocation model.\n");
    } else {
        // Possibly interrupted due to TimeLimit
        std::cerr << "[Warning] Optimization interrupted in reallocation model, using best solution so far\n";
        coverage.reset(pSol);
    }
    return pSol.cost < init_cost;
}

void neighborhood_r::retrieve_solution(cppied_solution &pSol,
                                       std::vector<replacement_set> &R) {
    std::vector<segment> new_sol;
    new_sol.reserve(pSol.path.size());
    int i=0;
    int r=0;
    for (; i < pSol.path.size(); ++i){
        if ( r< R.size() && i==R[r].r){
            for (int j=0; j< R[r].variables.size(); j++){
                double val = R[r].variables[j].get(GRB_DoubleAttr_X);
                if (val > 0.5){
                    new_sol.push_back(R[r].candidates[j]);
                    coverage.insert(pSol, new_sol.back());
                    break;
                }
            }
            ++r;
        } else {
            new_sol.push_back(pSol.path[i]);
        }
    }
    pSol.path.swap(new_sol);
    pSol.cost = geometry.cost(pSol);
}

void neighborhood_r::extract_duals(const cppied_solution& pSol,
                                   std::vector<GRBConstr>& constraints,
                                   std::vector<double>& coverage_duals,
                                   std::vector<double>& R_duals) {
    for (int i=0; i < geometry.n_cols * geometry.n_rows; i++)
        coverage_duals[i] = constraints[i].get(GRB_DoubleAttr_Pi);

    int offset = geometry.n_cols * geometry.n_rows;
    for (int i = 0; i< R_duals.size(); i++)
        R_duals[i] = constraints[i + offset].get(GRB_DoubleAttr_Pi);
}

void neighborhood_r::set_constraints(cppied_solution &pSol,
                                     std::vector<replacement_set> &R,
                                     std::vector<GRBConstr>& constraints,
                                     GRBModel &model) {
    for (auto& r: R)
        coverage.remove(pSol,pSol.path[r.r]);

    std::vector<GRBLinExpr> lin_expr_set(geometry.n_rows*geometry.n_cols,0);
    constraints.clear();
    constraints.reserve(geometry.n_rows*geometry.n_cols + R.size() + 1);

    //coverage_constraints lhs
    for (auto& r : R){
        for (int i=0; i< r.candidates.size(); i++){
            auto add_coverage_coef = [&](
                    cppied_solution& pSol, int v){
                for (SMdIt c(problem.s_pod, v); c; ++c){
                    lin_expr_set[c.index()] += r.variables[i] * c.value();
                }
            };
            coverage.apply(pSol, r.candidates[i], add_coverage_coef);
        }
    }

    //Coverage constraints rhs
    for (int i=0; i< lin_expr_set.size(); i++){
        GRBLinExpr rhs = problem.req(i) - pSol.coverage(i);
        constraints.push_back(model.addConstr(lin_expr_set[i],
                                              GRB_GREATER_EQUAL,
                                              rhs));
    }

    //Capacity constraints
    for (auto & i : R){
        GRBLinExpr rhs = 1;
        GRBLinExpr lhs = 0;
        for (auto& var : i.variables)
            lhs += var;
        constraints.push_back(model.addConstr(lhs, GRB_LESS_EQUAL, rhs));
    }

    //Objective constraint
    model.update();
    cost_t acc{0,0};
    auto func = [&](cost_t reduction, const replacement_set& rep){
        return reduction + replacement_cost(pSol, std::next(pSol.path.cbegin(), rep.r), rep.candidates.front());
    };
    acc = std::accumulate(R.begin(), R.end(), acc, func);
    GRBLinExpr rhs = acc.length;
    GRBLinExpr lhs = 0;
    for(auto & i : R) {
        for (auto &var: i.variables)
            lhs += var.get(GRB_DoubleAttr_Obj) * var;
    }
    constraints.push_back(model.addConstr(lhs, GRB_LESS_EQUAL, rhs));
}

void neighborhood_r::select(cppied_solution &pSol,
                            std::vector<int> &selection) {
    std::vector<cost_t> gains(pSol.path.size(), cost_t{0,0});

    int i=1;
    auto path_it = std::next(pSol.path.begin());
    for(; i< pSol.path.size()-1; ++i, ++path_it)
        gains[i] = gain(pSol, path_it);

    std::vector<int> candidates(pSol.path.size() - 1);
    std::iota(candidates.begin(), candidates.end(), 1);

    auto splitter = [&](int i){
        return i % 2 == 0;
    };

    std::vector<int> odds;
    odds.reserve(candidates.size() / 2 + 1);
    std::vector<int> evens;
    evens.reserve(candidates.size() / 2 + 1);
    std::partition_copy(candidates.begin(),candidates.end(),
                        std::back_inserter(evens),
                        std::back_inserter(odds),
                        splitter);
    std::vector<cost_t> cumulative_gains(2, cost_t{0,0});
    auto func = [&](cost_t acc, int i){
        return acc + gains[i];
    };
    cumulative_gains[0] = std::accumulate(evens.begin(), evens.end(),
                cumulative_gains[0], func);
    cumulative_gains[1] = std::accumulate(odds.begin(), odds.end(),
                cumulative_gains[1], func);

    int s = ris_heuristic::softmax_sample(cumulative_gains, rng);
    if (s)
        selection.swap(evens);
    else
        selection.swap(odds);
}

void neighborhood_r::set_initial_replacements(cppied_solution &pSol,
                                              std::vector<int> &selection,
                                              std::vector<replacement_set> &rep,
                                              GRBModel& model) {
    rep.clear();
    rep.reserve(selection.size());
    for (int i: selection){
        rep.push_back({i, {pSol.path[i]}, {}});
        add_replacements(pSol, rep.back().candidates);
        rep.back().variables.reserve(rep.back().candidates.size());
        for (auto& r : rep.back().candidates){
            segment other = geometry.flip_segment(r);
            cost_t costs = replacement_cost(pSol, pSol.path.cbegin() + i, r);
            cost_t costs_others = replacement_cost(pSol, pSol.path.cbegin() + i, other);
            if (costs_others < costs) {
                r = other;
                costs = costs_others;
            }
            rep.back().variables.push_back(model.addVar(0.0,
                                                        1.0,
                                                        costs.length,
                                                        GRB_CONTINUOUS));
            Z1 += costs.length * rep.back().variables.back();
            Z2 += costs.turns * rep.back().variables.back();
        }
    }
}

void neighborhood_r::add_replacements(cppied_solution& pSol,
                                         std::vector<segment> &candidates) {
    coverage.remove(pSol, candidates.front());
    std::vector<int> unsat;
    coverage.unsatisfied_cells(pSol, candidates.front(), unsat);
    iRectangle box = {
            {std::numeric_limits<int>::max(),
                    std::numeric_limits<int>::max()},
            {std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::min()}
    };
    if (!unsat.empty()) {
        coverage.unsatisfied_rectangle(unsat, box);
        auto segment_select = [&](segment &s) {
            if (coverage.insertion_satisfies(pSol, s, unsat))
                candidates.push_back(s);

        };
        segment dummy{path_engine::NULL_NODE, path_engine::NULL_NODE};
        for_each_segment(box, dummy, segment_select);
    }
    coverage.insert(pSol, candidates.front());
}

cost_t neighborhood_r::gain(const cppied_solution &pSol,
                            neighborhood::sVecIt source) {
    return geometry.removal_gain(pSol, source);
}

cost_t neighborhood_r::replacement_cost(const cppied_solution &pSol,
                                        neighborhood::sVecIt source,
                                        segment target) {
    cost_t c{0,0};
    if (source == std::prev(pSol.path.cend()))
        c = geometry.insert_tail_gain(*std::prev(source), target);
    else
        c = geometry.insert_between_gain(*std::prev(source), target, *std::next(source));
    return c;
}

std::pair<segment, double> neighborhood_r::dag_heuristic(const cppied_solution &pSol, int u, int v,
                                                         std::vector<double> &coverage_duals, int r, double r_dual,
                                                         double gamma_dual, const std::function<double(cost_t)> &f) {
    std::pair<segment, double> candidate = null_dag_heuristic(pSol, u, v, coverage_duals, r, r_dual, gamma_dual, f);
    std::pair<segment, double> other_candidate = reversed_dag_heuristic(pSol, u, v, coverage_duals, r, r_dual, gamma_dual, f);
    if (candidate.second <= other_candidate.second)
        return candidate;
    return other_candidate;
}

std::pair<segment, double> neighborhood_r::null_dag_heuristic(const cppied_solution& pSol,
                                   int first, int last,
                                      std::vector<double> &coverage_duals,
                                      int r, double r_dual,
                                      double gamma_dual,
                                      const std::function<double(cost_t)>& f) {
    std::vector<double> D(last - first + 3,
                          std::numeric_limits<double>::infinity());
    cost_t c_gain{0,0};
    if (r < pSol.path.size()-1)
        c_gain = geometry.dist(pSol.path[r-1], pSol.path[r+1]);
    D[0] = 0.0;
    std::vector<int> P(last - first + 3,0);
    for (int v = first; v<= last; v++){
        segment dummy = {v, path_engine::NULL_NODE};
        cost_t c = geometry.dist(pSol.path[r-1], dummy);
        double rc = f(c) - coverage_duals[v] - gamma_dual * c.length;
        D[v - first + 1] = rc;
    }
    for (int v = first; v<last; v++){
        double cost = D[v - first + 1] + f(cost_t{1,0}) - coverage_duals[v+1] - (gamma_dual * 1.0);
        if (D[v - first + 2] > cost) {
            D[v - first + 2] = cost;
            P[v - first + 2] = v - first + 1;
        }
    }
    double ac;
    for (int v = first; v<last; v++){
        segment dummy = {v, path_engine::NULL_NODE};
        if (r < pSol.path.size() - 1){
            cost_t c = geometry.dist(dummy, pSol.path[r+1]);
            ac = f(c) - f(c_gain) - (gamma_dual*(c.length - c_gain.length));
        }
        else
            ac = 0.0;
        if (D.back() > D[v - first + 1] + ac) {
            P.back() = v - first + 1;
            D.back() = D[v - first + 1] + ac;
        }
    }
    if (D.back() - r_dual >=0)
        return {segment{path_engine::NULL_NODE, path_engine::NULL_NODE}, D.back() - r_dual};

    //Backtrack
    int v = P.back();
    int u = v;

    while (P[u] != 0)
        u = P[u];

    u += first-1;
    v += first - 1;
    std::pair<segment, double> candidate = {
            {v, path_engine::NULL_NODE},
            D.back() - r_dual};
    if (u != v)
        candidate.first = {u,v};
    return candidate;
}

std::pair<segment, double> neighborhood_r::reversed_dag_heuristic(const cppied_solution& pSol,
                                                              int first, int last,
                                                              std::vector<double> &coverage_duals,
                                                              int r, double r_dual,
                                                              double gamma_dual,
                                                              const std::function<double(cost_t)>& f) {
    std::vector<double> D(last - first + 3,
                          std::numeric_limits<double>::infinity());
    cost_t c_gain{0,0};

    if (r < pSol.path.size()-1)
        c_gain = geometry.dist(pSol.path[r-1], pSol.path[r+1]);
    D[0] = 0.0;
    std::vector<int> P(last - first + 3,0);

    for (int v = last; v>= first; v--){
        segment dummy = {v, path_engine::REVERSED_NULL_NODE};
        cost_t c = geometry.dist(pSol.path[r-1], dummy);
        double rc = f(c) - coverage_duals[v] - gamma_dual * c.length;
        const int idx = last - v + 1;
        D[idx] = rc;
    }

    for (int v = last; v>first; v--){
        const int idx = last - v + 1;
        double cost = D[idx] + f(cost_t{1,0}) - coverage_duals[v+1] - (gamma_dual * 1.0);
        if (D[idx + 1] > cost) {
            D[idx + 1] = cost;
            P[idx + 1] = idx;
        }
    }

    double ac;
    for (int v = last; v>first; v--){
        segment dummy = {v, path_engine::REVERSED_NULL_NODE};
        if (r < pSol.path.size() - 1){
            cost_t c = geometry.dist(dummy, pSol.path[r+1]);
            ac = f(c) - f(c_gain) - (gamma_dual*(c.length - c_gain.length));
        }
        else
            ac = 0.0;
        const int idx = last - v + 1;
        if (D.back() > D[idx] + ac) {
            P.back() = idx;
            D.back() = D[idx] + ac;
        }
    }

    if (D.back() - r_dual >=0)
        return {segment{path_engine::NULL_NODE, path_engine::NULL_NODE}, D.back() - r_dual};

    //Backtrack
    int v = P.back();
    int u = v;

    while (P[u] != 0)
        u = P[u];

    u = last - u + 1;
    v = last - v + 1;
    std::pair<segment, double> candidate = {
            {v, path_engine::REVERSED_NULL_NODE},
            D.back() - r_dual};
    if (u != v)
        candidate.first = {u,v};
    return candidate;
}

void neighborhood_r::positions_coverage_rcs(const cppied_solution &,
                                            std::vector<double> &pRC,
                                            std::vector<double> &coverage_duals) {
    pRC.resize(problem.vertex.size(), 0.0);
    for (int v=0; v< pRC.size(); v++){
        for (SMdIt c(problem.s_pod, v); c; ++c)
            pRC[v] += c.value() * coverage_duals[c.index()];
    }
}

bool neighborhood_r::add_stage_columns(cppied_solution &pSol,
                                             std::vector<GRBConstr>& constraints,
                                             std::vector<replacement_set>& R,
                                             GRBModel& model,
                                             int stage) {
    std::vector<double> coverage_duals(geometry.n_rows*geometry.n_cols, 0.0);
    std::vector<double> r_duals(R.size(), 0.0);
    extract_duals(pSol, constraints, coverage_duals, r_duals);

    std::function<double(cost_t)> cost_func;
    double gamma = 0.0;
    if (stage == 1){
        cost_func = [](cost_t c){
            return static_cast<double>(c.length);
        };
    } else{
        cost_func = [](cost_t c){
            return static_cast<double>(c.turns);
        };
        gamma = constraints.back().get(GRB_DoubleAttr_Pi);
    }

    std::vector<double> positions_rcs;
    positions_coverage_rcs(pSol, positions_rcs, coverage_duals);

    bool negative_rcs = false;

    int i=0;
    for (; i< R.size(); i++){
        std::pair<segment, double> best = {{path_engine::NULL_NODE, path_engine::NULL_NODE}, 0.0};
        for (int v = 0; v<geometry.horizontal_bound; v+= geometry.n_cols){
            std::pair<segment, double> candidate = dag_heuristic(pSol,
                                                                 v,
                                                                 v+ geometry.n_cols -1,
                                                                 positions_rcs,
                                                                 R[i].r,
                                                                 r_duals[i],
                                                                 gamma,
                                                                 cost_func);
            if (candidate.second < best.second &&
                std::find(R[i].candidates.begin(),
                          R[i].candidates.end(),
                          candidate.first) == R[i].candidates.end())
                best = candidate;
        }
        for (int v= geometry.horizontal_bound; v<problem.vertex.size(); v+=geometry.n_rows){
            std::pair<segment, double> candidate = dag_heuristic(pSol,
                                                                 v,
                                                                 v+ geometry.n_rows -1,
                                                                 positions_rcs,
                                                                 R[i].r,
                                                                 r_duals[i],
                                                                 gamma,
                                                                 cost_func);
            if (candidate.second < best.second&&
                std::find(R[i].candidates.begin(),
                          R[i].candidates.end(),
                          candidate.first) == R[i].candidates.end())
                best = candidate;
        }
        if (best.second < 0.0){
            cost_t obj_values = replacement_cost(pSol,
                                                 std::next(pSol.path.cbegin(), R[i].r),
                                                 best.first);

            R[i].candidates.push_back(best.first);
            R[i].variables.push_back(model.addVar(0.0, 1.0,
                                                  obj_values.length,
                                                  GRB_CONTINUOUS));
            Z1 += obj_values.length * R[i].variables.back();
            Z2 += obj_values.turns * R[i].variables.back();
            add_stage_column(pSol, constraints, R[i], i, model);
            negative_rcs = true;
        }
    }
    return !negative_rcs;
}

void neighborhood_r::add_stage_column(cppied_solution &pSol,
                                            std::vector<GRBConstr> &constraints,
                                            replacement_set &R,
                                            int r_ctr,
                                            GRBModel &model) {
    segment new_seg = R.candidates.back();
    auto chg_coeff = [&](
            cppied_solution &, int v){
        for (SMdIt c(problem.s_pod, v); c; ++c)
            model.chgCoeff(constraints[c.index()],
                           R.variables.back(),
                           c.value());
    };
    coverage.apply(pSol, new_seg, chg_coeff);
    cost_t rep_c = replacement_cost(pSol,
                                    std::next(pSol.path.cbegin(), R.r),
                                    new_seg);
    int ctr_idx = geometry.n_rows*geometry.n_cols + r_ctr;
    model.chgCoeff(constraints[ctr_idx], R.variables.back(), 1.0);
    double length_val = rep_c.length;
    model.chgCoeff(constraints.back(), R.variables.back(), length_val);
}

void neighborhood_r::set_warm_start(cppied_solution &, std::vector<replacement_set> &R) {
    for (auto& r : R){
        r.variables[0].set(GRB_DoubleAttr_Start, 1.0);
        for (int i=1; i<r.variables.size();  ++i)
            r.variables[i].set(GRB_DoubleAttr_Start, 0.0);
    }
}
