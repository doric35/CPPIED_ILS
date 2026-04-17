#include "../../include/models/ilp.hpp"

void ilp::d_solve(cppied_solution &pSolution) {

    GRBEnv modeling_env = GRBEnv(true);
    auto log_item = ctx.config.find("WORKING_DIRECTORY");
    if (log_item != ctx.config.end()){
        std::filesystem::path log_path = log_item->second;
        log_path = log_path / "modeling.log";
        modeling_env.set("LogFile", log_path);
    } else
        modeling_env.set("LogFile", "modeling.log");
    modeling_env.set(GRB_IntParam_OutputFlag, 0);
    modeling_env.set(GRB_IntParam_LogToConsole, 0);
    modeling_env.set(GRB_IntParam_ThreadLimit, 1);
    modeling_env.start();
    GRBModel model = GRBModel(modeling_env);

    set_flow_sets(pSolution);
    set_variables(pSolution, model);
    set_constraints(pSolution, model);
    set_objectives(pSolution, model);

    int t = remaining_time();
    model.set(GRB_DoubleParam_TimeLimit, t + 1.0);
    if (no_rel)
        model.set(GRB_DoubleParam_NoRelHeurTime, double(t) / 2.0);

    subtour_elimination cb(*this);
    model.set(GRB_IntParam_LazyConstraints, 1);
    model.setCallback(&cb);

    model.update();
    model.optimize();
    //Retrieve solution
    int status = model.get(GRB_IntAttr_Status);
    if (status == GRB_OPTIMAL || status == GRB_SUBOPTIMAL ||
            (status == GRB_TIME_LIMIT && model.get(GRB_IntAttr_SolCount) > 0)) {
        // A feasible solution exists, safe to retrieve
        auto flow_setter = [&](){
            for (int u=0; u<problem.vertex.size(); ++u){
                for (int v =0; v<variables[u].outgoing_variables.size(); ++v){
                    variables[u].variables_flow[v] =
                            variables[u].outgoing_variables[v].get(GRB_DoubleAttr_X);
                }
                targets_minus_flow[u] = targets_minus[u].get(GRB_DoubleAttr_X);
                targets_plus_flow[u] = targets_plus[u].get(GRB_DoubleAttr_X);
            }
        };
        std::cout << "Solved" << std::endl;
        retrieve_solution(pSolution, flow_setter);
    } else if (status == GRB_INF_OR_UNBD || status == GRB_INFEASIBLE) {
        // No feasible solution found
        std::cerr << "[Error] Model infeasible or unbounded\n";
        throw std::runtime_error("Infeasibility or unboundedness found in ilp model.\n");
    } else {
        std::cerr << "[Warning] Optimization interrupted in ilp model, did not find a feasible solution. Status ["
                  << status << "]" << "\n";
    }
}

void ilp::set_variables(cppied_solution &pSolution, GRBModel &model) {
    source = {{},{}, {}, {}};
    source.outgoing_arcs_V1.push_back(problem.initial_position);
    source.outgoing_variables.push_back(
            model.addVar(0.0, 1.0, 0.0, GRB_BINARY)
            );
    variables.clear();
    variables.reserve(problem.vertex.size()*6);
    targets_minus.clear();
    targets_minus.reserve(problem.vertex.size());
    targets_plus.clear();
    targets_plus.reserve(problem.vertex.size());
    for (int u = 0; u< problem.vertex.size(); ++u){
        variables.push_back({{}, {}, {}, {}});
        int arc_id =0;
        for (SMiIt v(problem.adj, u); v; ++v){
            variables.back().outgoing_arcs_V1.push_back(v.index());
            variables.back().outgoing_variables.push_back(
                    model.addVar(0.0, GRB_INFINITY, 0.0, GRB_INTEGER)
                    );
            variables.back().vertex_to_arc.insert({v.index(), arc_id++});
            variables.back().variables_flow.push_back(0);
        }
        targets_minus.push_back(
                model.addVar(0.0, 1.0, 0.0, GRB_BINARY)
                );
        targets_plus.push_back(
                model.addVar(0.0, 1.0, 0.0, GRB_BINARY)
        );
    }
    targets_minus_flow.resize(targets_minus.size(), 0.0);
    targets_plus_flow.resize(targets_plus.size(), 0.0);
}

void ilp::set_constraints(cppied_solution &pSolution,
                          GRBModel &model) {
    {
        std::vector<GRBLinExpr> coverage_constraints(geometry.n_rows * geometry.n_cols, 0);
        //Coverage constraints
        for (SMdIt c(problem.s_pod, source.outgoing_arcs_V1[0]); c; ++c)
            coverage_constraints[c.index()] += source.outgoing_variables[0] * c.value();

        for (int u = 0; u < variables.size(); u++) {
            for (int v = 0; v < variables[u].outgoing_arcs_V1.size(); ++v) {
                for (SMdIt c(problem.s_pod, variables[u].outgoing_arcs_V1[v]); c; ++c)
                    coverage_constraints[c.index()] += variables[u].outgoing_variables[v] * c.value();
            }
        }

        for (int c =0; c<coverage_constraints.size(); c++)
            model.addConstr(coverage_constraints[c] >= problem.req(c));
    }
    {
        //Flow conservation constraints
        starting_minus = true;
        for (int u : minus_sets[problem.initial_position].V)
            if (geometry.is_horizontal({u, path_engine::NULL_NODE}) ==
                geometry.is_horizontal({problem.initial_position, path_engine::NULL_NODE}) &&
                geometry.dist({problem.initial_position, path_engine::NULL_NODE}, {u, path_engine::NULL_NODE}) == cost_t{1,0}){
                starting_minus = false;
            }
        for (int v = 0; v< problem.vertex.size(); ++v){
            GRBLinExpr flow_one = 0;
            for (int u : minus_sets[v].V)
                flow_one += variables[u].outgoing_variables[variables[u].vertex_to_arc[v]];

            for (int w : plus_sets[v].V)
                flow_one -= variables[v].outgoing_variables[variables[v].vertex_to_arc[w]];


            if (v == problem.initial_position && starting_minus)
                flow_one += source.outgoing_variables[0];

            model.addConstr(flow_one - targets_plus[v] == 0);
            GRBLinExpr flow_two = 0;
            for (int u : plus_sets[v].V)
                flow_two += variables[u].outgoing_variables[variables[u].vertex_to_arc[v]];

            for (int w : minus_sets[v].V)
                flow_two -= variables[v].outgoing_variables[variables[v].vertex_to_arc[w]];

            if (v == problem.initial_position && !starting_minus)
                flow_two += source.outgoing_variables[0];

            model.addConstr(flow_two - targets_minus[v] == 0);
        }
    }
    {
        //Source flow
        GRBLinExpr src_flow = 0 + source.outgoing_variables[0];
        model.addConstr(src_flow == 1);

        //Target flow
        GRBLinExpr trg_flow = 0;
        for (int v =0; v<problem.vertex.size(); v++)
            trg_flow += targets_minus[v];

        for (int v =0; v<problem.vertex.size(); v++)
            trg_flow += targets_plus[v];

        model.addConstr(trg_flow == 1);
    }
}

void ilp::set_flow_sets(cppied_solution &pSolution) {
    minus_sets.assign(problem.vertex.size(), {});
    plus_sets.assign(problem.vertex.size(), {});

    for (int u=0; u< problem.vertex.size(); ++u){
        for (SMiIt v(problem.adj, u); v; ++v){
            if (minus_sets[u].V.empty() || problem.adj.coeff(v.index(),minus_sets[u].V[0]) == 1)
                minus_sets[u].V.push_back(v.index());
            else
                plus_sets[u].V.push_back(v.index());
        }
    }
}

void ilp::set_objectives(cppied_solution &pSolution,
                         GRBModel &model) {
    Z1 = 0, Z2 = 0;
    for (int u =0; u< problem.vertex.size(); u++){
        for (int v = 0; v< variables[u].outgoing_variables.size(); v++){
            Z1 += variables[u].outgoing_variables[v];
            segment dummy_u = {u, path_engine::NULL_NODE};
            segment dummy_v = {variables[u].outgoing_arcs_V1[v], path_engine::NULL_NODE};
            Z2 += int(
                    geometry.is_horizontal(dummy_u) != geometry.is_horizontal(dummy_v)
                    ) * variables[u].outgoing_variables[v];
        }
    }
    model.setObjective(GRBLinExpr(0.0));
    model.setObjectiveN(Z1, 0, 2);
    model.setObjectiveN(Z2, 1, 1);
}

void ilp::retrieve_solution(cppied_solution &pSolution,
                            const std::function<void()>& set_flow_variables) {
    set_flow_variables();
    std::list<int> path;

    find_source_to_target(pSolution, path);
    auto current = std::prev(path.end());
    bool curr_minus;
    while (current != path.begin()){
        std::list<int> subpath;
        if (std::find(minus_sets[*current].V.begin(), minus_sets[*current].V.end(),*std::prev(current))
            != minus_sets[*current].V.end()){
            curr_minus = true;
        } else
            curr_minus = false;
        find_cycle(pSolution, subpath, *current, curr_minus);
        if (!subpath.empty())
            path.splice(std::next(current), subpath);
        --current;
    }
    std::list<int> subpath;
    find_cycle(pSolution, subpath, *current, starting_minus);
    path.splice(std::next(current), subpath);
    set_solution(pSolution, path);
}

void ilp::set_solution(cppied_solution &pSolution, std::list<int> &path) {
    pSolution.path.clear();
    pSolution.path.emplace_back(problem.initial_position, path_engine::NULL_NODE);
    path.pop_front();
    for (int v : path){
        if (geometry.is_horizontal({v, path_engine::NULL_NODE}) ==
            geometry.is_horizontal(pSolution.path.back()))
            pSolution.path.back().target = v;
        else{
            segment s = {v, path_engine::NULL_NODE};
            if (geometry.dist(pSolution.path.back(), s) > cost_t{1,1})
                s = geometry.flip_segment(s);
            pSolution.path.push_back(s);
        }
    }
    coverage.reset(pSolution);
    pSolution.cost = geometry.cost(pSolution);
}

void ilp::find_source_to_target(cppied_solution &pSolution,
                                std::list<int>& path) {
    int curr_v = problem.initial_position;
    bool curr_minus = starting_minus;
    path.push_back(curr_v);

    //Random walk to find target
    while (true){
        if ((curr_minus && targets_plus_flow[curr_v] > 0.5) ||
            (!curr_minus && targets_minus_flow[curr_v] > 0.5))
            break;

        int next_v = -1;

        if (curr_minus){
            for (int v : plus_sets[curr_v].V){
                if (variables[curr_v].variables_flow[variables[curr_v].vertex_to_arc[v]] > 0.5){
                    variables[curr_v].variables_flow[variables[curr_v].vertex_to_arc[v]] -= 1;
                    next_v = v;
                    break;
                }
            }
        } else {
            for (int v : minus_sets[curr_v].V){
                if (variables[curr_v].variables_flow[variables[curr_v].vertex_to_arc[v]] > 0.5){
                    variables[curr_v].variables_flow[variables[curr_v].vertex_to_arc[v]] -= 1;
                    next_v = v;
                    break;
                }
            }
        }
        if (next_v == -1) {
            std::cerr << "[ILP] Dead end at vertex " << curr_v
                      << " (curr_minus=" << curr_minus
                      << ", targets_plus=" << targets_plus_flow[curr_v]
                      << ", targets_minus=" << targets_minus_flow[curr_v] << ")\n";
            std::cerr << "  plus_sets: ";
            for (int v : plus_sets[curr_v].V)
                std::cerr << v << "(f=" << variables[curr_v].variables_flow[variables[curr_v].vertex_to_arc[v]] << ") ";
            std::cerr << "\n  minus_sets: ";
            for (int v : minus_sets[curr_v].V)
                std::cerr << v << "(f=" << variables[curr_v].variables_flow[variables[curr_v].vertex_to_arc[v]] << ") ";
            std::cerr << "\n";
            throw std::runtime_error("Did not find an outgoing flow from a node in ilp model.\n");
        }

        if (std::find(minus_sets[next_v].V.begin(),
                      minus_sets[next_v].V.end(),
                      curr_v) == minus_sets[next_v].V.end())
            curr_minus = false;
        else
            curr_minus = true;

        assert(curr_v != next_v);
        curr_v = next_v;
        path.push_back(curr_v);
    }
}

void ilp::find_cycle(cppied_solution &pSolution,
                     std::list<int> &path,
                     int current_node,
                     bool curr_minus) {
    while (true){
        int next_node = -1;

        if (curr_minus){
            for (int v : plus_sets[current_node].V){
                if (variables[current_node].variables_flow[variables[current_node].vertex_to_arc[v]] > 0.5){
                    variables[current_node].variables_flow[variables[current_node].vertex_to_arc[v]] -= 1;
                    next_node = v;
                    break;
                }
            }
        } else {
            for (int v : minus_sets[current_node].V){
                if (variables[current_node].variables_flow[variables[current_node].vertex_to_arc[v]] > 0.5){
                    variables[current_node].variables_flow[variables[current_node].vertex_to_arc[v]] -= 1;
                    next_node = v;
                    break;
                }
            }
        }

        if (next_node == -1)
            return;  // no outgoing arc in the expected direction — end of this cycle

        curr_minus = std::find(minus_sets[next_node].V.begin(),
                               minus_sets[next_node].V.end(), current_node)
                     != minus_sets[next_node].V.end();
        current_node = next_node;
        path.push_back(current_node);
    }
}