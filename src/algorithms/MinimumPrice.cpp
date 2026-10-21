//
// Created by Dominik Richard on 2026-10-21.
//

#include "../../include/algorithms/MinimumPrice.hpp"

Reallocation MinimumPrice::solve() {
    solveHorizontal();
    solveVertical();
    return incumbent;
}

void MinimumPrice::solveHorizontal() {
    for (int u = 0; u < geometry.horizontal_bound; u+= geometry.n_cols) {
        Reallocation move = solveEdge(u, u + geometry.n_cols -1);
        updateIncumbent(move);
    }
}

void MinimumPrice::solveVertical() {
    for (int u = geometry.horizontal_bound; u < geometry.n_vertex; u+= geometry.n_rows) {
        Reallocation move = solveEdge(u, u + geometry.n_rows - 1);
        updateIncumbent(move);
    }
}

Reallocation MinimumPrice::solveEdge(int u, int v) {
    Reallocation argmin;
    ReallocationDynamicProgram forwardDp = getForwardDp(u, v);
    ReallocationDynamicProgram backwardDp = getBackwardDp(u, v);
    int nullNode = path_engine::NULL_NODE;
    if (forwardDp.values.back() <= backwardDp.values.back()){
        argmin = getReallocation(forwardDp);
        argmin.S.source += u, argmin.S.target += u;
    } else{
        argmin = getReallocation(backwardDp);
        argmin.S.source = v - argmin.S.source, argmin.S.target = v - argmin.S.target;
        nullNode = path_engine::REVERSED_NULL_NODE;
    }
    argmin.S.target = argmin.S.source == argmin.S.target ? nullNode : argmin.S.target;
    return argmin;
}

Reallocation MinimumPrice::getReallocation(const ReallocationDynamicProgram &dp) {
    int last_vertex = dp.predecessors.back();

    const auto match = std::ranges::find_last_if(
            dp.predecessors.begin(),
            dp.predecessors.begin() + last_vertex + 1,
            [](int p) { return p == -1; }
    );

    int first_vertex = static_cast<int>(std::distance(dp.predecessors.begin(), match.begin()));
    return {segment{first_vertex, last_vertex}, dp.values.back(), index};
}

ReallocationDynamicProgram MinimumPrice::getForwardDp(int u, int v) {
    ReallocationDynamicProgram dp{
            std::vector<double>(v - u + 2, MAX_INT),
            std::vector<int>(v - u + 2, -1)
    };
    dp.values[0] = sourceLinkPrice(segment{u, path_engine::NULL_NODE});
    for (int l = u + 1; l<= v; l++){
        double first_node_value = sourceLinkPrice(segment{l, path_engine::NULL_NODE});
        double internal_price = internLinkPrice(segment{l, path_engine::NULL_NODE});
        if (dp.values[l-u-1] + internal_price < first_node_value)
            dp.predecessors[l-u] = l-u-1;
        dp.values[l-u] = std::min(dp.values[l-u-1] + internal_price, first_node_value);
    }
    for (int l= u; l<=v; l++){
        double last_node_value = targetLinkPrice(segment{l, path_engine::NULL_NODE});
        if (dp.values[l-u] + last_node_value < dp.values.back())
            dp.predecessors.back() = l-u;
        dp.values.back() = std::min(dp.values.back(), dp.values[l-u] + last_node_value);
    }
    return dp;
}

ReallocationDynamicProgram MinimumPrice::getBackwardDp(int u, int v) {
    ReallocationDynamicProgram dp{
            std::vector<double>(v - u + 2, MAX_INT),
            std::vector<int>(v - u + 2, -1)
    };
    dp.values[0] = sourceLinkPrice(segment{v, path_engine::REVERSED_NULL_NODE});
    for (int l = v - 1; l >= u; l--){
        double first_node_value = sourceLinkPrice(segment{l, path_engine::REVERSED_NULL_NODE});
        double internal_price = internLinkPrice(segment{l, path_engine::REVERSED_NULL_NODE});
        if (dp.values[v-l-1] + internal_price < first_node_value)
            dp.predecessors[v-l] = v-l-1;
        dp.values[v-l] = std::min(dp.values[v-l-1] + internal_price, first_node_value);
    }
    for (int l = v; l >= u; l--){
        double last_node_value = targetLinkPrice(segment{l, path_engine::REVERSED_NULL_NODE});
        if (dp.values[v-l] + last_node_value < dp.values.back())
            dp.predecessors.back() = v-l;
        dp.values.back() = std::min(dp.values.back(), dp.values[v-l] + last_node_value);
    }
    return dp;
}

void MinimumPrice::updateIncumbent(Reallocation candidate) {
    if (candidate.value < incumbent.value) incumbent = candidate;
}

cost_t MinimumPrice::getGain(Reallocation &move) {
    cost_t cost{0,0};
    if (index < solution.path.size() - 1){
        cost = geometry.insert_between_gain(solution.path[index -1], move.S, solution.path[index + 1]);
    } else{
        cost = geometry.insert_tail_gain(solution.path[index - 1], move.S);
    }
    return cost;
}