#include "../../include/neighborhoods/neighborhood_reduce.hpp"

bool neighborhood_reduce::local_search(cppied_solution& pSol) {
    std::vector<cost_t> gains(pSol.path.size());

    gains[0] = {0,0};
    int i=1;
    auto path_it = std::next(pSol.path.begin());

    for(; i< pSol.path.size(); ++i, ++path_it)
        gains[i] = gain(pSol, path_it);

    auto best = std::max_element(gains.begin(), gains.end());
    if (*best <= cost_t{0,0})
        return false;

    cost_t bound = {0,0};
    auto erase_func = [&] (int i){
        return gains[i] <= bound;
    };

    auto filter = [&](std::vector<int>& candidates){
        std::erase_if(candidates, erase_func);
    };

    auto geometry_split_func = [&](int i){
        return geometry.is_horizontal(pSol.path[i]);
    };
    auto splitter = [&](const std::vector<int>& candidates,
                        std::vector<std::vector<int>>& solution){
        solution.resize(2);
        for (auto& vec: solution)
            vec.reserve(candidates.size());
        std::partition_copy(candidates.begin(), candidates.end(),
                            std::back_inserter(solution[0]),
                            std::back_inserter(solution[1]),
                            geometry_split_func);
        std::erase_if(solution, [](const std::vector<int>& v){return v.empty();});
    };
    auto interval_transform = [&](int i){
        int x1 = path_engine::is_node(pSol.path[i].target) ?
                 std::min(pSol.path[i].target, pSol.path[i].source) : pSol.path[i].source;
        int x2 = std::max(pSol.path[i].source, pSol.path[i].target);
        bool rotated = !geometry.is_horizontal(pSol.path[i]);
        int      y         = rotated ? geometry.get_column(pSol.path[i]): geometry.get_row(pSol.path[i]);
        Interval candidate = {gains[i], x1, x2, y, rotated, i};
        return candidate;
    };
    auto interval_function = [&](
            const std::vector<int>& candidates,
            IntervalSet& pSol){
        pSol.reserve(candidates.size());
        std::transform(candidates.begin(), candidates.end(),
                       std::back_inserter(pSol), interval_transform);
    };
    ris_heuristic selection_heuristic(2*problem.max_range,
                                      static_cast<int>(gains.size()),
                                      splitter,
                                      filter,
                                      interval_function);

    std::vector<int> selection = selection_heuristic.solve_selection();

    update(pSol,selection);
    return true;
}

void neighborhood_reduce::update(cppied_solution &pSol,
                              std::vector<int> &selection) {
    std::vector<segment> new_sol;
    new_sol.reserve(pSol.path.size() - selection.size());
    std::sort(selection.begin(), selection.end());

    int j = 0;
    for (int i=0; i< pSol.path.size(); ++i){
        if (j < selection.size() && i == selection[j]){
            coverage.remove(pSol, pSol.path[i]);
            ++j;
        } else
            new_sol.push_back(pSol.path[i]);
    }
    pSol.path.swap(new_sol);
    pSol.cost = geometry.cost(pSol);
}

cost_t neighborhood_reduce::gain(cppied_solution &pSol,
                              neighborhood::sVecIt target) {
    if (coverage.can_remove(pSol, *target))
        return geometry.cost(*target) + cost_t{1,0};
    return cost_t{0,0};
}