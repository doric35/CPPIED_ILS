#include "../../include/neighborhoods/neighborhood_trim.hpp"

bool neighborhood_trim::local_search(cppied_solution& pSol) {
    std::vector<segment> trials(pSol.path.size());
    std::vector<cost_t> gains(pSol.path.size());

    int i=1;
    auto path_it = std::next(pSol.path.begin());
    auto trial_it = std::next(trials.begin());

    for(; i< pSol.path.size(); ++i, ++path_it, ++trial_it){
        trials[i] = trim(pSol, path_it);
        gains[i] = gain(pSol, path_it, trial_it);
    }

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

    auto path_split_func = [](int i){
        return i % 2 == 0;
    };
    auto geometry_split_func = [&](int i){
        return geometry.is_horizontal(pSol.path[i]);
    };
    auto splitter = [&](const std::vector<int>& candidates,
            std::vector<std::vector<int>>& solution){
        std::vector<int> odds;
        std::vector<int> evens;
        odds.reserve(candidates.size() / 2 + 1);
        evens.reserve(candidates.size() / 2 + 1);
        std::partition_copy(candidates.begin(), candidates.end(),
                            std::back_inserter(evens), std::back_inserter(odds),
                            path_split_func);

        solution.resize(4);
        for (auto& vec: solution)
            vec.reserve(candidates.size() / 2 + 1);
        std::partition_copy(evens.begin(), evens.end(),
                            std::back_inserter(solution[0]),
                            std::back_inserter(solution[1]),
                            geometry_split_func);
        std::partition_copy(odds.begin(), odds.end(),
                            std::back_inserter(solution[2]),
                            std::back_inserter(solution[3]),
                            geometry_split_func);
        std::erase_if(solution, [](const std::vector<int>& v){return v.empty();});
    };
    auto interval_transform = [&](int i){
        int x1 = is_node(pSol.path[i].target) ?
                std::min(pSol.path[i].target, pSol.path[i].source) : pSol.path[i].source;
        int x2 = std::max(pSol.path[i].source, pSol.path[i].target);
        bool rotated = !geometry.is_horizontal(pSol.path[i]);
        int y = rotated ? geometry.get_column(pSol.path[i]): geometry.get_row(pSol.path[i]);
        interval candidate = {gains[i], x1, x2, y, rotated, i};
        return candidate;
    };
    auto interval_function = [&](const std::vector<int>& candidates,
            std::vector<interval>& pSol){
        pSol.reserve(candidates.size());
        std::transform(candidates.begin(), candidates.end(),
                       std::back_inserter(pSol), interval_transform);
    };
    ris_heuristic selection_heuristic(2*problem.max_range,
                                      splitter,
                                      filter,
                                      interval_function);

    std::vector<int> selection;
    selection_heuristic.solve_selection(gains, selection);

    update(pSol, trials, gains, selection);

    return true;
}

void neighborhood_trim::update(cppied_solution &pSol,
                               const std::vector<segment> &trials,
                               const std::vector<cost_t> &gains,
                               std::vector<int> &selection) {
    auto swap = [&](int i){
        coverage.remove(pSol, pSol.path[i]);
        if (is_node(trials[i].source))
            coverage.insert(pSol, trials[i]);
        pSol.path[i] = trials[i];
    };

    std::for_each(selection.cbegin(), selection.cend(), swap);

    cost_t marginal_gain = {0,0};
    auto reduce_gain = [&] (cost_t g, int i){
        return gains[i] + g;
    };

    marginal_gain = std::reduce(selection.cbegin(), selection.cend(),
                                marginal_gain, reduce_gain);
    std::erase_if(pSol.path,
                  [](const segment& s){return !is_node(s.source);});
    pSol.cost -= marginal_gain;
}

cost_t neighborhood_trim::gain(const cppied_solution &pSol,
                               neighborhood::sVecIt source,
                               neighborhood::sVecIt target) {
    cost_t gain = geometry.removal_gain(pSol, source);
    if (!is_node(target->source))
        return gain;
    if (source != std::prev(pSol.path.end()))
        gain -= geometry.insert_between_gain(*std::prev(source), *target, *std::next(source));
    else
        gain -= geometry.insert_tail_gain(*std::prev(source), *target);
    return gain;
}