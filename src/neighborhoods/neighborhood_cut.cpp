#include "../../include/neighborhoods/neighborhood_cut.hpp"

bool neighborhood_cut::local_search(cppied_solution& pSol) {
    std::vector<segment> trials(pSol.path.size());
    std::vector<cost_t> gains(pSol.path.size());

    trials[0] = {NULL_NODE, NULL_NODE};
    gains[0] = {0,0};
    int i=1;
    auto path_it = std::next(pSol.path.begin());
    auto trial_it = std::next(trials.begin());

    for(; i< pSol.path.size(); ++i, ++path_it, ++trial_it){
        trials[i] = cut(pSol, path_it);
        gains[i] = gain(pSol, trial_it);
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
        int x1 = is_node(pSol.path[i].target) ?
                 std::min(pSol.path[i].target, pSol.path[i].source) : pSol.path[i].source;
        int x2 = std::max(pSol.path[i].source, pSol.path[i].target);
        bool rotated = !geometry.is_horizontal(pSol.path[i]);
        int y = rotated ? geometry.get_column(pSol.path[i]): geometry.get_row(pSol.path[i]);
        interval candidate = {gains[i], x1, x2, y, rotated, i};
        return candidate;
    };
    auto interval_function = [&](
            const std::vector<int>& candidates,
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

    update(pSol, trials,selection);

    return true;
}

void neighborhood_cut::update(cppied_solution &pSol,
                               std::vector<segment> &trials,
                               std::vector<int> &selection) {
    std::vector<segment> new_sol;
    new_sol.reserve(pSol.path.size() + (2*selection.size()));
    std::sort(selection.begin(), selection.end());

    int j = 0;
    for (int i=0; i< pSol.path.size(); ++i){
        if (j < selection.size() && i == selection[j]){
            assert(is_node(trials[i].source));
            coverage.remove(pSol, trials[i]);
            geometry.correct_segment_direction(trials[i],
                                               geometry.get_direction(pSol.path[i]));
            std::array<segment, 2> S = cut_diff(pSol.path[i], trials[i]);
            for (auto& s:S){
                if (is_node(s.source))
                    new_sol.push_back(s);
            }
            ++j;
        } else
            new_sol.push_back(pSol.path[i]);
    }
    pSol.path.swap(new_sol);
    pSol.cost = geometry.cost(pSol);
}

segment neighborhood_cut::cut(cppied_solution &pSol, neighborhood::sVecIt seg) {
    segment cut_seg = {NULL_NODE, NULL_NODE};
    direction dir = geometry.get_direction(*seg);
    if (is_node(seg->target)){
        int v1 = std::min(seg->source, seg->target);
        int v2 = std::max(seg->source, seg->target);
        std::vector<int> dp_gain(v2 - v1 + 1, 0);
        segment dummy{0,0};

        for (int i=v1; i<= v2; ++i){
            dummy = {i, NULL_NODE};
            dp_gain[i - v1] = static_cast<int>(coverage.can_remove(pSol, dummy));
            dp_gain[i - v1] += (dp_gain[i-v1] - 1)*(geometry.n_rows + geometry.n_cols);
        }

        int res = dp_gain[0], max_ending = dp_gain[0];
        int curr_start = 0, curr_end = 0, best_start = 0, best_end = 0;
        for (int i=1; i< dp_gain.size(); ++i){
            if (dp_gain[i] > max_ending + dp_gain[i])
                curr_start = i, max_ending = dp_gain[i]; // Start new
            else
                max_ending += dp_gain[i];
            curr_end = i;
            if (max_ending > res)
                res = max_ending, best_start = curr_start, best_end = curr_end;
        }

        if (res > 0){
            if (best_start == best_end){
                cut_seg = {best_start + v1, NULL_NODE};
                geometry.correct_segment_direction(cut_seg, dir);
            } else {
                cut_seg = {best_start + v1, best_end + v1};
                geometry.correct_segment_direction(cut_seg, dir);
            }
        }
    }
    return cut_seg;
}

std::array<segment, 2> neighborhood_cut::cut_diff(const segment &a,
                                                  segment &b) {
    std::array<segment,2> s{{{NULL_NODE, NULL_NODE},
                            {NULL_NODE, NULL_NODE}}};
    direction dir = geometry.get_direction(a);
    geometry.correct_segment_direction(b, dir);
    //Full overlap
    if (a == b)
        return s;

    if (!is_node(b.target))
        b.target = b.source;

    //Bisection
    if (a.source != b.source && a.target != b.target){
        int step = (a.source - b.source) / std::abs(a.source - b.source);

        s[0] = {a.source, b.source + step};
        if (s[0].source == s[0].target) s[0].target = NULL_NODE;
        geometry.correct_segment_direction(s[0], dir);

        s[1] = {b.target - step, a.target};
        if (s[1].source == s[1].target) s[1].target = NULL_NODE;
        geometry.correct_segment_direction(s[1], dir);
    }//Front
    else if (a.target != b.target){
        int step = (a.target - b.target) / std::abs(a.target - b.target);
        s[0] = {b.target + step, a.target};
        if (s[0].source == s[0].target) s[0].target = NULL_NODE;
        geometry.correct_segment_direction(s[0], dir);
    }//Tail
    else{
        int step = (a.source - b.source) / std::abs(a.source - b.source);
        s[0] = {a.source, b.source + step};
        if (s[0].source == s[0].target) s[0].target = NULL_NODE;
        geometry.correct_segment_direction(s[0], dir);
    }
    return s;
}

cost_t neighborhood_cut::gain(const cppied_solution &pSol,
                               neighborhood::sVecIt target) {
    cost_t gain = {0,0};
    if (is_node(target->source))
        gain += geometry.cost(*target) + cost_t{1,0};
    return gain;
}
