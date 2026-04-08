#include "../../include/neighborhoods/neighborhood_n21.hpp"

bool neighborhood_n21::local_search(cppied_solution& pSol) {
    std::vector<n21::trial> trials(pSol.path.size());

    int i=1;
    for(; i< pSol.path.size()-1; ++i){
        auto path_it = std::next(pSol.path.begin(),i);
        trials[i] = replace(pSol, path_it);
    }

    auto cmp = [&](const n21::trial& a, const n21::trial& b){
        return a.gain < b.gain;
    };
    auto best = std::max_element(
            trials.begin(),
            trials.end(),
            cmp);
    if (best->gain <= cost_t{0,0})
        return false;

    cost_t bound = {0,0};
    auto erase_func = [&] (int i){
        return trials[i].gain <= bound;
    };

    auto filter = [&](std::vector<int>& candidates){
        std::erase_if(candidates, erase_func);
    };

    auto geometry_split_func = [&](int i){
        return geometry.is_horizontal(pSol.path[i]);
    };
    auto splitter = [&](
            const std::vector<int>& candidates,
            std::vector<std::vector<int>>& solution){
        std::vector<std::vector<int>> no_goods_filtered_sol;
        std::vector<n21::packing> packings;
        no_goods_filtered_sol.emplace_back();
        packings.push_back({{}, Eigen::VectorXi::Zero(pSol.coverage.size())});

        auto shuffled_candidates = candidates;
        std::shuffle(shuffled_candidates.begin(),
                     shuffled_candidates.end(), rng);

        for (int i : shuffled_candidates){
            int k = trials[i].position, l = trials[i].other;
            int p=0;
            for (; p< no_goods_filtered_sol.size(); p++){
                if (
                        packings[p].P.contains(i-1) || packings[p].P.contains(i) || packings[p].P.contains(i+1) ||
                        (k >= 0 && (packings[p].P.contains(k-1) || packings[p].P.contains(k) || packings[p].P.contains(k+1))) ||
                        (l>=0 && (packings[p].P.contains(l-1) || packings[p].P.contains(l) || packings[p].P.contains(l+1))) ||
                        (l>=0 && coverage.intersect_mask(pSol, pSol.path[l], packings[p].Cover)) ||
                        coverage.intersect_mask(pSol, pSol.path[i], packings[p].Cover)
                        )
                    continue;

                packings[p].P.insert(i-1); packings[p].P.insert(i); packings[p].P.insert(i+1);
                if (k >= 0) {
                    packings[p].P.insert(k - 1);
                    packings[p].P.insert(k);
                    packings[p].P.insert(k + 1);
                }
                if (l >= 0) {
                    packings[p].P.insert(l - 1);
                    packings[p].P.insert(l);
                    packings[p].P.insert(l + 1);
                    coverage.mask(pSol, pSol.path[l], packings[p].Cover);
                }
                coverage.mask(pSol, pSol.path[i], packings[p].Cover);
                no_goods_filtered_sol[p].push_back(i);
                break;
            }
            if (p==no_goods_filtered_sol.size()){
                no_goods_filtered_sol.emplace_back();
                packings.push_back({{}, Eigen::VectorXi::Zero(pSol.coverage.size())});
                packings[p].P.insert(i-1); packings[p].P.insert(i); packings[p].P.insert(i+1);
                if (k>= 0) {
                    packings[p].P.insert(k - 1);
                    packings[p].P.insert(k);
                    packings[p].P.insert(k + 1);
                }
                if (l>=0) {
                    packings[p].P.insert(l - 1);
                    packings[p].P.insert(l);
                    packings[p].P.insert(l + 1);
                    coverage.mask(pSol, pSol.path[l], packings[p].Cover);
                }
                coverage.mask(pSol, pSol.path[i], packings[p].Cover);
                no_goods_filtered_sol[p].push_back(i);
            }
        }
        solution.resize(no_goods_filtered_sol.size()*2, {});
        for (int i=0; i< solution.size(); i+=2){
            std::partition_copy(no_goods_filtered_sol[i/2].begin(),
                                no_goods_filtered_sol[i/2].end(),
                                std::back_inserter(solution[i]),
                                std::back_inserter(solution[i+1]),
                                geometry_split_func);
        }
        auto empty_vec = [](const std::vector<int>& v){
            return v.empty();
        };
        erase_if(solution, empty_vec);
    };
    auto interval_transform = [&](int i){
        int x1 = is_node(pSol.path[i].target) ?
                 std::min(pSol.path[i].target, pSol.path[i].source) : pSol.path[i].source;
        int x2 = std::max(pSol.path[i].source, pSol.path[i].target);
        bool rotated = !geometry.is_horizontal(pSol.path[i]);
        int y = rotated ? geometry.get_column(pSol.path[i]): geometry.get_row(pSol.path[i]);
        interval candidate = {trials[i].gain, x1, x2, y, rotated, i};
        return candidate;
    };
    auto interval_function = [&](const std::vector<int>& candidates,
                                 std::vector<interval>& interval_sol){
        interval_sol.reserve(candidates.size());
        std::transform(candidates.begin(), candidates.end(),
                       std::back_inserter(interval_sol), interval_transform);
    };
    ris_heuristic selection_heuristic(2*problem.max_range,
                                      splitter,
                                      filter,
                                      interval_function);

    std::vector<int> selection;
    std::vector<cost_t> gains;
    gains.reserve(trials.size());
    auto extract_gains = [&](const n21::trial& t){
        return t.gain;
    };
    std::transform(trials.cbegin(), trials.cend(),
                   std::back_inserter(gains), extract_gains);
    selection_heuristic.solve_selection(gains, selection);

    cost_t marginal_gain = update(pSol, trials, gains, selection);

    return marginal_gain > cost_t{0,0};
}

cost_t neighborhood_n21::update(cppied_solution &pSol,
                              const std::vector<n21::trial> &trials,
                              const std::vector<cost_t> &gains,
                              std::vector<int> &selection) {
    std::vector<segment> new_sol;
    new_sol.reserve(pSol.path.size() + (2*selection.size()));
    std::sort(selection.begin(), selection.end());

    std::vector<std::pair<int, segment>> input_set;
    input_set.reserve(trials.size() * 2);

    for (int i : selection) {
        if (is_node(trials[i].s.source)) {
            assert(trials[i].position > 0);
            input_set.emplace_back(trials[i].position, trials[i].s);
        }
    }

    auto cmp = [](
            const std::pair<int, segment>& a,
            const std::pair<int, segment>& b){
        return a.first < b.first;
    };

    std::sort(input_set.begin(), input_set.end(), cmp);

    std::vector<int> others;
    others.reserve(selection.size());
    auto extract_other = [&](int i){
        return trials[i].other;
    };
    std::transform(selection.begin(), selection.end(),
                   std::back_inserter(others), extract_other);
    std::sort(others.begin(), others.end());
    erase_if(others, [&](int i){return !is_node(i);});

    int i=0, j=0, k=0, l=0;
    for (; i< pSol.path.size(); ++i){
        if (l < input_set.size() && i==input_set[l].first){
            coverage.insert(pSol, input_set[l].second);
            new_sol.push_back(input_set[l].second);
            l++;
        }
        if (k < others.size() && i == others[k]){
            coverage.remove(pSol, pSol.path[i]);
            k++;
            continue;
        } else if (j < selection.size() && i == selection[j]){
            coverage.remove(pSol, pSol.path[i]);
            j++;
            continue;
        }else
            new_sol.push_back(pSol.path[i]);
    }
    cost_t marginal_gain{0,0};
    auto func = [&](const cost_t& gain, int i){
        return gain + gains[i];
    };
    marginal_gain = std::reduce(selection.begin(), selection.end(),
                                marginal_gain, func);
    pSol.path.swap(new_sol);
    pSol.cost -= marginal_gain;
    return marginal_gain;
}

n21::trial neighborhood_n21::replace(cppied_solution &pSol,
                                     neighborhood::sVecIt seg) {
    segment seg_save = *seg;
    int position = static_cast<int>(std::distance(pSol.path.cbegin(), seg));
    cost_t g = geometry.removal_gain(pSol, seg);

    if (coverage.can_remove(pSol, seg_save)){
        n21::trial t = {{NULL_NODE, NULL_NODE},
                        -1, -1, g};
        return t;
    }

    coverage.remove(pSol, *seg);
    pSol.path.erase(seg);
    n21::trial best = explore_replacements(pSol, seg_save, g);

    if (best.other >= position)    best.other++;
    if (best.position >= position) best.position++;
    if (best.other == best.position) best.position++;

    coverage.insert(pSol, seg_save);
    pSol.path.insert(std::next(pSol.path.begin(),position), seg_save);

    return best;
}

n21::trial neighborhood_n21::explore_replacements(cppied_solution &pSol,
                                                  const segment &ref,
                                                  cost_t extraction_gain) {
    std::vector<int> unsat;
    std::vector<int> other_candidates;
    other_candidates.resize(pSol.path.size()-2);
    std::iota(other_candidates.begin(), other_candidates.end(), 1);

    coverage.unsatisfied_cells(pSol, ref, unsat);
    iRectangle box = {
            {std::numeric_limits<int>::max(),
                    std::numeric_limits<int>::max()},
            {std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::min()}
    };
    coverage.unsatisfied_rectangle(unsat, box);
    iRectangle other_box{};
    if (geometry.is_horizontal(ref)){
        other_box = {
                {0, std::max(0, box.ul.y - problem.max_range)},
                {geometry.n_cols,
                 std::min(geometry.n_rows, box.lr.y + problem.max_range)}
        };
    } else {
        other_box = {
                {std::max(0, box.ul.x - problem.max_range),0},
                {std::min(geometry.n_cols, box.ul.x + problem.max_range),
                    geometry.n_rows}
        };
    }

    auto pair_select = [&](int i){
        bool valid = geometry.is_in_rectangle(pSol.path[i], other_box);
        cost_t other_gain = geometry.removal_gain(pSol,
                                                  std::next(pSol.path.cbegin(),i));
        if (valid && other_gain + extraction_gain > cost_t{0,0})
            return false;

        return true;
    };

    std::erase_if(
            other_candidates,
            pair_select);

    n21::trial best = select_best_trial(pSol, other_candidates, unsat);
    best.gain += extraction_gain;

    return best;
}

n21::trial neighborhood_n21::select_best_trial(cppied_solution &pSol,
                                               std::vector<int> &candidates,
                                               std::vector<int>& global_unsat) {
    n21::trial best = {
            {NULL_NODE, NULL_NODE},
            -1,
            -1,
            {std::numeric_limits<int>::min(), std::numeric_limits<int>::min()}
    };
    std::vector<int> union_unsat;
    std::vector<int> unsat;
    union_unsat.reserve(global_unsat.size());
    std::sort(global_unsat.begin(), global_unsat.end());
    for (int i : candidates){
        union_unsat.clear();
        unsat.clear();
        coverage.remove(pSol,pSol.path[i]);
        coverage.unsatisfied_cells(pSol, pSol.path[i], unsat);
        std::sort(unsat.begin(), unsat.end());
        std::set_union(global_unsat.begin(), global_unsat.end(),
                       unsat.begin(), unsat.end(),
                       std::back_inserter(union_unsat));
        iRectangle box{
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::min(),
            std::numeric_limits<int>::min()
        };
        coverage.unsatisfied_rectangle(union_unsat, box);

        n21::trial t = select_boxed_candidate(pSol,
                                              box,
                                              union_unsat,
                                              std::next(pSol.path.cbegin(), i));
        t.gain = geometry.removal_gain(pSol, std::next(pSol.path.cbegin(), i)) - t.gain;
        if (t.gain > best.gain)
            best = t;
        coverage.insert(pSol, pSol.path[i]);
    }
    return best;
}

n21::trial neighborhood_n21::select_boxed_candidate(cppied_solution &pSol,
                                                    const iRectangle &box,
                                                    std::vector<int> &unsat,
                                                    sVecIt ref) {
    n21::trial best{
            {NULL_NODE, NULL_NODE},
            -1,
            -1,
            {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()}
    };
    if (unsat.empty()){
        best.other = static_cast<int>(std::distance(pSol.path.cbegin(), ref));
        best.position = NULL_NODE;
        best.gain = {0,0};
        return best;
    }
    auto trial_select = [&](const segment& s){
        if (coverage.insertion_satisfies(pSol, s, unsat)){
            segment other = geometry.flip_segment(s);
            std::array<segment, 2> candidates = {s,
                                                 other};
            int ref_pos = static_cast<int>(std::distance(pSol.path.cbegin(), ref));
            for (auto& elem : candidates){
                std::vector<std::pair<int,cost_t>> best_local =
                        geometry.get_top_k_insertion(pSol, elem, 3);
                std::pair<int,cost_t> local_best = {ref_pos,
                                                    geometry.insert_between_gain(
                                                            pSol.path[ref_pos-1],
                                                            elem,
                                                            pSol.path[ref_pos+1])};
                if (local_best.second < best.gain)
                    best = {elem, ref_pos,
                            local_best.first, local_best.second};
                for (auto& [pos, gain] : best_local){
                    if (pos == ref_pos || pos == ref_pos + 1){
                        gain = geometry.insert_between_gain(pSol.path[ref_pos - 1],
                                                            elem,
                                                            pSol.path[ref_pos + 1]);
                    }
                    if (gain < best.gain)
                        best = {elem, ref_pos, pos, gain};
                }
            }
        }
    };
    segment dummy = {NULL_NODE, NULL_NODE};
    for_each_segment(box, dummy, trial_select);
    return best;
}