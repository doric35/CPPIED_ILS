#include "../../include/neighborhoods/neighborhood_n12.hpp"

bool neighborhood_n12::local_search(cppied_solution& pSol) {
    std::vector<n12::trial> trials(pSol.path.size());
    std::vector<cost_t> gains(pSol.path.size());

    gains[0] = {std::numeric_limits<int>::min(),
               std::numeric_limits<int>::min()};

    int i=1;
    for(; i< pSol.path.size(); ++i){
        auto path_it = std::next(pSol.path.begin(),i);
        n12::evaluated_trial et = replace(pSol, path_it);
        std::tie(trials[i], gains[i]) = {et.t, et.gain};
    }

    auto best = std::max_element(gains.begin(), gains.end());
    if (*best <= cost_t{0,0} - allowed_degradation)
        return false;

    selection_heuristic_update(pSol, trials, gains);
    return true;
}

void neighborhood_n12::selection_heuristic_update(cppied_solution &pSol,
                                                  std::vector<n12::trial> &trials,
                                                  std::vector<cost_t> &gains) {
    cost_t bound = cost_t{0,0} - allowed_degradation;
    auto erase_func = [&] (int i){
        return gains[i] <= bound;
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
        std::vector<std::set<int>> no_goods;
        no_goods_filtered_sol.emplace_back();
        no_goods.emplace_back();

        auto shuffled_candidates = candidates;
        std::shuffle(shuffled_candidates.begin(),
                     shuffled_candidates.end(), rng);

        for (int i : shuffled_candidates){
            int k=-1, l=-1;
            if (!trials[i].s1.top_k.empty())
                k = trials[i].s1.top_k[0].first;
            if (!trials[i].s2.top_k.empty())
                l = trials[i].s2.top_k[0].first;
            int p=0;
            for (; p< no_goods_filtered_sol.size(); p++){
                if (no_goods[p].contains(i) ||
                    ( k != -1 && no_goods[p].contains(k)) ||
                    (l != -1 && no_goods[p].contains(l)))
                    continue;
                no_goods[p].insert(i-1);no_goods[p].insert(i);no_goods[p].insert(i+1);
                if ( k!= -1)
                    no_goods[p].insert(k-1);no_goods[p].insert(k);no_goods[p].insert(k+1);
                if (l!= -1)
                    no_goods[p].insert(l-1);no_goods[p].insert(l);no_goods[p].insert(l+1);
                no_goods_filtered_sol[p].push_back(i);
                break;
            }
            if (p==no_goods_filtered_sol.size()){
                no_goods_filtered_sol.emplace_back();
                no_goods.emplace_back();
                no_goods[p].insert(i-1);no_goods[p].insert(i);no_goods[p].insert(i+1);
                if (k != -1)
                    no_goods[p].insert(k-1);no_goods[p].insert(k);no_goods[p].insert(k+1);
                if (l != -1)
                    no_goods[p].insert(l-1);no_goods[p].insert(l);no_goods[p].insert(l+1);
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
        int x1 = path_engine::is_node(pSol.path[i].target) ?
                 std::min(pSol.path[i].target, pSol.path[i].source) : pSol.path[i].source;
        int x2 = std::max(pSol.path[i].source, pSol.path[i].target);
        bool rotated = !geometry.is_horizontal(pSol.path[i]);
        int y = rotated ? geometry.get_column(pSol.path[i]): geometry.get_row(pSol.path[i]);
        interval candidate = {gains[i], x1, x2, y, rotated, i};
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
    selection_heuristic.solve_selection(gains, selection);

    update(pSol, trials, gains, selection);
}

void neighborhood_n12::update(cppied_solution &pSol,
                             const std::vector<n12::trial> &trials,
                             const std::vector<cost_t> &gains,
                             std::vector<int> &selection) {
    std::vector<segment> new_sol;
    new_sol.reserve(pSol.path.size() + (2*selection.size()));
    std::sort(selection.begin(), selection.end());

    std::vector<std::pair<int, segment>> input_set;
    input_set.reserve(trials.size() * 2);

    for (int i : selection){
        if (!trials[i].s1.top_k.empty()) {
            int k = trials[i].s1.top_k[0].first;
            segment s = trials[i].s1.s;
            input_set.emplace_back(k, s);
        }
        if (!trials[i].s2.top_k.empty()){
            int k = trials[i].s2.top_k[0].first;
            segment s = trials[i].s2.s;
            input_set.emplace_back(k, s);
        }
    }

    auto cmp = [](
            const std::pair<int, segment>& a,
            const std::pair<int, segment>& b){
        return a.first < b.first;
    };

    std::stable_sort(input_set.begin(), input_set.end(), cmp);

    int j=0, i=0, k=0;
    for (; i< pSol.path.size(); ++i){
        if (j < input_set.size() && input_set[j].first == i){
            new_sol.push_back(input_set[j].second);
            coverage.insert(pSol, input_set[j].second);
            if (j+1 < input_set.size() &&
                input_set[j+1].first == i) {
                coverage.insert(pSol, input_set[j+1].second);
                new_sol.push_back(input_set[j+1].second);
                ++j;
            }
            ++j;
        }
        if (k >= selection.size() || selection[k] != i)
            new_sol.push_back(pSol.path[i]);
        else {
            coverage.remove(pSol, pSol.path[i]);
            k++;
        }
    }
    cost_t marginal_gain{0,0};
    auto func = [&](const cost_t& gain, int i){
        return gain + gains[i];
    };
    marginal_gain = std::accumulate(selection.begin(),
                                selection.end(),
                                marginal_gain, func);
    pSol.path.swap(new_sol);
    pSol.cost -= marginal_gain;
}

n12::evaluated_trial neighborhood_n12::replace(cppied_solution &pSol, neighborhood::sVecIt seg) {
    segment seg_save = *seg;
    cost_t extraction_gain = geometry.removal_gain(pSol, seg);
    int position = static_cast<int>(std::distance(pSol.path.cbegin(), seg));

    n12::evaluated_trial best_trial =
            {n12::trial{n12::candidate{{path_engine::NULL_NODE, path_engine::NULL_NODE}, {}},
                        n12::candidate{{path_engine::NULL_NODE, path_engine::NULL_NODE},{}}},
             cost_t{std::numeric_limits<int>::min(),std::numeric_limits<int>::min()}};
    if (extraction_gain <= cost_t{0,0} - allowed_degradation)
        return best_trial;

    coverage.remove(pSol, *seg);
    pSol.path.erase(seg);
    best_trial = explore_replacements(pSol, seg_save);
    correct_trial(best_trial, position);
    coverage.insert(pSol, seg_save);
    auto reinserted_item = pSol.path.insert(std::next(pSol.path.begin(),position), seg_save);
    best_trial.gain = gain(pSol, reinserted_item, best_trial);
    return best_trial;
}

void neighborhood_n12::correct_trial(n12::evaluated_trial &t,
                                     int pos) {
    if (!t.t.s1.top_k.empty() &&
        t.t.s1.top_k[0].first >= pos) t.t.s1.top_k[0].first++;
    if (!t.t.s2.top_k.empty() &&
        t.t.s2.top_k[0].first >= pos) t.t.s2.top_k[0].first++;
}

n12::evaluated_trial neighborhood_n12::explore_replacements(cppied_solution &pSol,
                                                            const segment &ref) {
    n12::evaluated_trial best_trial =
            {n12::trial{n12::candidate{{path_engine::NULL_NODE, path_engine::NULL_NODE}, {}},
                        n12::candidate{{path_engine::NULL_NODE, path_engine::NULL_NODE},{}}},
             cost_t{std::numeric_limits<int>::max(),
                    std::numeric_limits<int>::max()}};
    std::vector<int> unsat;
    coverage.unsatisfied_cells(pSol, ref, unsat);
    iRectangle box = {
            {std::numeric_limits<int>::max(),
                    std::numeric_limits<int>::max()},
            {std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::min()}
    };
    coverage.unsatisfied_rectangle(unsat, box);
    auto unsat_cp = unsat;
    auto pair_select = [&](const segment& s){
        coverage.insert(pSol,s);
        coverage.filter_unsatisfied_cells(pSol, unsat);
        n12::evaluated_trial new_trial =
                select_best_pair(pSol,s, box, unsat);

        trial_selector(best_trial, new_trial);
        coverage.remove(pSol,s);
        unsat = unsat_cp;
    };
    //Allow for just splitting self
    segment dummy = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    for_each_segment(box, dummy, pair_select);
    return best_trial;
}

n12::evaluated_trial neighborhood_n12::select_best_pair(cppied_solution& pSol,
                                              const segment& s1,
                                              const iRectangle& box,
                                              std::vector<int>& unsat) {
    n12::evaluated_trial best_trial;
    best_trial.t = {{{path_engine::NULL_NODE, path_engine::NULL_NODE}, {}},
                    {{path_engine::NULL_NODE, path_engine::NULL_NODE}, {}}};
    best_trial.gain = {std::numeric_limits<int>::max(),
                       std::numeric_limits<int>::max()};

    std::map<segment, std::vector<std::pair<int,cost_t>>> top_k_cache;
    auto sat_topologic_order = [&](
            const segment& other){
        return s1.source <= other.source;
    };
    auto trial_select = [&](const segment& seg){
        if (sat_topologic_order(seg) &&
            coverage.insertion_satisfies(pSol, seg, unsat)){

            coverage.insert(pSol, seg);
            std::array<std::pair<segment, segment>, 2> trim_configs{{
              {s1, seg},
              {seg, s1}
            }};

            set_trims(pSol, trim_configs);
            std::array<std::pair<segment, segment>, 8> test_configs{{
                {trim_configs[0].first, trim_configs[0].second},
                {trim_configs[0].first, geometry.flip_segment(trim_configs[0].second)},
                {geometry.flip_segment(trim_configs[0].first), trim_configs[0].second},
                {geometry.flip_segment(trim_configs[0].first), geometry.flip_segment(trim_configs[0].second)},
                {trim_configs[1].first, trim_configs[1].second},
                {trim_configs[1].first, geometry.flip_segment(trim_configs[1].second)},
                {geometry.flip_segment(trim_configs[1].first), trim_configs[1].second},
                {geometry.flip_segment(trim_configs[1].first), geometry.flip_segment(trim_configs[1].second)}
            }};

            for (auto& config : test_configs){
                n12::trial trial_basis = {{config.first,{}},{config.second, {}}};
                if (path_engine::is_node(trial_basis.s1.s.source) &&
                        top_k_cache.find(config.first) != top_k_cache.end()) {
                    trial_basis.s1.top_k = top_k_cache[config.first];
                } else if (path_engine::is_node(trial_basis.s1.s.source)){
                    trial_basis.s1.top_k = geometry.get_top_k_insertion(pSol, trial_basis.s1.s, 2);
                    top_k_cache[config.first] = trial_basis.s1.top_k;
                }
                if (path_engine::is_node(trial_basis.s2.s.source) &&
                    top_k_cache.find(config.second) != top_k_cache.end()) {
                    trial_basis.s2.top_k = top_k_cache[config.second];
                } else if (path_engine::is_node(trial_basis.s2.s.source)){
                    trial_basis.s2.top_k = geometry.get_top_k_insertion(pSol, trial_basis.s2.s, 2);
                    top_k_cache[config.second] = trial_basis.s2.top_k;
                }

                n12::evaluated_trial new_trial = select_best_insertions(pSol, trial_basis);
                trial_selector(best_trial, new_trial);
            }
            coverage.remove(pSol, seg);
        }
    };
    //Allow for just splitting self
    segment dummy = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    for_each_segment(box, dummy, trial_select);
    return best_trial;
}

void neighborhood_n12::set_trims(cppied_solution &pSol,
                                 std::array<std::pair<segment, segment>, 2> &candidates) {
    for (auto& candidate : candidates){
        segment s1_trimed = trim(pSol, candidate.first);
        coverage.remove(pSol,candidate.first);

        if (path_engine::is_node(s1_trimed.source))
            coverage.insert(pSol, s1_trimed);
        segment s2_trimed = trim (pSol, candidate.second);

        if (path_engine::is_node(s1_trimed.source))
            coverage.remove(pSol, s1_trimed);
        coverage.insert(pSol,candidate.first);

        candidate = {s1_trimed, s2_trimed};
    }
}

n12::evaluated_trial neighborhood_n12::select_best_insertions(cppied_solution &pSol,
                                                              n12::trial& candidate) {
    n12::evaluated_trial t;
    if (!path_engine::is_node(candidate.s1.s.source) && !path_engine::is_node(candidate.s2.s.source)){
        t = {candidate, cost_t{0,0}};
    } else if (!path_engine::is_node(candidate.s1.s.source)){
        std::swap(candidate.s1, candidate.s2);
        t = {candidate, candidate.s1.top_k[0].second};
    } else if (!path_engine::is_node(candidate.s2.s.source)){
        t = {candidate, candidate.s1.top_k[0].second};
    }
    if (!path_engine::is_node(candidate.s1.s.source) || !path_engine::is_node(candidate.s2.s.source))
        return t;

    if (candidate.s1.top_k[0].first == candidate.s2.top_k[0].first){
        cost_t best_cost = std::min(
                candidate.s1.top_k[0].second + candidate.s2.top_k[1].second,
                candidate.s1.top_k[1].second + candidate.s2.top_k[0].second);
        int i= candidate.s1.top_k[0].first;
        cost_t ordered_cost = geometry.dist(pSol.path[i-1], candidate.s1.s) +
                geometry.cost(candidate.s1.s) + geometry.dist(candidate.s1.s, candidate.s2.s) +
                geometry.cost(candidate.s2.s) + geometry.dist(candidate.s2.s, pSol.path[i])-
                geometry.dist(pSol.path[i-1], pSol.path[i]);
        cost_t reversed_cost = geometry.dist(pSol.path[i-1], candidate.s2.s) +
                               geometry.cost(candidate.s2.s) + geometry.dist(candidate.s2.s, candidate.s1.s) +
                               geometry.cost(candidate.s1.s) + geometry.dist(candidate.s1.s, pSol.path[i]) -
                               geometry.dist(pSol.path[i-1], pSol.path[i]);
        if (ordered_cost < best_cost && ordered_cost <= reversed_cost){
            t = {candidate, ordered_cost};
        } else if (reversed_cost < best_cost && reversed_cost < ordered_cost){
            std::swap(candidate.s1, candidate.s2);
            t = {candidate, reversed_cost};
        } else if (best_cost == candidate.s1.top_k[0].second + candidate.s2.top_k[1].second){
            candidate.s2.top_k[0].first = candidate.s2.top_k[1].first;
            t ={candidate,
                candidate.s1.top_k[0].second + candidate.s2.top_k[1].second};
        } else {
            candidate.s1.top_k[0].first = candidate.s1.top_k[1].first;
            t = {candidate,
                 candidate.s2.top_k[0].second + candidate.s1.top_k[1].second};
        }
    } else {
        t = {candidate,
             candidate.s2.top_k[0].second + candidate.s1.top_k[0].second};
    }
    return t;
}

cost_t neighborhood_n12::gain(const cppied_solution& pSol,
                              neighborhood::sVecIt source,
                              n12::evaluated_trial& t) {
    cost_t g = geometry.removal_gain(pSol, source);
    g -= t.gain;
    return g;
}

bool neighborhood_n12::apply_selector(cppied_solution& pSol,
                                      const n12::evaluated_candidates &candidates) {
    const int N = static_cast<int>(candidates.size());
    std::vector<n12::trial> trials(N);
    std::vector<cost_t>     gains(N);

    gains[0] = {std::numeric_limits<int>::min(),
                std::numeric_limits<int>::min()};

    // Stochastic reduction: trial_selector (Gumbel or default) picks one winner
    // per segment from its precomputed pool
    cost_t restart_cost = {INT_MAX, INT_MAX};
    for (int i = 1; i < N; ++i) {
        if (candidates[i].empty()) {
            gains[i] = {INT_MIN, INT_MIN};
            continue;
        }
        n12::evaluated_trial best{
                {{{path_engine::NULL_NODE, path_engine::NULL_NODE}, {}},
                 {{path_engine::NULL_NODE, path_engine::NULL_NODE}, {}}},
                restart_cost
        };
        for (const auto& et : candidates[i])
            trial_selector(best, et);

        trials[i] = best.t;
        gains[i]  = best.gain;
    }

    auto best = std::max_element(gains.begin(), gains.end());
    if (*best <= cost_t{0,0} - allowed_degradation)
        return false;

    // Reuse the existing ris_heuristic + update logic (identical to local_search)
    // … (splitter / filter / interval_function lambdas unchanged) …
    selection_heuristic_update(pSol, trials, gains);
    return true;
}

void neighborhood_n12::precompute(cppied_solution &pSol,
                                  n12::evaluated_candidates &candidates_shell) {
    auto saved_selector = trial_selector;
    candidates_shell.resize(pSol.path.size());

    int i=1;
    trial_selector = [&](n12::evaluated_trial& best,
                        const n12::evaluated_trial& other) {
        candidates_shell[i].push_back(other);
        correct_trial(candidates_shell[i].back(), i);
    };
    for(; i< pSol.path.size(); ++i){
        auto path_it = std::next(pSol.path.begin(),i);
        replace(pSol, path_it);
    }

    //Correct computed gains;
    for (i=1; i< pSol.path.size(); i++){
        for (auto& t : candidates_shell[i])
            t.gain = gain(pSol, std::next(pSol.path.begin(), i), t);
    }

    trial_selector = saved_selector;
}

