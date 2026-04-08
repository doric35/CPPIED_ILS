#include "../../include/heuristics/ris_heuristic.hpp"

void ris_heuristic::solve_selection(const std::vector<cost_t> &pGains,
                                    std::vector<int> &pSelection) {
    std::vector<int> candidates;
    candidates.resize(pGains.size());
    std::iota(candidates.begin(), candidates.end(), 0);

    std::vector<std::vector<int>> candidate_sets;
    filter_function(candidates);
    split_function(candidates,candidate_sets);

    std::vector<std::vector<interval>> interval_sets(candidate_sets.size());

    for (int i=0; i< candidate_sets.size(); i++)
        interval_function(candidate_sets[i], interval_sets[i]);

    if (!interval_sets.empty()){
        std::vector<bool> no_goods(interval_sets.size(), false);
        int no_good_count=0;
        std::vector<std::vector<interval>> solution_sets;
        solution_sets.reserve(interval_sets.size());
        std::vector<cost_t> gain_sets(interval_sets.size(),
                                      cost_t{0,0});

        for (size_t i=0; i<interval_sets.size(); i++) {
            solution_sets.emplace_back();
            if (!interval_sets[i].empty())
                set_selection(interval_sets[i],
                              solution_sets[i], gain_sets[i]);
            else {
                no_good_count +=1;
                no_goods[i] = true;
            }
        }
        if (no_good_count < interval_sets.size()){
            int item = softmax_sample(gain_sets,
                                      no_goods);
            pSelection.reserve(solution_sets[item].size());
            for (auto& i : solution_sets[item])
                pSelection.push_back(i.id);
        }
    }
}

int ris_heuristic::softmax_sample(std::vector<cost_t> &gains,
                                         std::mt19937& generator) {
    int max_turns = 0;
    for (auto& g:gains)
        max_turns = std::max(max_turns, g.turns);

    int M = max_turns + 1;
    std::vector<double> scores(gains.size());
    double max_score = 0.0;
    for (size_t i=0; i<gains.size(); i++){
        scores[i] = M * gains[i].length + gains[i].turns;
        max_score = std::max(max_score, scores[i]);
    }

    auto softmax = [max_score](double score){
        return std::exp(score - max_score);
    };

    std::transform(scores.cbegin(), scores.cend(),
                   scores.begin(), softmax);

    std::discrete_distribution<> d(scores.begin(),
                                   scores.end());
    int item = d(generator);

    return item;
}

int ris_heuristic::softmax_sample(std::vector<cost_t> &gains,
                                  std::vector<bool>& no_goods) {
    int max_turns = 0;
    for (auto& g:gains)
        max_turns = std::max(max_turns, g.turns);

    int M = max_turns + 1;
    std::vector<double> scores(gains.size());
    double max_score = 0.0;
    for (size_t i=0; i<gains.size(); i++){
        scores[i] = M * gains[i].length + gains[i].turns;
        max_score = std::max(max_score, scores[i]);
    }

    auto softmax = [max_score](double score){
        return std::exp(score - max_score);
    };

    std::transform(scores.cbegin(), scores.cend(),
                   scores.begin(), softmax);

    std::discrete_distribution<> d(scores.begin(),
                                   scores.end());
    int item = d(rng);
    while (no_goods[item])
        item = d(rng);
    return item;
}

void ris_heuristic::set_selection(std::vector<interval> &intervals,
                                  std::vector<interval> &solution,
                                  cost_t &gain) {
    solution.reserve(intervals.size());

    auto lb_cmp = [](
            const interval& a, int y){
        return a.y < y;
    };
    auto sort_cmp = [](
            const interval& a, const interval& b){
        if (a.y == b.y)
            return a.x2 < b.x2;
        return a.y < b.y;
    };

    std::sort(intervals.begin(), intervals.end(), sort_cmp);
    int start = intervals.front().y;
    int end = intervals.back().y;

    std::vector<std::vector<interval>> sols(end - start + 1);
    std::vector<cost_t> gains(end - start + 1, cost_t{0,0});

    auto source = intervals.begin();
    int i=0;
    while (source != intervals.end()){
        i = source->y - start;
        auto target = std::lower_bound(
                source, intervals.end(),
                source->y + 1, lb_cmp);
        interval_dp(source, target, sols[i], gains[i]);

        source = target;
    }

    std::vector<int> spacing_sol;
    spacing_sol.reserve(gains.size());
    spacing_dp(gains, spacing_sol, gain);

    for (int item: spacing_sol)
        solution.insert(solution.end(),
                        sols[item].begin(),
                        sols[item].end());
}

void ris_heuristic::interval_dp(ris_heuristic::iVecIt begin,
                                ris_heuristic::iVecIt end,
                                std::vector<interval> &sol,
                                cost_t &gain) {
    //Intialization: linear time
    int n_items = static_cast<int>(std::distance(begin, end));
    if (n_items == 0) {
        gain = cost_t{0,0};
        return;
    }
    std::vector<cost_t> DP(n_items);
    std::vector<int> P(n_items, -1);
    DP[0] = begin->gain;

    auto cmp = [&](
            const interval& a,
            int bound){
        return a.x2 < bound;
    };

    //Forward pass
    int i = 1;
    auto item = std::next(begin);
    for (; i< n_items; ++i, ++item){
        int bound = item->x1;
        auto lb = std::lower_bound(
                begin, item,
                bound, cmp);
        if (lb == begin) {
            P[i] = -1;
            DP[i] = std::max(DP[i-1], item->gain);
        } else {
            P[i] = static_cast<int>(std::distance(begin, lb)) - 1;
            DP[i] = std::max(DP[i-1],
                             item->gain + DP[P[i]]);
        }
    }

    //Backward pass: Linear time
    std::uniform_real_distribution<> d(0.0, 1.0);
    i = n_items - 1;
    while (i >= 0) {
        const auto& current = *(begin + i);
        cost_t include = current.gain + (P[i] >= 0 ? DP[P[i]] : cost_t{0,0});
        cost_t exclude = (i > 0 ? DP[i-1] : cost_t{0,0});

        if (include > exclude) {
            sol.push_back(current);
            i = P[i];
        } else if (include < exclude) {
            i = i - 1;
        } else {
            // tie → random choice
            if (d(rng) < 0.5) {
                sol.push_back(current);
                i = P[i];
            } else {
                i = i - 1;
            }
        }
    }
    gain = DP.back();
}

void ris_heuristic::spacing_dp(
        const std::vector<cost_t> & gains,
        std::vector<int> &sol, cost_t &gain) {
    //Intialization: linear time
    std::vector<cost_t> dp_value(gains.size());
    dp_value[0] = gains[0];

    //Forward Pass: linear time
    size_t base_iter = std::min(static_cast<size_t>(spacing),
                             gains.size());
    dp_value[0] = gains[0];
    int i=1;
    for (; i<base_iter; ++i)
        dp_value[i] = std::max(gains[i], dp_value[i-1]);

    for (; i< gains.size(); ++i)
        dp_value[i] = std::max(dp_value[i-1],
                               dp_value[i-spacing] + gains[i]);

    //Backward Pass: pseudo linear time
    auto cmp = [&](
            const cost_t& a,
            const cost_t& v){
        return a < v;
    };

    i = int(gains.size())-1;
    while (i >= base_iter){
        if (dp_value[i] > dp_value[i-1]) {
            sol.push_back(i);
            i -= spacing;
        }

        else if (dp_value[i] == dp_value[i-1] &&
                 dp_value[i] - gains[i] == dp_value[i-spacing]){
            auto lb = std::lower_bound(std::next(dp_value.begin(),i - spacing + 1),
                                       std::next(dp_value.begin(), i),
                                       dp_value[i], cmp);

            int a,b;
            a = 0, b = static_cast<int>(std::distance(lb, std::next(dp_value.begin(), i)));

            std::uniform_int_distribution<int> dist(a,b);
            int selection = dist(rng);
            i -= selection;

            if ( i >= spacing && dp_value[i] - gains[i] == dp_value[i-spacing]){
                sol.push_back(i);
                i -= spacing;
            } else if (i < spacing)
                continue;
            else --i;
        } else --i;
    }

    std::uniform_real_distribution<> d(0.0,1.0);
    double w = -1.0;
    int selection = i;
    for (int j=0; j< i; j++){
        if (gains[j] == dp_value[i]){
            double wj = d(rng);
            if (wj > w){
                selection = j;
                w = wj;
            }
        }
    }
    assert(gains[selection] == dp_value[i]);

    sol.push_back(selection);
    gain = dp_value.back();
}