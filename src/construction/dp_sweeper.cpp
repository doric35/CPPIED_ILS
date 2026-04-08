#include "../../include/construction/dp_sweeper.hpp"

void dp_sweeper::construct(cppied_solution &pSol, std::function<void(cppied_solution&)> f) {
    std::vector<int> unsat;
    unsat.reserve(geometry.n_rows * geometry.n_cols);

    if (pSol.path.empty()) {
        segment initial = {problem.initial_position, NULL_NODE};
        coverage.insert(pSol, initial);
        pSol.path.push_back(initial);
    }

    coverage.unsatisfied_cells(pSol, unsat);
    if (!unsat.empty()) {
        while (!unsat.empty()) {
            iRectangle box = {std::numeric_limits<int>::max(),
                              std::numeric_limits<int>::max(),
                              std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::min()};
            coverage.unsatisfied_rectangle(unsat, box);
            std::array<sweeper::segment_set, 2> S{{
                                                          {{}, 0.0, true},
                                                          {{}, 0.0, false}
                                                  }};
            S[0].S.reserve(geometry.n_rows + 1);
            S[1].S.reserve(geometry.n_cols + 1);

            auto set_candidates = [&](const segment &seg) {
                if (geometry.is_horizontal(seg))
                    S[0].S.push_back(maximum_subarray(pSol, seg));
                else
                    S[1].S.push_back(maximum_subarray(pSol, seg));
            };
            for_each_segment(box, set_candidates);
            filter_segment_set(pSol, S[0]);
            filter_segment_set(pSol, S[1]);
            select(pSol, S);
            unsat.clear();
            coverage.unsatisfied_cells(pSol, unsat);
            f(pSol);
        }
        for (int i = 1; i < pSol.path.size(); i += 2)
            pSol.path[i] = geometry.flip_segment(pSol.path[i]);
    }
    pSol.cost = geometry.cost(pSol);
    neighborhood_tsp n(ctx, problem);
    n.local_search(pSol);
}

void dp_sweeper::select(cppied_solution &pSol,
                        std::array<sweeper::segment_set, 2>& S) {
    bool select_horizontal = (S[0].gain > S[1].gain) ||
            (S[0].gain == S[1].gain && S[0].S.size() <= S[1].S.size());

    int choice = mChoice(S);
    if (!(choice == 0 || choice == 1)){
        choice = 0;
        if (!select_horizontal)
            choice = 1;
    }

    if (S[choice].S.empty())
        mPenalty /= 10.0;

    for (auto& p : S[choice].S){
        pSol.path.push_back(p.first);
        coverage.insert(pSol, p.first);
    }
}

void dp_sweeper::filter_segment_set(cppied_solution &,
                                    sweeper::segment_set &S) {
    std::vector<std::pair<segment,double>> new_set;
    new_set.reserve(S.S.size() / (2*problem.max_range));

    std::vector<double> DP(int(S.S.size()), 0.0);
    for (int i=0; i<std::min(2*problem.max_range, int(S.S.size())); ++i)
        DP[i] = S.S[i].second;

    for (int i = 2*problem.max_range; i<S.S.size(); ++i)
        DP[i]=std::max(DP[i-1],
                       DP[i-2*problem.max_range] + S.S[i].second);

    //Backtrack
    int i = static_cast<int>(DP.size()) - 1;
    while (i >= 2*problem.max_range){
        if (DP[i] > DP[i-1]) {
            new_set.emplace_back(S.S[i].first, S.S[i].second);
            i -= 2*problem.max_range;
        }
        else --i;
    }
    auto elem = std::max_element(DP.begin(), std::next(DP.begin(), i+1));
    if (*elem > 0.0){
        int j = static_cast<int>(std::distance(DP.begin(), elem));
        new_set.emplace_back(S.S[j].first, S.S[j].second);
    }
    S.S.swap(new_set);
    S.gain = DP.back();
}

std::pair<segment, double> dp_sweeper::maximum_subarray(cppied_solution &pSol, const segment &ref) {
    std::vector<double> gain;
    std::vector<int> position;
    gain.reserve(geometry.cost(ref).length+1);
    position.reserve(geometry.cost(ref).length+1);
    auto gain_func = [&](
            cppied_solution& pSol, int v){
        double marginal_gain = 0.0, local_gain = 0.0;
        for (SMdIt c(problem.s_pod, v); c; ++c){
            local_gain = std::min(problem.req(c.index()) - pSol.coverage(c.index()), c.value());
            local_gain = (local_gain >= 0) * local_gain - (local_gain < 0) * mPenalty;
            marginal_gain += local_gain;
        }
        if (marginal_gain <= 0.0)
            marginal_gain = -std::numeric_limits<double>::infinity();
        position.push_back(v);
        gain.push_back(marginal_gain);
    };
    coverage.apply(pSol, ref, gain_func);

    assert(!gain.empty());
    //Dp phase
    double sum = gain[0], best_sum = gain[0];
    int best_l = 0, best_u = 0, curr_l=0, curr_u=1;
    for (; curr_u < gain.size(); curr_u++){
        if (gain[curr_u] > sum + gain[curr_u]){
            sum = gain[curr_u];
            curr_l = curr_u;
        } else
            sum += gain[curr_u];
        if (sum > best_sum){
            best_sum = sum;
            best_l = curr_l;
            best_u = curr_u;
        }
    }
    if (best_l == best_u)
        return {segment{position[best_l], NULL_NODE}, best_sum};
    return {segment{position[best_l], position[best_u]}, best_sum};
}