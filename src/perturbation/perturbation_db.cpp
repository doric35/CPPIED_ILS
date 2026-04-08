#include "../../include/perturbations/perturbation_db.hpp"

void perturbation_db::d_perturbate(cppied_solution &pSol) {
    std::vector<int> db_sample;
    int n_candidates = (int(pSol.path.size()) / 2) - 2;
    int k = static_cast<int>(std::max(std::log2(pSol.path.size()), 8.0));
    k = std::min(k, n_candidates);
    if (k < 4)
        return;
    db_sample.reserve(k);
    uniform_sample(pSol, db_sample, k);
    int ref_idx = greedy_sample(pSol, db_sample);
    int ref = db_sample[ref_idx];  // ✅ convert to path index

    top_6_sample(pSol, db_sample, ref);
    bridge_sample(pSol, db_sample, ref);
    std::sort(db_sample.begin(), db_sample.end());
    double_bridge(pSol, db_sample);
}

void perturbation_db::uniform_sample(cppied_solution& pSol,
                    std::vector<int>& sampling,
                    int k){
    std::vector<int> candidates;
    candidates.reserve(pSol.path.size());
    std::uniform_int_distribution<> d(0,1);
    int mod = d(rng);
    for (int i=1; i< pSol.path.size() - 1 ; i++){
        if (i % 2 == mod)
            candidates.push_back(i);
    }
    std::shuffle(candidates.begin(), candidates.end(), rng);
    std::copy(candidates.begin(),
              candidates.begin() + k,
              std::back_inserter(sampling));
}

int perturbation_db::greedy_sample(cppied_solution &pSol,
                                   std::vector<int> &candidates) {
    std::vector<cost_t> gains;
    gains.reserve(candidates.size());

    auto func = [&](int i){
        return gain(pSol, candidates, i);
    };

    std::transform(candidates.begin(), candidates.end(),
                   std::back_inserter(gains), func);

    auto elem = std::max_element(gains.begin(), gains.end());
    int sample = static_cast<int>(std::distance(gains.begin(), elem));
    return sample;
}

void perturbation_db::top_6_sample(cppied_solution &pSol,
                                   std::vector<int> &candidates,
                                   int ref) {
    std::vector<cost_t> costs;
    costs.reserve(candidates.size());

    auto func = [&](int i){
        if (i==ref)
            return cost_t{0,0};
        return geometry.dist(pSol.path[ref], pSol.path[i]);
    };

    std::transform(candidates.begin(), candidates.end(),
                   std::back_inserter(costs), func);

    std::vector<int> sample(candidates.size(), 0);
    std::iota(sample.begin(), sample.end(), 0);

    auto cmp = [&](int i, int j){
        return costs[i] < costs[j];
    };
    std::stable_sort(sample.begin(), sample.end(), cmp);
    std::vector<int> top_6;
    top_6.reserve(6);
    for ( int i=0; i< std::min(6, int(sample.size())); i++)
        top_6.push_back(candidates[sample[i]]);

    candidates.swap(top_6);
}

void perturbation_db::bridge_sample(cppied_solution &, std::vector<int> &candidates, int ref) {
    std::uniform_real_distribution<> d(0.0,1.0);
    std::vector<std::pair<int,double>> paired_weights;
    paired_weights.reserve(candidates.size());
    for (int i:candidates){
        if (i==ref)
            paired_weights.emplace_back(i, -1.0);
        else
            paired_weights.emplace_back(i, d(rng));
    }
    auto cmp = [&](
            const std::pair<int,double>& a, const std::pair<int,double>& b){
        return a.second < b.second;
    };
    std::stable_sort(paired_weights.begin(), paired_weights.end(), cmp);

    candidates.clear();
    for (int i=0; i< 4; i++)
        candidates.push_back(paired_weights[i].first);

}

cost_t perturbation_db::gain(cppied_solution &pSol,
                             std::vector<int> &sampling,
                             int i) {
    cost_t best = {std::numeric_limits<int>::max(),
                   std::numeric_limits<int>::max()};
    cost_t base = geometry.dist(pSol.path[i], pSol.path[i+1]);
    for (int j : sampling){
        if (i != j && base + geometry.dist(pSol.path[i],pSol.path[j]) < best)
            best = base + geometry.dist(pSol.path[i],pSol.path[j]);

    }
    return best;
}

void perturbation_db::double_bridge(cppied_solution &pSol,
                                    std::vector<int> &selection) {
    std::vector<segment> new_sol;
    new_sol.reserve(2*pSol.path.size());
    for (int i=0; i <= selection[0]; i++)
        new_sol.push_back(pSol.path[i]);
    geometry.link(new_sol, pSol.path[selection[2]+1]);
    for (int i=selection[2] + 2; i<= selection[3]; i++)
        new_sol.push_back(pSol.path[i]);
    geometry.link(new_sol, pSol.path[selection[1] + 1]);
    for (int i=selection[1] + 2; i<= selection[2]; i++)
        new_sol.push_back(pSol.path[i]);
    geometry.link(new_sol, pSol.path[selection[0] + 1]);
    for (int i=selection[0] + 2; i<= selection[1]; i++)
        new_sol.push_back(pSol.path[i]);
    geometry.link(new_sol, pSol.path[selection[3] + 1]);
    for (int i=selection[3] + 2; i< pSol.path.size(); i++)
        new_sol.push_back(pSol.path[i]);
    pSol.path.swap(new_sol);
    pSol.cost = geometry.cost(pSol);
}
