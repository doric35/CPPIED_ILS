#include "../../include/perturbations/perturbation_ri.hpp"

void perturbation_ri::d_perturbate(cppied_solution &pSol) {
    std::vector<int> insertion_sample;
    insertion_sample.reserve(static_cast<size_t>(std::ceil(ratio*double(pSol.path.size()))));

    bias_sample(pSol, insertion_sample);
    best_insertions(pSol, insertion_sample);
}

void perturbation_ri::bias_sample(cppied_solution &pSol,
                                  std::vector<int> &sampling) {
    std::vector<double> weights(problem.vertex.size(), 0.0);
    for (int v=0; v < weights.size(); v++){
        segment dummy = {v, path_engine::NULL_NODE};
        weights[v] = coverage.over_coverage(pSol, dummy);
    }
    std::uniform_real_distribution<> dis(0.0, 1.0);

    std::vector<std::pair<double,int>> associative_container;
    associative_container.reserve(weights.size());

    double key;
    for (int i=0; i<weights.size(); i++){
        double u = dis(rng);
        if (weights[i] <= 0.0)
            key = -std::numeric_limits<double>::infinity();
        else
            key = std::log(u)/weights[i];
        associative_container.emplace_back(key, i);
    }

    auto cmp = [](
            const std::pair<double,int>& a, const std::pair<double,int>& b){
        return a.first > b.first;
    };

    std::nth_element(associative_container.begin(),
                                 associative_container.begin() + static_cast<size_t>(std::ceil(ratio*pSol.path.size())),
                                 associative_container.end(), cmp);

    for (int i=0; i< static_cast<size_t>(std::ceil(ratio*pSol.path.size())); ++i)
        sampling.push_back(associative_container[i].second);
}

void perturbation_ri::best_insertions(cppied_solution &pSol,
                                      std::vector<int> &sampling) {
    std::shuffle(sampling.begin(), sampling.end(), rng);
    int k = static_cast<int>(std::ceil(ratio * double(pSol.path.size())));
    std::array<int, 2> orientations{path_engine::NULL_NODE, path_engine::REVERSED_NULL_NODE};
    std::uniform_int_distribution<> dist(0, 1);
    std::set<int> no_goods;
    std::vector<std::pair<int, segment>> candidates;
    candidates.reserve(k);

    for (int i=0; i<sampling.size(); i++){
        candidates.emplace_back(0,
                                segment{sampling[i], orientations[dist(rng)]});
        std::vector<std::pair<int,cost_t>> result =
                std::move(geometry.get_top_k_insertion(pSol, candidates.back().second, i+1));

        for (auto& candidate_insertion : result){
            auto elem = no_goods.find(candidate_insertion.first);
            if (elem == no_goods.end()){
                no_goods.insert(candidate_insertion.first);
                candidates.back().first = candidate_insertion.first;
                break;
            }
        }
    }
    auto cmp = [](
            const std::pair<int, segment>& a, const std::pair<int, segment>& b){
        return a.first < b.first;
    };
    std::sort(candidates.begin(),candidates.end(), cmp);

    std::vector<segment> new_sol;
    new_sol.reserve(pSol.path.size() + k);
    int j=0;
    for (int i=0; i< pSol.path.size(); i++){
        if (j < candidates.size() && i == candidates[j].first){
            new_sol.push_back(candidates[j].second);
            coverage.insert(pSol, candidates[j].second);
            ++j;
        }
        new_sol.push_back(pSol.path[i]);
    }
    pSol.path.swap(new_sol);
    pSol.cost = geometry.cost(pSol);
}