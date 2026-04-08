#include "../../include/perturbations/perturbation_pt.hpp"

void perturbation_pt::d_perturbate(cppied_solution &pSol) {
    int k = static_cast<int>(std::ceil(ratio*double(pSol.path.size())));
    std::vector<int> sampling;
    sampling.reserve(k);

    uniform_sample(pSol, sampling);
    replace(pSol, sampling);

    pSol.cost = geometry.cost(pSol);
    neighborhood_trim N(ctx, problem);
    while (N.local_search(pSol)) continue;
}

void perturbation_pt::uniform_sample(cppied_solution &pSol,
                                     std::vector<int> &sampling) {
    int k = static_cast<int>(std::ceil(ratio*double(pSol.path.size())));
    std::vector<double> weights(pSol.path.size(), 0.0);
    std::uniform_real_distribution<> d(0.0,1.0);
    auto sample = [&](double& a){
        a = d(rng);
    };
    std::for_each(weights.begin(), weights.end(), sample);

    std::vector<int> candidates(pSol.path.size()-1, 0);
    std::iota(candidates.begin(), candidates.end(), 1);

    auto cmp = [&](int a, int b){
        return weights[a] < weights[b];
    };

    std::nth_element(candidates.begin(),
                     candidates.begin() + k,
                     candidates.end(),
                     cmp);

    std::copy(candidates.begin(),
              candidates.begin() + k,
              std::back_inserter(sampling));
}

void perturbation_pt::replace(cppied_solution &pSol,
                              std::vector<int> &sampling) {
    std::sort(sampling.begin(), sampling.end());
    std::vector<segment> new_sol;
    new_sol.reserve(2*pSol.path.size());
    int j=0;
    for (int i=0; i< pSol.path.size(); i++){
        if (j<sampling.size() && i == sampling[j]){
            replace_segment(pSol, new_sol, i);
            ++j;
        } else {
            new_sol.push_back(pSol.path[i]);
        }
    }
    pSol.path.swap(new_sol);
}

void perturbation_pt::replace_segment(cppied_solution &pSol,
                                      std::vector<segment> &new_sol,
                                      int i) {
    coverage.remove(pSol, pSol.path[i]);
    std::vector<int> unsat;
    coverage.unsatisfied_cells(pSol, pSol.path[i], unsat);
    iRectangle box = {
            {std::numeric_limits<int>::max(),
                    std::numeric_limits<int>::max()},
            {std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::min()}
    };
    coverage.unsatisfied_rectangle(unsat, box);
    bool horizontal = geometry.is_horizontal(pSol.path[i]);
    direction dir = direction::E;
    std::uniform_int_distribution<> d(0,1);
    int flip = d(rng);
    if (horizontal)
        dir = direction::S;
    if (flip)
        dir = flip_direction(dir);
    auto segment_select = [&](segment& s){
        bool new_horizontal = geometry.is_horizontal(s);
        if (new_horizontal != horizontal){
            geometry.correct_segment_direction(s, dir);
            dir = flip_direction(dir);
            new_sol.push_back(s);
            coverage.insert(pSol, s);
        }
    };
    for_each_segment(box, segment_select);
}