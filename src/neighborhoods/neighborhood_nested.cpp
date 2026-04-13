#include "../../include/neighborhoods/neighborhood_nested.hpp"

bool neighborhood_nested::local_search(cppied_solution& pSol) {
    cost_t controlled_degradation = {static_cast<int>(std::ceil(pSol.cost.length*0.05)),
                                     static_cast<int>(std::ceil(pSol.cost.turns*0.05))};
    if (controlled_degradation <= cost_t{0,0})
        return false;

    std::array<cppied_solution, 4> candidates{
        pSol, pSol, pSol, pSol
    };

    degrade(candidates,
            controlled_degradation);
    upgrade(candidates);

    auto cmp = [&](
            const cppied_solution& a,
            const cppied_solution& b){
        return a.cost < b.cost;
    };

    auto elem = std::min_element(
            candidates.begin(),
            candidates.end(),
            cmp);
    if (elem->cost < pSol.cost){
        pSol = *elem;
        return true;
    }
    return false;
}

void neighborhood_nested::degrade(std::array<cppied_solution, 4> & pCandidates,
                                  cost_t allowed_degradation) {
    neighborhood_n12 degradation_nbh(ctx, problem);
    degradation_nbh.set_degradation(allowed_degradation);
    n12::evaluated_candidates trials_memory;
    degradation_nbh.precompute(pCandidates[0], trials_memory);

    auto gumbel = [](){
        double u = (double)rand() / RAND_MAX;
        return -std::log(-std::log(u));
    };

    double ref_weight = gumbel();
    cost_t restart_cost = {std::numeric_limits<int>::max(),
                           std::numeric_limits<int>::max()};
    double beta = 0.5;
    double turn_weight = 0.1;
    auto selector = [&](
            n12::evaluated_trial& best,
            const n12::evaluated_trial& other){
        double w = -beta * (other.gain.length + turn_weight * other.gain.turns) + gumbel();
        if (best.gain == restart_cost){
            best = other;
            ref_weight = w;
        } else if (w > ref_weight){
            ref_weight = w;
            best = other;
        }
    };
    degradation_nbh.set_selector(selector);
    for (auto& sol: pCandidates) {
        degradation_nbh.apply_selector(sol, trials_memory);
    }
}

void neighborhood_nested::upgrade(std::array<cppied_solution, 4> &pCandidates) {
    neighborhood_n21 improvemennt_nbh(ctx, problem);
    for (auto& sol: pCandidates)
        improvemennt_nbh.local_search(sol);
}