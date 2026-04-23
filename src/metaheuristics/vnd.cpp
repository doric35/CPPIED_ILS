#include "../../include/metaheuristics/vnd.hpp"

bool vnd::search(cppied_solution &pSolution) {
    cost_t c = pSolution.cost;
    cost_t i_c = c;
    cost_t c_local{};

    do {
        c = pSolution.cost;
        if (callbacks.stopCriteria && callbacks.stopCriteria(pSolution, nullptr))
            break;

        for (const auto& i : N_setup)
            find_local_optima(pSolution, i);

        do {
            c_local = pSolution.cost;
            if (callbacks.stopCriteria && callbacks.stopCriteria(pSolution, nullptr))
                break;
            for (const auto &i : N_simple)
                find_local_optima(pSolution, i);
        } while (pSolution.cost < c_local);

        bool nest_improved = false;
        for (const auto &i : N_nested) {
            if (callbacks.stopCriteria && callbacks.stopCriteria(pSolution, nullptr))
                break;
            bool improved = i->local_search(pSolution);
            nest_improved = improved || nest_improved;
        }
        if (nest_improved) continue;

        for (const auto& i : N_large) {
            if (callbacks.stopCriteria && callbacks.stopCriteria(pSolution, nullptr))
                break;
            i->local_search(pSolution);
        }
    } while (pSolution.cost < c);
    return pSolution.cost < i_c;
}

void vnd::find_local_optima(cppied_solution &pSol, const std::unique_ptr<neighborhood> &N) {
    bool improved = true;
    while (improved) {
        if (callbacks.stopCriteria && callbacks.stopCriteria(pSol, nullptr))
            break;
        improved = N->local_search(pSol);
    }
}

