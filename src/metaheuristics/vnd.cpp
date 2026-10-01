#include "../../include/metaheuristics/vnd.hpp"

bool vnd::search(cppied_solution &pSolution) {
    cost_t previous_cost = pSolution.cost;
    cost_t initial_cost = previous_cost;
    do {
        previous_cost = pSolution.cost;
        simpleNeighborhoodsDescent(pSolution);
        nestedNeighborhoodsDescent(pSolution);
        largeNeighborhoodsDescent(pSolution);
        setCompleteSolution(pSolution);
    } while (pSolution.cost < previous_cost);
    return pSolution.cost < initial_cost;
}

void vnd::simpleNeighborhoodsDescent(cppied_solution &pSol) {
    cost_t best_cost = pSol.cost;
    for (const auto& i : N_setup) {
        if (callbacks.stopCriteria && callbacks.stopCriteria(pSol, nullptr))
            break;
        find_local_optima(pSol, i);
    }

    do {
        best_cost = pSol.cost;
        if (callbacks.stopCriteria && callbacks.stopCriteria(pSol, nullptr))
            break;
        for (const auto &i : N_simple)
            find_local_optima(pSol, i);
    } while (pSol.cost < best_cost);
}

void vnd::nestedNeighborhoodsDescent(cppied_solution &pSol) {
    for (const auto &i : N_nested) {
        if (callbacks.stopCriteria && callbacks.stopCriteria(pSol, nullptr))
            break;
        i->local_search(pSol);
    }
}

void vnd::largeNeighborhoodsDescent(cppied_solution &pSol) {
    for (const auto& i : N_large) {
        if (callbacks.stopCriteria && callbacks.stopCriteria(pSol, nullptr))
            break;
        i->local_search(pSol);
    }
}

void vnd::find_local_optima(cppied_solution &pSol, const std::unique_ptr<neighborhood> &N) {
    bool improved = true;
    while (improved) {
        if (callbacks.stopCriteria && callbacks.stopCriteria(pSol, nullptr))
            break;
        improved = N->local_search(pSol);
    }
}

