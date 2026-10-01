#include "../../include/metaheuristics/ils.hpp"

void ils::d_solve(cppied_solution &pSolution) {
    incumbent = pSolution;
    setInitialIncumbent();
    if (isConstructOnly()) { pSolution = incumbent; return; }

    rootLocalSearch();

    if (isLocalSearchOnly()){
        setTerminalIncumbent();
        pSolution = incumbent;
        return;
    }

    auto trial = incumbent;
    while (!stopping_criterion()){
        iteratedSearch(trial);
        trial = restart();
    }

    setTerminalIncumbent();
    pSolution = incumbent;
}

void ils::constructInitialSolution() {
    if (!R.empty()){
        auto starter = uniform_sample(R);
        //std::cout << "Selected starter " << std::endl;
        starter->restart(incumbent);
    } else {
        dp_sweeper h(ctx, problem);
        auto saver = [](const cppied_solution& pSol){
            ;
        };
        h.construct(incumbent, saver);
    }
    setCompleteSolution(incumbent);
}

void ils::rootLocalSearch() {
    setLocalSearchCallbacks();
    local_search(incumbent);
}

void ils::iteratedSearch(cppied_solution &pSolution) {
    int no_improvement_allowed = L(iterations++);
    int no_improvement_count = 0;
    while (no_improvement_count <= no_improvement_allowed && !stopping_criterion()){
        cppied_solution trial = perturbate(pSolution);
        local_search(trial);
        if (trial.cost < pSolution.cost) pSolution = trial;
        no_improvement_count += setIncumbent(pSolution);
    }
}

cppied_solution ils::perturbate(const cppied_solution &pSolution) {
    if (P.empty()) return pSolution;

    cppied_solution trial = pSolution;
    auto perturbation_function = uniform_sample(P);

    perturbation_function->perturbate(trial);
    return trial;
}

cppied_solution ils::restart() {
    cppied_solution trial = incumbent;
    if (!R.empty() && !stopping_criterion()){
        auto starter = uniform_sample(R);
        starter->restart(trial);
    }
    setCompleteSolution(trial);
    if (!stopping_criterion())
        local_search(trial);
    return trial;
}

void ils::setInitialIncumbent() {
    iterations = 1;
    std::cout << "Constructing initial solution..." << std::endl;
    constructInitialSolution();
    std::cout << "Done." << std::endl;

    std::cout << "Starting ILS iterations...\n" << std::endl;
    std::cout << "Iteration | Time (s) | Length | Turns " << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << iterations << " | " << getElapsedTime() << " | "
              << incumbent.cost.length << " | " << incumbent.cost.turns << std::endl;
    status = algorithm_flag::SUBOPTIMAL;
}

void ils::setTerminalIncumbent() {
    geometry.complete(incumbent);
    coverage.reset(incumbent);
    status = algorithm_flag::SUBOPTIMAL;
    std::cout << "SUCCESS TERMINATION" << std::endl;
}

void ils::setLocalSearchCallbacks() {
    CPPIEDCallbacks ls_callbacks{};

    ls_callbacks.stopCriteria = [&](const cppied_solution& sol, const std::any& ctx){
        return stopping_criterion();
    };

    ls.set_callbacks(ls_callbacks);
}

bool ils::local_search(cppied_solution &pSol){
    cost_t c = pSol.cost;
    ls.search(pSol);
    return c > pSol.cost;
}

int ils::setIncumbent(const cppied_solution &pSolution) {
    if (pSolution.cost < incumbent.cost){
        incumbent = pSolution;
        std::cout << iterations << " | " << getElapsedTime() << " | "
                  << incumbent.cost.length << " | " << incumbent.cost.turns << std::endl;
        return 0;
    } else return 1;
}

bool ils::stopping_criterion() {
    int elapsed_seconds = getElapsedTime();
    return elapsed_seconds >= ctx.max_time;
}

int ils::getElapsedTime() {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = current_time - ctx.start_time;
    int elapsed_sec = static_cast<int>(
            std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
    );
    return elapsed_sec;
}
