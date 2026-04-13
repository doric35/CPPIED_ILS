#include "../../include/metaheuristics/ils.hpp"

void ils::d_solve(cppied_solution &pSolution) {
    std::cout << "Construction " << std::endl;
    if (!R.empty()){
        auto starter = uniform_sample(R);
        std::cout << "Selected starter " << std::endl;
        starter->restart(pSolution);
    } else {
        dp_sweeper h(ctx, problem);
        auto saver = [](const cppied_solution& pSol){
            ;
        };
        h.construct(pSolution, saver);
    }
    int iterations = 1;
    geometry.complete(pSolution);
    coverage.reset(pSolution);
    //std::cout << "Initial cost: " << pSolution.cost << std::endl;

    auto cb_tmp = callbacks.onSatisfy;

    if (callbacks.onSatisfy){
        auto sat_check = [&](
                const cppied_solution& sol, const std::any& ctx){
            auto msg = std::any_cast<std::string>(ctx);
            if (pSolution.cost != geometry.cost(pSolution)){
                std::cerr << "Solution cost miscalculated: " << msg << std::endl;
                assert(false);
            };
            auto tmp = sol;
            coverage.reset(tmp);
            if (!tmp.coverage.isApprox(sol.coverage, 1e-9)){
                std::cerr << "Solution coverage drift: " << msg << std::endl;
                assert(false);
            }
            if (!(sol.coverage.array() >= problem.req.array()).all()){
                std::cerr << "Solution coverage unsat: " << msg << std::endl;
                assert(false);
            }
        };
        callbacks.onSatisfy = sat_check;
    }

    if (R.empty() && P.empty() && ls.neighborhoods_count() <=2)
        return;

    if (callbacks.onSatisfy) {
        std::string msg = "Before starting first ILS local search.";
        callbacks.onSatisfy(pSolution, msg);
    }

    local_search(pSolution);
    auto trial = pSolution;
    luby L;

    if (callbacks.onSatisfy) {
        std::string msg = "Before starting ils iterations.";
        callbacks.onSatisfy(pSolution, msg);
    }

    CPPIEDCallbacks ls_callbacks{};
    ls_callbacks.onSatisfy = callbacks.onSatisfy;
    ls_callbacks.stopCriteria = [&](const cppied_solution& sol, const std::any& ctx){
        return stopping_criterion();
    };
    ls.set_callbacks(ls_callbacks);

    while (!stopping_criterion()){
        int no_improvement_allowed = L(iterations++);
        int no_improvement = 0;
        while (no_improvement <= no_improvement_allowed && !stopping_criterion()){
            if (!P.empty()){
                auto local_trial = trial;
                auto perturbater = uniform_sample(P);
                try {
                    perturbater->perturbate(local_trial);
                } catch (std::runtime_error& e){
                    std::cerr << "Runtime error from solution perturbation: " <<
                        typeid(*perturbater).name() << std::endl;
                    std::cerr << e.what();
                    for (auto& seg: trial.path)
                        std::cerr << seg << std::endl;
                    throw e;
                }
                if (callbacks.onSatisfy) {
                    std::string msg = std::string("From ILS perturbation: ") + typeid(*perturbater).name();
                    callbacks.onSatisfy(local_trial, msg);
                }

                local_search(local_trial);
                if (local_trial.cost < trial.cost)
                    trial = local_trial;
                if (trial.cost >= pSolution.cost)
                    ++no_improvement;
                else
                    pSolution = trial;
            } else {
                local_search(trial);
                if (trial.cost >= pSolution.cost){
                    ++no_improvement;
                    trial = pSolution;
                } else {
                    pSolution = trial;
                }
            }
        }
        if (trial.cost < pSolution.cost)
            pSolution = trial;
        else
            trial = pSolution;
        if (!R.empty()){
            auto starter = uniform_sample(R);
            starter->restart(trial);
            if (callbacks.onSatisfy) {
                std::string msg = "After ILS restart.";
                callbacks.onSatisfy(trial, msg);
            }
        }
        geometry.complete(trial);
        coverage.reset(trial);
        local_search(trial);
    }
    geometry.complete(pSolution);
    coverage.reset(pSolution);
    callbacks.onSatisfy = cb_tmp;
}

bool ils::local_search(cppied_solution &pSol){
    cost_t c = pSol.cost;
    ls.search(pSol);
    return c > pSol.cost;
}

bool ils::stopping_criterion() {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = current_time - ctx.start_time;
    int elapsed_sec = static_cast<int>(
            std::chrono::duration_cast<std::chrono::seconds>(elapsed).count()
    );
    return elapsed_sec >= ctx.max_time;
}
