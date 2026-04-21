#include "../../include/metaheuristics/vnd.hpp"

bool vnd::search(cppied_solution &pSolution) {
    bool local_improved = true, improved = false;
    //std::cout << "Entering VND." << std::endl;
    cost_t i_c = pSolution.cost;
    cost_t c = i_c;
    auto cb_tmp = callbacks.onSatisfy;
    std::vector<segment> tmp;
    if (callbacks.onSatisfy){
        auto sat_check =
                [&](const cppied_solution& sol, const std::any& ctx) {
        auto* n = std::any_cast<neighborhood*>(ctx);
        if (n != N[0].get() && improved && sol.cost >= c) {
            std::cerr << "Improvement miscalculated: " << typeid(*n).name() << std::endl;
            for (auto &e: tmp)
                std::cerr << e << std::endl;
            assert(false);
        }
        if (!(sol.coverage.array() >= problem.req.array()).all()) {
            std::cerr << "Coverage unsat: " << typeid(*n).name() << std::endl;
            for (auto &e: tmp)
                std::cerr << e << std::endl;
            assert(false);
        }
        if (sol.cost != geometry.cost(sol)) {
            std::cerr << "Solution cost miscalculated: " << typeid(*n).name() << std::endl;
            for (auto &e: tmp)
                std::cerr << e << std::endl;
            assert(false);
        }
        if (sol.cost > c) {
            std::cerr << "Solution cost increased in a local search: " << typeid(*n).name() << std::endl;
            for (auto &e: tmp)
                std::cerr << e << std::endl;
            assert(false);
        }
        };
        callbacks.onSatisfy = sat_check;
    }
    while (local_improved){
        if (callbacks.stopCriteria && callbacks.stopCriteria(pSolution, nullptr)) {
            //std::cout << "Called stopCriteria stopped VND." << std::endl;
            break;
        }

        local_improved = false;
        for (const auto &i : N){
            if (callbacks.stopCriteria && callbacks.stopCriteria(pSolution, nullptr)) {
                //std::cout << "Called stopCriteria stopped VND." << std::endl;
                break;
            }
            if (callbacks.onSatisfy)
                tmp = pSolution.path;
            improved = false;
            try {
                improved = i->local_search(pSolution);
            } catch (std::runtime_error& e){
                auto& r = *i.get();
                std::string err_msg = std::string("Runtime error from neighborhood: ") + typeid(r).name();
                std::cerr << err_msg << std::endl;
                std::cerr << e.what() << std::endl;
                for (auto &seg: pSolution.path)
                    std::cerr << seg << std::endl;
                throw e;
            }
            if (callbacks.onSatisfy)
                callbacks.onSatisfy(pSolution, i.get());
            local_improved = local_improved || improved;
        }
        if (!local_improved && pSolution.cost < c){
            c = pSolution.cost;
            geometry.complete(pSolution);
            coverage.reset(pSolution);
            local_improved = true;
        } else if (local_improved){
            assert(pSolution.cost <= c); //Cost should not get worst during search.
            c = pSolution.cost;
        }
    }
    callbacks.onSatisfy = cb_tmp;
    //std::cout << "Leaving VND." << std::endl;
    return pSolution.cost < i_c;
}

