#include "../../include/restart/restart_backtrack.hpp"

void restart_backtrack::restart(cppied_solution &pSol) {
    std::vector<bool> choices;
    if (history.empty())
        sample(pSol);
    else{
        pSol.path.clear();
        pSol.coverage.setZero();
        choices = history.front();
        history.pop();
    }
    std::vector<bool> new_choices;
    auto save_history = [&](
            const cppied_solution& pSol){
        new_choices.push_back(!geometry.is_horizontal(pSol.path.back()));
        if (new_choices.size() > choices.size())
            history.push(new_choices);
    };

    int i =-1;
    auto choice_func = [&](
            const std::array<sweeper::segment_set,2>&){
        //std::cout << "In choice func" << std::endl;
        ++i;
        if (i==choices.size() - 1)
            return 1 - choices[i];
        else if (i >= choices.size())
            return -1;
        return int(choices[i]);
    };

    dp_sweeper method(ctx, problem);
    method.set_choice(choice_func);
    //std::cout << "Starting construct" << std::endl;
    method.construct(pSol, save_history);
}

void restart_backtrack::sample(cppied_solution &pSol) {
    std::vector<std::pair<int, double>> weights;
    std::vector<segment> new_sol;
    int n_items = 0;
    if (!pSol.path.empty())
        n_items = static_cast<int>(std::sqrt(pSol.path.size()-1));
    weights.reserve(n_items);
    std::uniform_real_distribution<> d(0.0,1.0);

    for (int i=1; i< pSol.path.size(); i++)
        weights.emplace_back(i, d(rng));

    auto cmp = [](
            const std::pair<int, double>& a, const std::pair<int, double>& b){
        return a.second < b.second;
    };
    std::sort(weights.begin(), weights.end(), cmp);

    new_sol.reserve(n_items);
    if (!pSol.path.empty())
        new_sol.push_back(pSol.path[0]);
    for (int i=0; i<n_items-1; i++)
        new_sol.push_back(pSol.path[weights[i].first]);

    pSol.path.swap(new_sol);
    if (int(pSol.path.size()) > 0)
        coverage.reset(pSol);
    else
        pSol.coverage.setZero();
}
