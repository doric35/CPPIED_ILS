#pragma once

#include "neighborhood.hpp"

namespace n12 {
    struct candidate {
        segment s;
        std::vector<std::pair<int, cost_t>> top_k;
    };

    struct trial {
        candidate s1;
        candidate s2;
    };

    struct evaluated_trial{
        trial t;
        cost_t gain;
    };

    using evaluated_candidates = std::vector<std::vector<evaluated_trial>>;
}

class neighborhood_n12 : public neighborhood{
public:
    using neighborhood::neighborhood;

    void set_degradation(cost_t c){
        allowed_degradation = c;
    }

    void set_selector(std::function<void(n12::evaluated_trial&,
                                         const n12::evaluated_trial&)> f){
        trial_selector = f;
    }

    bool local_search(cppied_solution&) override;
    void selection_heuristic_update(cppied_solution& pSol,
                                    std::vector<n12::trial> &trials,
                                    std::vector<cost_t> &gains);
    n12::evaluated_trial replace(cppied_solution&, sVecIt seg);
    n12::evaluated_trial explore_replacements(cppied_solution& pSol, const segment& ref);
    cost_t gain(const cppied_solution&,
                neighborhood::sVecIt source,
                n12::evaluated_trial& t);
    n12::evaluated_trial select_best_pair(cppied_solution&,
                                const segment& s1,
                                const iRectangle& box,
                                std::vector<int>& unsat);
    n12::evaluated_trial select_best_insertions(cppied_solution&,
                                                n12::trial& candidate);

    void set_trims(cppied_solution&,
                   std::array<std::pair<segment, segment>, 2>& candidates);

    void update(cppied_solution&,
                const std::vector<n12::trial>& trials,
                const std::vector<cost_t>& gains,
                std::vector<int>& selection);

    bool apply_selector(cppied_solution& pSol,
                        const n12::evaluated_candidates& candidates);
    void precompute(cppied_solution& pSol,
                    n12::evaluated_candidates& candidates_shell);
    void correct_trial(n12::evaluated_trial& t, int p);
protected:
    cost_t allowed_degradation{0,0};
    std::function<void(n12::evaluated_trial&,
                       const n12::evaluated_trial&)> trial_selector = [](
                               n12::evaluated_trial& best,
                               const n12::evaluated_trial& other){
        if (other.gain < best.gain)
            best = other;
    };
};