#pragma once

#include "neighborhood.hpp"

namespace n21 {
    struct trial {
        segment s;
        int other;
        int position;
        cost_t gain;
    };

    struct packing {
        std::set<int> P;
        Eigen::VectorXi Cover;
    };
}

class neighborhood_n21 : public neighborhood{
public:
    using neighborhood::neighborhood;

    bool local_search(cppied_solution&) override;
    n21::trial replace(cppied_solution&, sVecIt seg);
    n21::trial explore_replacements(cppied_solution& pSol,
                                    const segment& ref,
                                    cost_t extraction_gain);
    n21::trial select_best_trial(cppied_solution&,
                                 std::vector<int>& candidates,
                                 std::vector<int>& global_unsat);
    n21::trial select_boxed_candidate(cppied_solution&,
                                      const iRectangle& box,
                                      std::vector<int>& unsat,
                                      sVecIt ref);
    void configure_trial(cppied_solution&,
                         n21::trial& candidate);

    cost_t update(cppied_solution&,
                const std::vector<n21::trial>& trials,
                const std::vector<cost_t>& gains,
                std::vector<int>& selection);
};