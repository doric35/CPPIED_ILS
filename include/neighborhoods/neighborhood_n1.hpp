#pragma once

#include "neighborhood.hpp"

class neighborhood_n1 : public neighborhood {
public:
    using neighborhood::neighborhood;

    bool local_search(cppied_solution&) override;
    std::pair<segment, cost_t> replace(cppied_solution&, sVecIt seg);
    cost_t gain(const cppied_solution&, sVecIt source, segment target);
    void select(const std::vector<segment>& trials,
                const std::vector<cost_t>& gains,
                std::vector<int>& selection);
    void update(cppied_solution&,
                const std::vector<segment>& trials,
                const std::vector<cost_t>& gains,
                std::vector<int>& selection);
};
