#pragma once

#include "neighborhood.hpp"

class neighborhood_trim : public neighborhood{
public:
    using neighborhood::neighborhood;

    bool local_search(cppied_solution&) override;
    cost_t gain(const cppied_solution&, sVecIt source, sVecIt target);
    void update(cppied_solution&,
                const std::vector<segment>& trials,
                const std::vector<cost_t>& gains,
                std::vector<int>& selection);
};