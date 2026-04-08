#pragma once

#include "neighborhood.hpp"
#include "neighborhood_n12.hpp"
#include "neighborhood_n21.hpp"

class neighborhood_nested : public neighborhood {
public:
    using neighborhood::neighborhood;

    bool local_search(cppied_solution&) override;
    void degrade(std::array<cppied_solution,4>&,
            cost_t allowed_degradation);
    void upgrade(std::array<cppied_solution,4>&);
};