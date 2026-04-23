#pragma once

#include "neighborhood.hpp"

class neighborhood_reduce : public neighborhood{
public:
    using neighborhood::neighborhood;

    bool local_search(cppied_solution&) override;
    cost_t gain(cppied_solution&, sVecIt target);
    void update(cppied_solution&,
                std::vector<int>& selection);
};