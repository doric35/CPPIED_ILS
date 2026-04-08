#pragma once

#include "neighborhood.hpp"

class neighborhood_cut : public neighborhood{
public:
    using neighborhood::neighborhood;

    bool local_search(cppied_solution&) override;
    segment cut(cppied_solution&, sVecIt seg);
    cost_t gain(const cppied_solution&, sVecIt target);
    void update(cppied_solution&,
                std::vector<segment>& trials,
                std::vector<int>& selection);
    std::array<segment, 2> cut_diff(const segment& a,
                                    segment& b);
};