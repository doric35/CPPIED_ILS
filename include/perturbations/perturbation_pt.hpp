#pragma once

#include "perturbation.hpp"
#include "../neighborhoods/neighborhood_trim.hpp"

class perturbation_pt : public perturbation{
public:
    using perturbation::perturbation;

    void d_perturbate(cppied_solution&) override;
    void uniform_sample(cppied_solution&, std::vector<int>& sampling);
    void replace(cppied_solution&, std::vector<int>& sampling);
    void replace_segment(cppied_solution&,
                         std::vector<segment>& new_sol,
                         int i);

protected:
    double ratio = 0.05;
};