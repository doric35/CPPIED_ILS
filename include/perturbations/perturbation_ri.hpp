#pragma once

#include "perturbation.hpp"

class perturbation_ri : public perturbation{
public:
    using perturbation::perturbation;

    void d_perturbate(cppied_solution&) override;
    void bias_sample(cppied_solution&, std::vector<int>& sampling);
    void best_insertions(cppied_solution&, std::vector<int>& sampling);

protected:
    double ratio = 0.05;
};