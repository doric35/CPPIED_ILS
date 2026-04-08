#pragma once

#include "perturbation.hpp"

class perturbation_db : public perturbation{
public:
    using perturbation::perturbation;

    void d_perturbate(cppied_solution&) override;
    void uniform_sample(cppied_solution&,
                        std::vector<int>& sampling,
                        int k);
    int greedy_sample(cppied_solution&,
                      std::vector<int>& candidates);
    void top_6_sample(cppied_solution&,
                      std::vector<int>& candidates,
                      int ref);
    void bridge_sample(cppied_solution&,
                       std::vector<int>& candidates,
                       int ref);
    void double_bridge(cppied_solution&,
                       std::vector<int>& selection);
    void replace(cppied_solution&, std::vector<int>& sampling);
    void replace_segment(cppied_solution&,
                         std::vector<segment>& new_sol,
                         int i);
    cost_t gain(cppied_solution&, std::vector<int>& sampling, int i);

protected:
    double ratio = 0.05;
};