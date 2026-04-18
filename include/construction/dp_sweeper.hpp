#pragma once

#include "construction.hpp"
#include "../neighborhoods/neighborhood_tsp.hpp"

namespace sweeper{
    struct segment_set{
        std::vector<std::pair<segment, double>> S;
        double gain;
        bool horizontal;
    };
}

class dp_sweeper: public construction{
protected:
    double mPenalty=0.01;

    std::function<int(const std::array<sweeper::segment_set,2>&)> mChoice =
            [](const std::array<sweeper::segment_set,2>&){
        return -1;
    };

public:
    using construction::construction;

    void construct(cppied_solution&,
                   std::function<void(cppied_solution&)> save_history) override;

    std::pair<segment,double> maximum_subarray(cppied_solution&, const segment& ref);
    void filter_segment_set(cppied_solution&, sweeper::segment_set& S);
    void set_choice(std::function<int(const std::array<sweeper::segment_set,2>&)> choice) {
        mChoice = std::move(choice);
    }
    double get_penalty(){return mPenalty;}
    void select(cppied_solution&,
                std::array<sweeper::segment_set,2>& S);
};