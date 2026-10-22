#pragma once

#include "MaximumOneTwoReallocation.hpp"
#include "../heuristics/ris_heuristic.hpp"

class RandomOneTwoReallocation : public MaximumOneTwoReallocation{
private:
    std::random_device rd;
    std::mt19937 rng;

protected:
    void setIncumbentMove() override;
    void applyNegativeGainsCorrection(std::vector<cost_t>& gains);

public:
    //----Constructors & Destructors----
    template<typename... Args>
    RandomOneTwoReallocation(Args&&... args) :
        MaximumOneTwoReallocation(std::forward<Args>(args)...),
        rd(), rng(rd()){
            insertionCostBound = cost_t{MAX_INT, MAX_INT};
            insertionAlgorithm.setBound(insertionCostBound);
        }

    //----Getters----

    //----Setters----

    //----Member Functions----

    //----Static Functions----

    //----Friend Classes----
};
