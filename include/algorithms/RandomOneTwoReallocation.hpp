#pragma once

#include "MaximumOneTwoReallocation.hpp"
#include "../heuristics/ris_heuristic.hpp"

class RandomOneTwoReallocation : public MaximumOneTwoReallocation{
private:
    std::random_device rd;
    std::mt19937 rng;

protected:
    void setIncumbentMove() override;

public:
    //----Constructors & Destructors----
    template<typename... Args>
    RandomOneTwoReallocation(Args&&... args) :
        MaximumOneTwoReallocation(std::forward<Args>(args)...),
        rd(), rng(rd()){
            insertionCostBound = cost_t{MAX_INT, MAX_INT};
        }

    //----Getters----

    //----Setters----

    //----Member Functions----

    //----Static Functions----

    //----Friend Classes----
};
