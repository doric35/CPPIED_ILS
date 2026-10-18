#pragma once

#include "MaximumMove.hpp"

struct MoveReduce{
    segment s;
    cost_t gain;
};

class SatisfySegmentReduce : public MaximumMove<MoveReduce>{
private:
    void setIncumbentMove() override;
protected:
    bool isValidMove(MoveReduce& move) override{ return coverage.can_remove(solution, solution.path[index]); }
    cost_t getGain(MoveReduce& move) override {
        cost_t gain = geometry.removal_gain(solution, solution.path.begin() + index);
        if (gain <= cost_t{0,0})
            gain += cost_t{0,1};
        return gain;
    }

public:
    //----Constructors & Destructors----
    template<typename... Args>
    SatisfySegmentReduce(Args&&... args) : MaximumMove<MoveReduce>(std::forward<Args>(args)...){}
    ~SatisfySegmentReduce() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    MoveReduce solve() override;

    //----Static Functions----

    //----Friend Classes----
};
