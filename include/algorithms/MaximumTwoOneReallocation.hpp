#pragma once

#include "MaximumMove.hpp"
#include "MinimumSegmentAllocation.hpp"

struct TwoOneReallocationMove{
    segment replacement;
    int reallocationIndex;
    int removalIndex;
    cost_t gain;
};

class MaximumTwoOneReallocation : public MaximumMove<TwoOneReallocationMove> {
private:
    cost_t removalGain;
    segment firstRemoved;
    iRectangle box;
protected:
    void setRemovalCandidates();
    void setCandidateBox();
    bool isValidCandidate(segment& s);
    void setCandidateBoxFromHorizontalFirst();
    void setCandidateBoxFromVerticalFirst();
    void correctIncumbentIndexes();
    bool isValidMove(TwoOneReallocationMove& move) override{ return true; };
    cost_t getGain(TwoOneReallocationMove& move) override;
public:
    //----Constructors & Destructors----
    template <typename... Args>
    explicit MaximumTwoOneReallocation(Args&&... args) :
    MaximumMove<TwoOneReallocationMove>(std::forward<Args>(args)...),
    removalGain(), box(){
        incumbent.replacement = {path_engine::NULL_NODE, path_engine::NULL_NODE};
        removalGain = geometry.removal_gain(solution, solution.path.begin() + index);
        firstRemoved = solution.path[index];
    }

    ~MaximumTwoOneReallocation() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    TwoOneReallocationMove solve() override;

    //----Static Functions----

    //----Friend Classes----
};
