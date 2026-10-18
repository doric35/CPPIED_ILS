#pragma once

#include "MaximumMove.hpp"

struct ReplacementMove{
    segment s;
    cost_t gain;
};

class MaximumReplacement : public MaximumMove<ReplacementMove> {
private:

    void setCompleteCoverageHorizontalSegments();
    void setCompleteCoverageVerticalSegments();
protected:
    std::vector<int> unsatisfiedCells;
    iRectangle unsatisfiedBox;

    virtual void setCompleteCoverageSegments();
    void flipCandidates();
    bool isValidMove(ReplacementMove& move) override;
    cost_t getGain(ReplacementMove& move) override;

    void setUnsatisfiedCells();

    void setUnsatisfiedRectangle();

public:
    //----Constructors & Destructors----
    template <typename... Args>
    explicit MaximumReplacement(Args&&... args) :
        MaximumMove<ReplacementMove>(std::forward<Args>(args)...),
        unsatisfiedCells(), unsatisfiedBox(){
        incumbent.s = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    }

    ~MaximumReplacement() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    ReplacementMove solve() override;

    //----Static Functions----

    //----Friend Classes----
};
