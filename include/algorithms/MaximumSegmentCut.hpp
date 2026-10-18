#pragma once

#include "MaximumMove.hpp"

struct CutMove{
    segment s;
    cost_t gain;
};

class MaximumSegmentCut : public MaximumMove<CutMove>{
private:
    segment nonOrientedSegment;
    std::vector<cost_t> cutValues;
    void setCutValues();
    void setNonOrientedCutValues();
    void setIncumbentMove() override;
protected:
    bool isValidMove(CutMove& move) override{ return true; }
    cost_t getGain(CutMove& move) override { return geometry.cost(move.s) + cost_t{1,0}; }
public:
    //----Constructors & Destructors----
    template<typename... Args>
    explicit MaximumSegmentCut(Args&&... args) : MaximumMove<CutMove>(std::forward<Args>(args)...),
            cutValues(), nonOrientedSegment(solution.path[index]){
        incumbent.s = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    }
    ~MaximumSegmentCut() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    CutMove solve() override;

    //----Static Functions----

    //----Friend Classes----
};
