#pragma once

#include "../algorithms/SatisfySegmentReduce.hpp"
#include "SimpleNeighborhood.hpp"

class neighborhood_reduce : public SimpleNeighborhood<MoveReduce, SatisfySegmentReduce>{
private:
protected:
    void removeSelectedMoves(std::vector<int> &selectedMoves);

    void applyMoves(std::vector<int>& selectedMoves) override;
    void splitMoves(std::vector<int>& candidates) override;
public:
    using SimpleNeighborhood::SimpleNeighborhood;
};