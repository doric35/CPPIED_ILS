#pragma once

#include "SimpleNeighborhood.hpp"
#include "../algorithms/MaximumSegmentTrim.hpp"

class neighborhood_trim : public SimpleNeighborhood<ReplacementMove, MaximumSegmentTrim>{
private:
    void swapSelectedMoves(std::vector<int>& selectedMoves);
    void swap(int i);
protected:
    void applyMoves(std::vector<int>& selectedMoves) override;
    void splitMoves(std::vector<int>& candidates) override;
public:
    using SimpleNeighborhood::SimpleNeighborhood;
};