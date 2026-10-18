#pragma once

#include "SimpleNeighborhood.hpp"
#include "../algorithms/MaximumSegmentCut.hpp"

class neighborhood_cut : public SimpleNeighborhood<CutMove, MaximumSegmentCut>{
private:
    void pushSegments(std::list<segment>& path, std::array<segment, 2>& replacements);
    segment getFirstCut(segment& nonOrientedCut, segment& nonOrientedSegment);
    segment getSecondCut(segment& nonOrientedCut, segment& nonOrientedSegment);
protected:
    void applyMoves(std::vector<int>& selectedMoves) override;
    void cutSelectedMoves(std::vector<int> &selectedMoves);
    void splitMoves(std::vector<int>& candidates) override;
    std::array<segment, 2> cut(int i);
    std::array<segment, 2> cutDiff(segment &cut, int i);
public:
    using SimpleNeighborhood::SimpleNeighborhood;
};