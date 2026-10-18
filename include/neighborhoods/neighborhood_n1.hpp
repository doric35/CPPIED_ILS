#pragma once

#include "SimpleNeighborhood.hpp"
#include "../algorithms/MaximumReplacement.hpp"

class neighborhood_n1 : public SimpleNeighborhood<ReplacementMove, MaximumReplacement> {
private:
    void swapSelectedMoves(std::vector<int>& selectedMoves);
    void swap(int i);
protected:
    void applyMoves(std::vector<int>& selectedMoves) override;
    void splitMoves(std::vector<int>& candidates) override;
public:
    using SimpleNeighborhood::SimpleNeighborhood;
};
