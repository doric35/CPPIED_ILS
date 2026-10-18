#pragma once

#include "SimpleNeighborhood.hpp"
#include "../algorithms/MaximumOneTwoReallocation.hpp"
#include "../heuristics/RandomKeyPacking.hpp"

class neighborhood_n12 : public SimpleNeighborhood<OneTwoReallocationMove, MaximumOneTwoReallocation>{
private:
    std::vector<std::pair<segment, int>> orderedMoves{};

    void swapSelectedMoves(std::vector<int>& selectedMoves);
    void removeSelectedMovesCoverage(std::vector<int>& selectedMoves);
    void addSelectedMovesCoverage(std::vector<int>& selectedMoves);
    void setOrderedMoves(std::vector<int> &selectedMoves);
    void computeMarginalGain(std::vector<int> &selectedMoves);
    std::vector<Key> constructPackingKeys(std::vector<int>& candidates);
    void splitPackings(std::vector<int> &candidates, std::vector<Packing>& packing);
protected:
    void clear() override { SimpleNeighborhood::clear(); orderedMoves.clear(); }
    void applyMoves(std::vector<int>& selectedMoves) override;
    void splitMoves(std::vector<int>& candidates) override;
public:
    using SimpleNeighborhood::SimpleNeighborhood;
};