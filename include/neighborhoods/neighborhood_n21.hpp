#pragma once

#include "SimpleNeighborhood.hpp"
#include "../algorithms/MaximumTwoOneReallocation.hpp"
#include "../heuristics/RandomKeyPacking.hpp"


class neighborhood_n21 : public SimpleNeighborhood<TwoOneReallocationMove, MaximumTwoOneReallocation>{
private:
    std::vector<std::pair<segment, int>> orderedMoves{};
    std::vector<int> pairedRemovals{};
    std::vector<int> selectedMoves{};

    void swapSelectedMoves();
    void removeSelectedMovesCoverage();
    void addSelectedMovesCoverage();
    void setOrderedMoves();
    void computeMarginalGain();
    std::vector<Key> constructPackingKeys(std::vector<int>& candidates);
    void splitPackings(std::vector<int> &candidates, std::vector<Packing>& packing);

    bool isSelectedOrRemoved(int index,
                             std::vector<int>::const_iterator selector_iterator,
                             std::vector<int>::const_iterator removal_iterator);
    bool isSelected(int index, std::vector<int>::const_iterator selector_iterator);
    bool isRemoved(int index, std::vector<int>::const_iterator removal_iterator);
    bool isMoveToInsert(int index, std::vector<std::pair<segment,int>>::const_iterator move_iterator);
    void pushRemainingMoves(std::vector<std::pair<segment,int>>::iterator move_iterator);
protected:
    void clear() override { SimpleNeighborhood::clear(); orderedMoves.clear(); pairedRemovals.clear(); }
    void applyMoves(std::vector<int>& selectedMoves) override;
    void splitMoves(std::vector<int>& candidates) override;
    Interval transformToInterval(int i) override;
    Interval correctIntervalEncoding(Interval& interval);
    Interval getSecondInterval(Interval& first, int second);
    Interval getSecondPerpendicularInterval(Interval& first, int second);
    Interval getIntervalsUnion(Interval& first, Interval& second);
public:
    template<typename... Args>
    explicit neighborhood_n21(Args&&... args) : SimpleNeighborhood<TwoOneReallocationMove, MaximumTwoOneReallocation>(std::forward<Args>(args)...){
        spacingConstraint = 6;
    }
};