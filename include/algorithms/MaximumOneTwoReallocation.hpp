#pragma once

#include "MaximumMove.hpp"
#include "MinimumSegmentPairInsertion.hpp"

//This algorithm is at a heuristic stage. Returned pair is not the true optimal pair.

struct OneTwoReallocationMove{
    std::array<segment, 2> replacements;
    std::array<int, 2> reallocationIndexes;
    cost_t gain;
};

class MaximumOneTwoReallocation : public MaximumMove<OneTwoReallocationMove>{
private:
    cost_t removalGain;
    cost_t bound;
    std::unordered_set<SegmentPair, HashSegmentPair> processedPairs;
    std::unordered_map<std::uint64_t, std::array<std::pair<int,cost_t>, 4>> memoizedTopInsertions;

    std::vector<int> getUnsatisfiedCells();
    iRectangle getUnsatisfiedRectangle(std::vector<int>& unsatisfiedCells);

    std::vector<segment> getPartialCoverageSegments(iRectangle& box);
    std::vector<segment> getPartialCoverageHorizontalSegments(iRectangle& box);
    std::vector<segment> getPartialCoverageVerticalSegments(iRectangle& box);

    void setCandidates(std::vector<int>& unsatisfiedCells);
    std::vector<segment> getReplacements(std::vector<int>& unsatisfiedCells);
    void setSecondLevelCandidates(std::vector<int> unsatisfiedCells, segment& firstLevel);
    void filterCandidates(std::vector<int>& unsatisfiedCells);
    void trimCandidates();

    void setCandidatesTopInsertions();
    void setMoveTopInsertion(OneTwoReallocationMove& move);

    void setSimpleRemovalIncumbent();

    segment trimSegment(segment& s);

    void setIncumbentMove() override;
protected:
    cost_t insertionCostBound;
    virtual void setCompleteCoverageSegments(){}
    bool isValidMove(OneTwoReallocationMove& move) override{ return false; }
    void correctIncumbentIndexInsertion();
    cost_t getGain(OneTwoReallocationMove& move) override{ return cost_t{0,0}; }
public:
    //----Constructors & Destructors----
    template <typename... Args>
    explicit MaximumOneTwoReallocation(Args&&... args) :
    MaximumMove<OneTwoReallocationMove>(std::forward<Args>(args)...),
    removalGain(), bound(), insertionCostBound(), processedPairs(), memoizedTopInsertions(){
        incumbent.replacements[0] = {path_engine::NULL_NODE, path_engine::NULL_NODE};
        incumbent.replacements[1] = {path_engine::NULL_NODE, path_engine::NULL_NODE};
        removalGain = geometry.removal_gain(solution, solution.path.begin() + index);
        insertionCostBound = removalGain;
    }

    ~MaximumOneTwoReallocation() = default;

    //----Getters----

    //----Setters----

    //----Member Functions----
    OneTwoReallocationMove solve() override;

    //----Static Functions----

    //----Friend Classes----
};
