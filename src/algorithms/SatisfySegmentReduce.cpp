#include "../../include/algorithms/SatisfySegmentReduce.hpp"

MoveReduce SatisfySegmentReduce::solve() {
    MoveReduce move{solution.path[index], cost_t{0,0}};
    if (isValidMove(move))
        setIncumbentMove();
    return incumbent;
}

void SatisfySegmentReduce::setIncumbentMove() {
    incumbent.s = solution.path[index];
    incumbent.gain = getGain(incumbent);
}