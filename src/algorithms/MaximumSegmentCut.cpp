#include "../../include/algorithms/MaximumSegmentCut.hpp"

CutMove MaximumSegmentCut::solve() {
    setCutValues();
    setIncumbentMove();
    return incumbent;
}

void MaximumSegmentCut::setCutValues() {
    const segment& s = solution.path[index];
    const int n = geometry.cost(s).length + 1;
    cutValues.resize(n);
    if (n==1){
        cutValues[0] = {static_cast<int>(coverage.can_remove(solution, solution.path[index])), 0};
    } else {
        nonOrientedSegment = geometry.getNonOriented(solution.path[index]);
        setNonOrientedCutValues();

    }
}

void MaximumSegmentCut::setNonOrientedCutValues() {
    cutValues[0] = {static_cast<int>(coverage.can_remove_vertex(solution, nonOrientedSegment.source)), 0};
    for (int u = nonOrientedSegment.source + 1; u <= nonOrientedSegment.target; ++u){
        int removable_node = static_cast<int>(coverage.can_remove_vertex(solution, u));
        cutValues[u - nonOrientedSegment.source] = {
                (removable_node * cutValues[u - nonOrientedSegment.source -1].length) + removable_node, 0};
    }
}

void MaximumSegmentCut::setIncumbentMove() {
    auto argmax = std::max_element(cutValues.begin(), cutValues.end());
    if (argmax->length <= 0) return;
    auto first = std::prev(argmax, argmax->length - 1);
    int u = std::distance(cutValues.begin(), first);
    int v = std::distance(cutValues.begin(), argmax);
    incumbent = {{u + nonOrientedSegment.source, v + nonOrientedSegment.source}, *argmax};
    if (argmax->length == 1) incumbent.s.target = path_engine::NULL_NODE;
    geometry.correct_segment_direction(incumbent.s, geometry.get_direction(solution.path[index]));
}