#include "../../include/neighborhoods/neighborhood_cut.hpp"

void neighborhood_cut::splitMoves(std::vector<int> &candidates) {
    splitHorizontalVerticalMoves(candidates);
}

void neighborhood_cut::applyMoves(std::vector<int> &selectedMoves) {
    if (selectedMoves.empty()) return;
    std::sort(selectedMoves.begin(), selectedMoves.end());
    cutSelectedMoves(selectedMoves);
    solution.cost = geometry.cost(solution);
}

void neighborhood_cut::cutSelectedMoves(std::vector<int> &selectedMoves) {
    std::list<segment> path;
    int begin = 0;
    for (int i : selectedMoves) {
        path.insert(path.end(), solution.path.begin() + begin, solution.path.begin() + i);
        std::array<segment, 2> segments = cut(i);
        pushSegments(path, segments);
        begin = i + 1;
    }
    path.insert(path.end(), solution.path.begin() + begin, solution.path.end());
    solution.path.clear();
    solution.path.assign(path.begin(), path.end());
}

std::array<segment, 2> neighborhood_cut::cut(int i) {
    coverage.remove(solution, candidatesMoves[i].s);
    return cutDiff(candidatesMoves[i].s, i);
}

void neighborhood_cut::pushSegments(std::list<segment> &path, std::array<segment, 2> &replacements) {
    if (path_engine::is_node(replacements[0].source))
        path.push_back(replacements[0]);
    if (path_engine::is_node(replacements[1].source))
        path.push_back(replacements[1]);
}

std::array<segment, 2> neighborhood_cut::cutDiff(segment &cut, int i) {
    std::array<segment,2> replacement{{{path_engine::NULL_NODE, path_engine::NULL_NODE},
                            {path_engine::NULL_NODE, path_engine::NULL_NODE}}};

    segment non_oriented_source = geometry.getNonOriented(solution.path[i]);
    segment non_oriented_cut = geometry.getNonOriented(cut);

    replacement[0] = getFirstCut(non_oriented_cut, non_oriented_source);
    replacement[1] = getSecondCut(non_oriented_cut, non_oriented_source);

    direction d = geometry.get_direction(solution.path[i]);

    if (path_engine::is_node(replacement[0].source))
        geometry.correct_segment_direction(replacement[0], d);
    if (path_engine::is_node(replacement[1].source))
        geometry.correct_segment_direction(replacement[1], d);

    if (!path_engine::is_node(replacement[0].source) || replacement[0].source != solution.path[i].source)
        std::swap(replacement[0], replacement[1]);

    return replacement;
}

segment neighborhood_cut::getFirstCut(segment& nonOrientedCut, segment& nonOrientedSegment) {
    segment first_cut = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    if (nonOrientedCut.source != nonOrientedSegment.source){
        first_cut.source = nonOrientedSegment.source;
        if (nonOrientedCut.source != nonOrientedSegment.source + 1)
            first_cut.target = nonOrientedCut.source - 1;
    }
    return first_cut;
}

segment neighborhood_cut::getSecondCut(segment& nonOrientedCut, segment& nonOrientedSegment) {
    segment second_cut = {path_engine::NULL_NODE, path_engine::NULL_NODE};
    if (nonOrientedCut.source != nonOrientedSegment.target && nonOrientedCut.target != nonOrientedSegment.target){
        second_cut.source = std::max(nonOrientedCut.source, nonOrientedCut.target) + 1;
        if (second_cut.source != nonOrientedSegment.target)
            second_cut.target = nonOrientedSegment.target;
    }
    return second_cut;
}
